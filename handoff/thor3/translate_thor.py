#!/usr/bin/env python3
"""Traduz para o português do Brasil os exemplos mantidos (kept.jsonl) do professor, com o próprio professor, no modo sem raciocínio.

Para cada exemplo traduz:
- o enunciado e as alternativas (só múltipla escolha; os de matemática já têm o enunciado em português);
- o raciocínio, em blocos de parágrafos, só nos 70% de problemas com raciocínio (hash do id, a mesma divisão do treino);
  os outros 30% entram no treino sem raciocínio e não precisam dele traduzido;
- o texto final antes da linha `Resposta: X` / `#### N` (se ainda não estiver em português).
Cada pedaço é conferido: terminou, não trouxe `<think>`, tamanho compatível com o original, números preservados e não "respondeu
a questão" em vez de traduzir. Um pedaço ruim derruba o exemplo (com o motivo em `verdict`). O original em inglês fica no
registro, para o par inglês x português do outro lado.

Só usa a biblioteca padrão e o tokenizer do `transformers` (para o template de chat). Fala com POST /v1/completions.
Retomável: ids já gravados em --out são pulados. Saída: --out (um registro por exemplo) e <out>.report.json.
Uso:
  python translate_thor.py --kept ../thor2/out/kept.jsonl --kept out/kept.jsonl --out out/translated.jsonl \
      --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --model-name teacher --concurrency 96
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
INSTR = ("Traduza para o português do Brasil o texto entre <texto> e </texto>. O texto é um trecho do raciocínio de outra pessoa "
         "(ou uma questão com alternativas): não resolva, não responda, não corrija e não comente; traduza tudo, frase por frase, "
         "na mesma ordem. Preserve os números, as fórmulas, o LaTeX, os símbolos, as letras das alternativas e as quebras de linha. "
         "Escreva só a tradução, sem as marcas <texto>.\n\n<texto>\n")
MIN_RATIO, MAX_RATIO = 0.8, 1.7  # caracteres da tradução / do original, para trechos de 200+ caracteres
SHORT_MIN, SHORT_MAX = 0.4, 2.6  # trechos curtos variam mais (uma palavra pode virar três)
BACKOFF_S = 2

# Heurística de idioma (cópia de lumi_ft.eval.shape.trace_language, para o pacote não depender do repositório)
_PT = frozenset("de que não uma para com os as na no por mais mas foi ao ele das tem seu sua ou ser quando muito nos já "
                "está também só pelo pela até isso ela entre era depois sem mesmo ter quem nas esse eles você então "
                "cada porque são vezes temos".split())
_EN = frozenset("the of and to is that it was for are with his they be at have this from or had by but what there we can "
                "were your when which their if will about then them would so these her has he you i each first".split())
_WORD = re.compile(r"[a-zà-ÿ]+")
_DIGITS = re.compile(r"\d+")
_ANSWERISH = re.compile(r"(\\boxed\{|Resposta:|####|\bAnswer:)")
_MARKER_MCQ = re.compile(r"^[\s\*>`]*Resposta:\s*\**\s*([A-J])\s*[\.\*`]*\s*$")
_MARKER_MATH = re.compile(r"^\s*####\s")


def trace_language(text: str, min_hits: int = 3, ratio: float = 2.0) -> str:
    words = _WORD.findall(text.lower())
    pt, en = sum(w in _PT for w in words), sum(w in _EN for w in words)
    if pt + en < min_hits:
        return "unk"
    if pt >= ratio * en:
        return "pt"
    if en >= ratio * pt:
        return "en"
    return "mixed"


def bucket(example_id: str, n: int = 10_000) -> int:
    """Igual a lumi_ft.data.gigaverbo.bucket: < 7.000 = problema com raciocínio no treino."""
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:12], 16) % n


def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    """Junta parágrafos (linha em branco entre eles) em blocos de até `max_chars`; um parágrafo maior fica sozinho."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + 2 + len(para) > max_chars:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def digit_runs(text: str) -> Counter:
    """Sequências de dígitos (o separador decimal e o de milhar mudam de idioma para idioma; os dígitos, não)."""
    return Counter(_DIGITS.findall(text))


def check_piece(src: str, out: str, finished: bool) -> str | None:
    """None se a tradução do pedaço serve; senão o motivo."""
    out = out.strip()
    if not finished:
        return "nao_terminou"
    if not out:
        return "vazio"
    if "<think>" in out or "</think>" in out:
        return "marcador_de_raciocinio"
    r = len(out) / max(1, len(src.strip()))
    lo, hi = (MIN_RATIO, MAX_RATIO) if len(src.strip()) >= 200 else (SHORT_MIN, SHORT_MAX)
    if r < lo or r > hi:
        return "tamanho_incompativel"
    if _ANSWERISH.search(out) and not _ANSWERISH.search(src):
        return "respondeu_em_vez_de_traduzir"
    want = digit_runs(src)
    if want:
        missing = sum((want - digit_runs(out)).values())
        if missing > max(1, int(0.1 * sum(want.values()))):
            return "numeros_divergentes"
    return None


def clean_output(text: str) -> str:
    """Tira o que vier antes de um `</think>` (se o modelo pensar mesmo assim) e as marcas <texto>."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<texto>", "").replace("</texto>", "").strip()


def question_block(q: str, options: list[str]) -> str:
    return q.strip() + "\n" + "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))


def split_question_block(text: str, n: int) -> tuple[str, list[str]] | None:
    """Inverso de `question_block`: (enunciado, alternativas) se as `n` letras voltarem em ordem, cada uma começando linha."""
    starts = []
    pos = 0
    for i in range(n):
        m = re.compile(rf"(?m)^[ \t]*{LETTERS[i]}\)[ \t]?").search(text, pos)
        if not m:
            return None
        starts.append(m)
        pos = m.end()
    if n < len(LETTERS) and re.search(rf"(?m)^[ \t]*{LETTERS[n]}\)[ \t]", text[pos:]):
        return None  # o tradutor inventou uma alternativa a mais
    stem = text[: starts[0].start()].strip()
    opts = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < n else len(text)
        opts.append(" ".join(text[m.end(): end].split()))
    if not stem or not all(opts):
        return None
    return stem, opts


def split_final(kind: str, answer: str) -> tuple[str, str]:
    """(corpo, linha final) da resposta: a linha `Resposta: X` (múltipla escolha) ou a primeira `#### N` (matemática)."""
    lines = answer.strip().splitlines()
    if kind == "mcq":
        return "\n".join(lines[:-1]).strip(), lines[-1].strip()
    for i, ln in enumerate(lines):
        if _MARKER_MATH.match(ln):
            return "\n".join(lines[:i]).strip(), ln.strip()
    return answer.strip(), ""


def render(tok, text: str) -> str:
    """Prompt sem raciocínio (enable_thinking=False). Se o template não fechar o raciocínio sozinho, fecha aqui."""
    msgs = [{"role": "user", "content": INSTR + text + "\n</texto>"}]
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in out[-40:]:
        out = out.rstrip("\n").removesuffix("<think>") + "<think></think>"
    return out


def post(endpoint: str, payload: dict, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def translate_piece(src: str, tok, a, post_fn=post) -> tuple[str, bool, int, str | None]:
    """(tradução, terminou, tokens gerados, erro de requisição)."""
    payload = {"model": a.model_name, "prompt": render(tok, src), "max_tokens": min(a.max_tokens, len(src) // 2 + 256),
               "temperature": 0.0, "top_p": 1.0, "stop": ["<|im_end|>"]}
    last = None
    for attempt in range(3):
        try:
            r = post_fn(a.endpoint, payload, a.api_key, a.timeout)
            c = r["choices"][0]
            return clean_output(c["text"]), c.get("finish_reason") == "stop", (r.get("usage") or {}).get("completion_tokens") or 0, None
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
            last = repr(e)[:200]
            time.sleep(BACKOFF_S * (attempt + 1))
    return "", False, 0, last


def jobs_for(rec: dict, think_frac: float, max_chars: int) -> list[tuple[str, str]]:
    """Pedaços a traduzir de um exemplo: [(nome, texto)]. `q` = enunciado+alternativas, `tN` = blocos do raciocínio, `f` = final."""
    jobs = []
    if rec["kind"] == "mcq":
        jobs.append(("q", question_block(rec["question"], rec["options"])))
    if bucket(rec["id"]) < think_frac * 10_000:
        jobs += [(f"t{k}", c) for k, c in enumerate(chunk_paragraphs(rec["thought"], max_chars))]
    body, _ = split_final(rec["kind"], rec["answer"])
    if body and trace_language(body) != "pt":
        jobs.append(("f", body))
    return jobs


def assemble(rec: dict, parts: dict) -> dict:
    """Registro final: original em inglês + tradução, ou o motivo da recusa em `verdict`."""
    out = {"id": rec["id"], "kind": rec["kind"], "source": rec["source"], "ref": rec["ref"], "with_thought": any(k.startswith("t") for k in parts),
           "question_en": rec["question"], "options_en": rec.get("options"), "thought_en": rec["thought"], "answer_en": rec["answer"],
           "completion_tokens": rec.get("completion_tokens")}
    bad = next((f"{k}:{v['why']}" for k, v in parts.items() if v["why"]), None)
    if bad:
        return {**out, "verdict": bad, "pieces": {k: v["text"][:400] for k, v in parts.items()}}
    if rec["kind"] == "mcq":
        sq = split_question_block(parts["q"]["text"], len(rec["options"]))
        if sq is None:
            return {**out, "verdict": "q:alternativas_nao_voltaram", "pieces": {"q": parts["q"]["text"][:400]}}
        out["question_pt"], out["options_pt"] = sq
    else:
        out["question_pt"], out["options_pt"] = rec["question"], None
    body, last = split_final(rec["kind"], rec["answer"])
    body_pt = parts["f"]["text"] if "f" in parts else body
    if rec["kind"] == "mcq":
        m = _MARKER_MCQ.match(last)
        last = f"Resposta: {m.group(1)}" if m else last
    out["answer_pt"] = (body_pt + "\n\n" + last).strip() if last else body_pt
    if out["with_thought"]:
        keys = sorted((k for k in parts if k.startswith("t")), key=lambda k: int(k[1:]))
        thought = "\n\n".join(parts[k]["text"].strip() for k in keys)
        if trace_language(thought) != "pt":
            return {**out, "verdict": "t:raciocinio_nao_pt", "pieces": {k: parts[k]["text"][:400] for k in keys[:2]}}
        out["thought_pt"] = thought
    out["verdict"] = "ok"
    return out


def run(a, tok, post_fn=post) -> dict:
    recs, seen = [], set()
    for path in a.kept:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    recs.append(r)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(recs)} exemplos, {len(done)} já feitos, {len(todo)} a fazer", flush=True)
    stats = Counter()
    t0, ntok = time.time(), 0
    deadline = t0 + a.time_budget_min * 60 if a.time_budget_min else None
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, out.open("a") as fout:
        for i in range(0, len(todo), a.block):
            if deadline and time.time() > deadline:
                print("tempo esgotado; o que já foi gravado fica em", out, flush=True)
                break
            block = todo[i: i + a.block]
            futs = {}
            for r in block:
                for name, src in jobs_for(r, a.think_frac, a.max_chars):
                    futs[(r["id"], name)] = (src, ex.submit(translate_piece, src, tok, a, post_fn))
            for r in block:
                parts = {}
                for (rid, name), (src, f) in futs.items():
                    if rid != r["id"]:
                        continue
                    text, fin, n, err = f.result()
                    ntok += n
                    parts[name] = {"text": text, "why": "erro_de_requisicao" if err else check_piece(src, text, fin)}
                rec = assemble(r, parts)
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                stats[f"{r['kind']}:{rec['verdict'].split(':')[-1]}"] += 1
            el = time.time() - t0
            print(f"[{el / 60:5.1f} min] {i + len(block)}/{len(todo)} exemplos | {ntok / el:.0f} tokens/s | {dict(stats)}", flush=True)
    rep_path = Path(str(out) + ".report.json")
    allrecs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    rep = {"exemplos": len(allrecs), "por_veredito": dict(Counter(f"{r['kind']}:{r['verdict'].split(':')[-1]}" for r in allrecs)),
           "por_fonte_ok": dict(Counter(r["source"] for r in allrecs if r["verdict"] == "ok")),
           "com_raciocinio_ok": sum(r["verdict"] == "ok" and r["with_thought"] for r in allrecs),
           "aproveitamento": round(sum(r["verdict"] == "ok" for r in allrecs) / max(1, len(allrecs)), 4),
           "esta_execucao": {"minutos": round((time.time() - t0) / 60, 1), "tokens_gerados": ntok}}
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", action="append", required=True, help="kept.jsonl do filter_traces_en.py (pode repetir)")
    ap.add_argument("--out", default="out/translated.jsonl")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--block", type=int, default=256, help="exemplos por bloco (grava ao fim de cada bloco)")
    ap.add_argument("--max-chars", type=int, default=3500, help="tamanho máximo de um bloco do raciocínio")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--think-frac", type=float, default=0.7, help="fração dos problemas com o raciocínio traduzido")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--show-template", action="store_true", help="só mostra o prompt renderizado de um exemplo e sai")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    if a.show_template:
        print(repr(render(tok, "Okay, so we need to compute 17 x 23.")))
        return 0
    rep = run(a, tok)
    return 0 if rep["exemplos"] else 1


if __name__ == "__main__":
    sys.exit(main())

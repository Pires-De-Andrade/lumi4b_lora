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
import os
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
PREFILL = "Tradução em português do Brasil:\n\n"  # [v2]
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")  # [v2]
_SUPERS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")  # [v2]
_SUPER_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")  # [v2]
# [v2] exemplos de tradução (inclusive das frases de comentário sobre o formato, que o modelo costuma copiar ou omitir); só na reserva
SHOTS = [
    ("So line should be `#### 12`. Ensure no extra spaces? Provide reasoning before that. Ensure final answer line is only that.",
     "Então a linha deve ser `#### 12`. Garanta que não haja espaços extras? Apresente o raciocínio antes disso. Garanta que a linha final seja só essa."),
    ("Thus the final answer: \"Resposta: C\".\n\nWe need to show reasoning and end with that line. Probably fine.",
     "Portanto, a resposta final: \"Resposta: C\".\n\nPrecisamos mostrar o raciocínio e terminar com essa linha. Provavelmente está bom."),
    ("Let's compute: 12 * 7 = 84. So total = 84. Then 320 - 84 = 236.",
     "Vamos calcular: 12 * 7 = 84. Então o total = 84. Depois, 320 - 84 = 236."),
    ("We need to answer: \"Which gas is most abundant?\" Options: A) Oxygen, B) Nitrogen. So answer B.",
     "Precisamos responder: \"Qual gás é o mais abundante?\" Opções: A) Oxigênio, B) Nitrogênio. Então a resposta é B."),
]

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
    text = text.replace("<texto>", "").replace("</texto>", "")
    # [v2] limpeza: espaços finos/inseparáveis e hífen inseparável viram os normais; (subscritos e sobrescritos: ver fix_scripts)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ").replace("‑", "-").translate(_SUBSCRIPTS)
    text = re.sub(r"^\s*Tradução( em português do Brasil)?:[ \t]*\n*", "", text)  # [v2] se o modelo repetir a marca
    return text.strip()


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


def render(tok, text: str, fewshot: bool = False) -> str:
    """Prompt sem raciocínio (enable_thinking=False). Se o template não fechar o raciocínio sozinho, fecha aqui."""
    msgs = []
    if fewshot:  # [v2]
        for en, pt in SHOTS:
            msgs += [{"role": "user", "content": INSTR + en + "\n</texto>"}, {"role": "assistant", "content": PREFILL + pt}]
    msgs.append({"role": "user", "content": INSTR + text + "\n</texto>"})
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in out[-40:]:
        out = out.rstrip("\n").removesuffix("<think>") + "<think></think>"
    return out + PREFILL  # [v2] abre a resposta do assistente com a marca de tradução (sem ela o modelo responde ou copia o inglês)


def post(endpoint: str, payload: dict, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fix_scripts(src: str, text: str) -> str:
    """[v2] Se o modelo trocou dígitos comuns por subscritos/sobrescritos que o original não tinha (O2 -> O₂, r^2 -> r²), volta ao estilo do original."""
    if not re.search(r"[₀-₉]", src):
        text = text.translate(_SUBSCRIPTS)
    if not _SUPER_RE.search(src):
        text = _SUPER_RE.sub(lambda m: "^" + m.group(0).translate(_SUPERS), text)
    if "```" not in src:  # [v2] o modelo às vezes embrulha a tradução em cerca de código
        m = re.fullmatch(r"```[A-Za-z]*\n(.*?)\n?```", text.strip(), re.S)
        if m:
            text = m.group(1)
    return text


def lang_bad(src: str, out: str, min_chars: int = 150) -> bool:
    """[v2] O modelo deixou o texto (ou parte dele) em inglês: tradução longa que o detector de idioma não vê como português."""
    return len(src.strip()) >= min_chars and trace_language(out) in ("en", "mixed")


USAGE = {"req": 0, "prompt_tokens": 0}  # [v2] custo no servidor
_USAGE_LOCK = threading.Lock()


def translate_piece(src: str, tok, a, post_fn=post, fewshot: bool = False) -> tuple[str, bool, int, str | None]:
    """(tradução, terminou, tokens gerados, erro de requisição)."""
    payload = {"model": a.model_name, "prompt": render(tok, src, fewshot), "max_tokens": min(a.max_tokens, len(src) // 2 + 256),
               "temperature": 0.0, "top_p": 1.0, "stop": ["<|im_end|>"]}
    last = None
    for attempt in range(3):
        try:
            r = post_fn(a.endpoint, payload, a.api_key, a.timeout)
            c = r["choices"][0]
            with _USAGE_LOCK:
                USAGE["req"] += 1
                USAGE["prompt_tokens"] += (r.get("usage") or {}).get("prompt_tokens") or 0
            return fix_scripts(src, clean_output(c["text"])), c.get("finish_reason") == "stop", (r.get("usage") or {}).get("completion_tokens") or 0, None
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
            last = repr(e)[:200]
            time.sleep(BACKOFF_S * (attempt + 1))
    return "", False, 0, last


def _sents(t: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]


_SHOT_PAIRS = {}  # [v2] frase em português de um exemplo -> frase em inglês correspondente
for _en, _pt in SHOTS:
    _pe, _pp = _sents(_en), _sents(_pt)
    if len(_pe) == len(_pp):
        _SHOT_PAIRS.update(dict(zip(_pp, _pe)))


def leaked(src: str, out: str) -> bool:
    """[v2] O modelo copiou para a tradução uma frase de um exemplo (few-shot) cuja frase em inglês não está no original."""
    return any(pt in out and en not in src for pt, en in _SHOT_PAIRS.items())


def split_units(text: str, max_chars: int = 250) -> list[tuple[str, str]]:
    """[v2] Quebra em frases (e linhas) agrupadas em até `max_chars`, guardando o separador que vem depois de cada unidade."""
    toks = re.split(r"(\n+|(?<=[.!?])[ \t]+)", text)
    pieces = [(toks[i], toks[i + 1] if i + 1 < len(toks) else "") for i in range(0, len(toks), 2)]
    out, buf, sep = [], "", ""
    for t, s in pieces:
        if not t.strip():
            sep = sep + t + s if buf else sep
            continue
        if buf and len(buf) + len(sep) + len(t) > max_chars:
            out.append((buf, sep))
            buf = t
        else:
            buf = buf + sep + t if buf else t
        sep = s
    if buf:
        out.append((buf, ""))
    return out


def translate_robust(src: str, tok, a, post_fn=post) -> tuple[str, bool, int, str | None]:
    """[v2] Traduz o pedaço inteiro; se o verificador (o mesmo, sem mudar limites) recusar, traduz frase por frase e junta."""
    text, fin, n, err = translate_piece(src, tok, a, post_fn)
    if err or (check_piece(src, text, fin) is None and not lang_bad(src, text)):
        return text, fin, n, err
    if os.environ.get("T3_SEM_2A_TENTATIVA") != "1":  # (só para medir o custo antes/depois)
        t2, f2, k2, e2 = translate_piece(src, tok, a, post_fn, fewshot=True)  # [v2] 2a tentativa: o pedaço inteiro, com exemplos (uma só chamada)
        n += k2
        if not e2 and check_piece(src, t2, f2) is None and not lang_bad(src, t2) and not leaked(src, t2):
            return t2, f2, n, None
    units = split_units(src, 120)
    if len(units) < 2:
        return text, fin, n, err
    outs, total = [], n
    for u, sep in units:
        t, f, k, e = translate_piece(u, tok, a, post_fn, fewshot=True)
        total += k
        if e or check_piece(u, t, f) is not None or lang_bad(u, t, 80) or leaked(u, t):
            t, f, k, e = translate_piece(u, tok, a, post_fn)  # tenta de novo sem exemplos
            total += k
            if e or check_piece(u, t, f) is not None or lang_bad(u, t, 80):
                return text, fin, total, err  # desiste: devolve a primeira tradução, que será recusada pelo verificador
        outs.append(t.strip() + sep)
    return "".join(outs).strip(), True, total, None


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
    # [v2] processamento contínuo (sem barreira por bloco): cada trabalhador traduz um exemplo inteiro e o grava quando termina;
    # outro exemplo entra no lugar. O original esperava o bloco todo terminar, e os exemplos lentos deixavam os demais parados.
    def process_example(r):
        parts, n_tok = {}, 0
        for name, src in jobs_for(r, a.think_frac, a.max_chars):
            text, fin, n, err = translate_robust(src, tok, a, post_fn)
            n_tok += n
            parts[name] = {"text": text, "why": "erro_de_requisicao" if err else check_piece(src, text, fin)}
            if parts[name]["why"]:
                break  # um pedaço ruim já derruba o exemplo (mesmo veredito do original); não gasta servidor com o resto
        return assemble(r, parts), n_tok

    from concurrent.futures import FIRST_COMPLETED, wait
    last_print = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, out.open("a") as fout:
        it = iter(todo)
        pending = set()
        stop_submit = False

        def submit_next() -> bool:
            r = next(it, None)
            if r is None:
                return False
            pending.add(ex.submit(process_example, r))
            return True

        for _ in range(a.concurrency):
            if not submit_next():
                break
        ndone = 0
        while pending:
            fin, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in fin:
                rec, n = f.result()
                ntok += n
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                stats[f"{rec['kind']}:{rec['verdict'].split(':')[-1]}"] += 1
                ndone += 1
                if not stop_submit and deadline and time.time() > deadline:
                    stop_submit = True
                    print("tempo esgotado; termina os que estão em andamento", flush=True)
                if not stop_submit:
                    submit_next()
            if time.time() - last_print > 60:
                last_print = time.time()
                el = time.time() - t0
                print(f"[{el / 60:5.1f} min] {ndone}/{len(todo)} exemplos | {ndone / el * 60:.1f} exemplos/min | {ntok / el:.0f} tokens/s | "
                      f"{USAGE['req'] / max(1, ndone):.1f} req e {USAGE['prompt_tokens'] / max(1, ndone):.0f} tokens de prompt por exemplo | {dict(stats)}", flush=True)
        el = time.time() - t0
        print(f"[{el / 60:5.1f} min] {ndone}/{len(todo)} exemplos | {ntok / max(el, 1e-9):.0f} tokens/s | {dict(stats)}", flush=True)
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
    ap.add_argument("--timeout", type=float, default=3600)  # [v2] (era 900: estoura com o servidor cheio)
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

"""Traduz para o português as questões de ciência (e os traços do R1) de data/mot_science_en.jsonl, com o Lumi no vLLM.

Registros com `ready_pt` (enunciado, alternativas e texto final já em português, como os de nvidia/Nemotron-SFT-Multilingual-v2)
só têm o raciocínio traduzido. Os problemas são divididos por hash (70% com raciocínio, 30% só resposta, como em build_teacher_dataset.py). Com raciocínio
traduz-se enunciado+alternativas, o raciocínio (em blocos de parágrafos) e o texto final; sem raciocínio, só enunciado+
alternativas e texto final. Tradução gulosa (T=0), modo sem thinking. Um bloco ruim (não terminou, tamanho estranho,
alternativas que não voltam, raciocínio fora do português) derruba o registro. A descontaminação (8-grama contra as
nossas provas) é feita depois, na CPU, por scripts/filter_translated.py. Em blocos de registros, retomável.

Saída: <out>/{think,nothink}.jsonl (para build_teacher_dataset.py --disjoint-inputs), raw.jsonl e report.json.
  scripts/detach.sh --gpu translate "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/translate_traces.py --n 200 --out data/mot_pilot"
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from lumi_ft.data import gigaverbo as GV
from lumi_ft.data import mot_science as M
from lumi_ft.data import translate as TR
from lumi_ft.eval import generate as G
from lumi_ft.eval import mcq
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION, load_tokenizer


def question_block(rec: dict) -> str:
    return rec["question"] + "\n" + "\n".join(f"{mcq.LETTERS[i]}: {o}" for i, o in enumerate(rec["options"]))


def pick(records: list[dict], n: int, per_question: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rs = list(records)
    rng.shuffle(rs)
    seen, out = Counter(), []
    for r in rs:
        if seen[r["problem_id"]] >= per_question:
            continue
        seen[r["problem_id"]] += 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/mot_science_en.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-question", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=400, help="registros por bloco de geração")
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(a.src).read_text().splitlines() if line.strip()]
    chosen = pick(records, a.n, a.per_question, a.seed)
    raw, f_think, f_nothink = out / "raw.jsonl", out / "think.jsonl", out / "nothink.jsonl"
    done = {json.loads(line)["id"] for line in raw.read_text().splitlines()} if raw.exists() else set()
    todo = [r for r in chosen if r["id"] not in done]
    print(f"{len(chosen)} registros escolhidos, {len(done)} já feitos", flush=True)

    from vllm import LLM, SamplingParams

    tok = load_tokenizer()
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=a.mem,
              max_num_seqs=a.max_seqs, enable_prefix_caching=True, seed=0, generation_config="vllm")
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)
    report = Counter()
    for i in range(0, len(todo), a.chunk):
        block = todo[i: i + a.chunk]
        jobs = []  # (registro, parte, texto de origem)
        for r in block:
            with_think = GV.bucket(r["problem_id"]) < 7000
            if not r.get("ready_pt"):  # enunciado, alternativas e texto final já em português: só o raciocínio precisa de tradução
                jobs.append((r, "q", question_block(r)))
                jobs.append((r, "f", r["final"]) if r["final"].strip() else None)
            if with_think:
                jobs += [(r, f"t{k}", c) for k, c in enumerate(TR.chunk_paragraphs(r["thought"]))]
        jobs = [j for j in jobs if j]
        prompts = [G.render_prompt(tok, TR.make_prompt(j[2]), thinking=False) for j in jobs]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        parts: dict[str, dict[str, tuple[str, str | None]]] = {r["id"]: {} for r in block}
        for (r, part, src), o in zip(jobs, outs):
            c = o.outputs[0]
            parts[r["id"]][part] = (c.text.strip(), TR.check_chunk(src, c.text, c.finish_reason == "stop"))
        with raw.open("a") as fr:
            for r in block:
                p = parts[r["id"]]
                with_think = GV.bucket(r["problem_id"]) < 7000
                why = next((f"{k}:{v[1]}" for k, v in p.items() if v[1]), None)
                rec = None
                if why is None:
                    ready = bool(r.get("ready_pt"))
                    sp_q = (r["question"], r["options"]) if ready else M.split_options(p["q"][0])
                    if sp_q is None or len(sp_q[1]) != len(r["options"]):
                        why = "alternativas_nao_voltaram"
                    else:
                        thought = TR.join_chunks([p[k][0] for k in sorted((k for k in p if k.startswith("t")), key=lambda k: int(k[1:]))]) if with_think else None
                        final = r["final"] if ready else (p["f"][0] if "f" in p else "")
                        if with_think and not TR.is_portuguese(thought):
                            why = "raciocinio_nao_pt"
                        else:
                            prompt = mcq.build_prompt({"question": sp_q[0], "options": sp_q[1]})
                            answer = (final + "\n\n" if final else "") + f"Resposta: {r['letter']}"
                            tag = "nemotron_pt" if ready else "mot_science_pt"
                            rec = {"id": f"{tag}:{r['id']}", "problem_id": r["problem_id"], "question": sp_q[0], "prompt": prompt,
                                   "answer": answer, "source": tag, "letter": r["letter"], "options": sp_q[1]}
                            if with_think:
                                rec["thought"] = thought
                why = why or "ok"
                report[("think" if with_think else "nothink") + ":" + why.split(":")[-1]] += 1
                fr.write(json.dumps({"id": r["id"], "verdict": why, "en": {"q": question_block(r)}, "pt": {k: v[0] for k, v in p.items()}}, ensure_ascii=False) + "\n")
                if rec:
                    with (f_think if with_think else f_nothink).open("a") as fk:
                        fk.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"bloco {i // a.chunk + 1}: {len(block)} registros, {len(jobs)} pedidos em {dt:.0f} s ({ntok / dt:.0f} tok/s) | {dict(report)}", flush=True)
    (out / "report.json").write_text(json.dumps({"args": vars(a), "verdicts": dict(report)}, ensure_ascii=False, indent=1))
    print(json.dumps(dict(report), ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())

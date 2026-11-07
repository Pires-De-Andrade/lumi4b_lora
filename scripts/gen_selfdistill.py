"""Gera dados de treino com o próprio Lumi (vLLM): raciocínio em português forçado por prefill, só o que acerta.

Com raciocínio: prefill "Vou raciocinar em português." + temperatura 0,6. Sem raciocínio: o modelo como é. Mesmos prompts
da avaliação (instrução `#### N`). Os problemas vêm de handoff/thor/data/prompts.jsonl, sem os duplicados da validação
(assinatura de números + Jaccard >= 0.3). Em blocos, retomável. Roda no ambiente do vLLM:
  scripts/detach.sh --gpu gen-self "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/gen_selfdistill.py"
Saída: data/selfdist/{think,nothink}.jsonl (para build_teacher_dataset.py), raw_*.jsonl e report.json.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from lumi_ft.data import selfdistill as S
from lumi_ft.data import teacher as T
from lumi_ft.eval import generate as G
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION, load_tokenizer


def words(t):
    return set(re.findall(r"[a-zà-ÿ]{4,}", t.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="handoff/thor/data/prompts.jsonl")
    ap.add_argument("--think-n", type=int, default=14_000)
    ap.add_argument("--nothink-n", type=int, default=3_000)
    ap.add_argument("--prefill", default="Vou raciocinar em português.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", default="data/selfdist")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    val = [json.loads(line) for line in Path("data/val_gsm8k.jsonl").read_text().splitlines()]
    val_by_sig: dict[tuple, list[set]] = {}
    for v in val:
        s = T.number_signature(v["question"])
        if len(s) >= 2:
            val_by_sig.setdefault(s, []).append(words(v["question"]))
    items, dropped = [], 0
    for line in Path(a.prompts).read_text().splitlines():
        r = json.loads(line)
        s = T.number_signature(r["question"])
        w = words(r["question"])
        if len(s) >= 2 and any(len(w & c) / max(1, len(w | c)) >= 0.3 for c in val_by_sig.get(s, [])):
            dropped += 1
            continue
        items.append(r)
    print(f"{len(items)} problemas depois de tirar {dropped} duplicados da validação", flush=True)
    think_items = items[: a.think_n]
    nothink_items = items[a.think_n: a.think_n + a.nothink_n]

    from vllm import LLM, SamplingParams

    tok = load_tokenizer()
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=a.mem,
              max_num_seqs=a.max_seqs, enable_prefix_caching=True, seed=0, generation_config="vllm")
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False, seed=0)
    report = {"problems_dropped_val_duplicates": dropped, "args": vars(a), "think": Counter(), "nothink": Counter()}
    for mode, its, pre, accept in (("think", think_items, a.prefill, S.accept_think), ("nothink", nothink_items, "", S.accept_nothink)):
        raw, kept = out / f"raw_{mode}.jsonl", out / f"{mode}.jsonl"
        done = {json.loads(line)["id"] for line in raw.read_text().splitlines()} if raw.exists() else set()
        todo = [r for r in its if r["id"] not in done]
        print(f"{mode}: {len(its)} problemas, {len(done)} já feitos", flush=True)
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i: i + a.chunk]
            prompts = [G.render_prompt(tok, r["prompt"], thinking=(mode == "think")) + pre for r in chunk]
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntok = 0
            with raw.open("a") as fr, kept.open("a") as fk:
                for r, o in zip(chunk, outs):
                    c = o.outputs[0]
                    text, finished = pre + c.text, c.finish_reason == "stop"
                    ntok += len(c.token_ids)
                    rec, why = accept(text, finished, r["ref"])
                    fr.write(json.dumps({"id": r["id"], "finished": finished, "n_gen": len(c.token_ids), "verdict": why, "text": text}, ensure_ascii=False) + "\n")
                    report[mode][why] += 1
                    if rec:
                        fk.write(json.dumps({"id": f"selfdist_{mode}:{r['id']}", "problem_id": r["id"], "question": r["question"],
                                             "source": "lumi_self", **rec}, ensure_ascii=False) + "\n")
            print(f"{mode} bloco {i // a.chunk + 1}: {len(chunk)} problemas em {dt:.0f} s ({ntok / dt:.0f} tok/s) | {dict(report[mode])}", flush=True)
    report["think"], report["nothink"] = dict(report["think"]), dict(report["nothink"])
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())

"""Fase anti-laço, passo 1: gera com o vLLM em decodificação gulosa (como a avaliação) e minera os laços (roda no ambiente do vLLM).

Para cada prompt de `data/antidoom_prompts.jsonl` (campo `split`), gera uma resposta no modo thinking, sem começo forçado, com temperatura 0 e
`--max-new` tokens. Das que não terminam, procura um laço no fim (`lumi_ft.train.antidoom.find_loop_start`) e grava o evento (ids do prompt, ids do
prefixo até o início da repetição e o token rejeitado). Também serve para medir a taxa de laço de um modelo (use `--split heldout`): `stats.json`
traz as taxas de resposta que não termina e de laço por tipo de prompt.
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/antidoom_mine.py --model runs/merged/gspo1_c8 --split mine --out runs/antidoom/r1/mine
Saída em --out: events.jsonl (um por laço), samples.jsonl (uma linha por prompt, sem ids) e stats.json.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from lumi_ft.eval import generate as G
from lumi_ft.modeling.load import EOS_IDS, load_tokenizer
from lumi_ft.train import antidoom as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pasta do modelo fundido")
    ap.add_argument("--prompts", default="data/antidoom_prompts.jsonl")
    ap.add_argument("--split", default="mine", choices=["mine", "heldout"])
    ap.add_argument("--limit", type=int, default=0, help="usa só os N primeiros prompts (0 = todos)")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--min-chars", type=int, default=40, help="tamanho mínimo, em caracteres, do bloco que se repete")
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    if a.limit:
        rows = rows[:a.limit]
    tok = load_tokenizer()
    texts = [G.render_prompt(tok, r["prompt"], thinking=True) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_new + 1536, gpu_memory_utilization=a.mem, max_num_seqs=a.max_seqs,
              enable_prefix_caching=True, seed=0, generation_config="vllm")
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cnt, n_tok = Counter(), 0
    with (out / "events.jsonl").open("w") as fe, (out / "samples.jsonl").open("w") as fs:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            n_tok += len(ids)
            finished = c.finish_reason == "stop"
            hit = None if finished else A.find_loop_start(ids, min_chars=a.min_chars, decode=tok.decode)
            cnt[f"{r['kind']}:total"] += 1
            cnt[f"{r['kind']}:nao_termina"] += int(not finished)
            cnt[f"{r['kind']}:laco"] += int(hit is not None)
            fs.write(json.dumps({"id": r["id"], "kind": r["kind"], "n_gen": len(ids), "finished": finished, "loop": hit is not None,
                                 "k": hit[1] if hit else None}) + "\n")
            if hit is not None:
                s, k = hit
                fe.write(json.dumps({"id": r["id"], "kind": r["kind"], "prompt_ids": list(o.prompt_token_ids), "prefix_ids": ids[:s],
                                     "rejected": ids[s], "k": k, "s": s, "n_gen": len(ids)}) + "\n")
    st = {"model": a.model, "split": a.split, "max_new": a.max_new, "min_chars": a.min_chars, "prompts": len(rows), "gen_seconds": round(dt),
          "gen_tokens": n_tok, "tok_per_s": round(n_tok / dt), "contagens": dict(cnt), "taxas": {}}
    for kind in sorted({r["kind"] for r in rows}):
        tot = cnt[f"{kind}:total"]
        st["taxas"][kind] = {"nao_termina": round(cnt[f"{kind}:nao_termina"] / tot, 4), "laco": round(cnt[f"{kind}:laco"] / tot, 4)}
    (out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st))


if __name__ == "__main__":
    main()

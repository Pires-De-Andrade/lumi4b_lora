"""GSM8K-PT nos dois modos: gera em lotes fixos, grava por lote e pontua (A-04 a A-08).

Retomável: se interrompido, repetir o mesmo comando continua do primeiro lote que falta.
`--score-only` pontua o que já foi gerado, sem carregar o modelo.

Piloto (fase 0, mede a cauda de saídas longas antes da baseline):
  scripts/detach.sh --gpu pilot-gsm8k PYTHONPATH=src python scripts/run_gsm8k.py --n 256 --out runs/pilot_gsm8k
Baseline completa (todos os 1.295 itens): --n 0
"""
import argparse
import json
import time
from pathlib import Path

from lumi_ft import gpu
from lumi_ft.eval import generate as G
from lumi_ft.eval import gsm8k as K

MODES = {"thinking": True, "nothinking": False}


def score(out_dir: Path, items: list[dict], mode: str) -> dict:
    by_id = {it["id"]: it for it in items}
    path = out_dir / f"{mode}.jsonl"
    lines = gpu.read_jsonl(path)
    batches = [r for r in lines if "batch" in r]
    gens = [it for b in batches for it in b["items"]]
    rows = [K.score_row(g, by_id[g["id"]]["answer"], thinking=MODES[mode]) for g in gens]
    (out_dir / f"{mode}_scores.json").write_text(json.dumps(rows, ensure_ascii=False))
    return K.summarize(rows, batches, thinking=MODES[mode])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="itens (0 = todos os 1.295)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="thinking,nothinking")
    ap.add_argument("--batch", type=int, default=G.MAX_BATCH)
    ap.add_argument("--budget", type=int, default=G.TOKEN_BUDGET)
    ap.add_argument("--max-new", type=int, default=G.MAX_NEW_TOKENS)
    ap.add_argument("--out", default="runs/pilot_gsm8k")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = a.modes.split(",")
    items, dataset_rev = K.load_items(a.n or None, a.seed)
    print(f"{len(items)} itens do {K.DATASET} @ {dataset_rev[:8]}", flush=True)

    if not a.score_only:
        import torch

        from lumi_ft.modeling.load import MODEL_ID, REVISION, load_model, load_tokenizer

        tok = load_tokenizer()
        model = load_model("native").eval()
        gen_fn = G.make_generate_fn(model, tok, a.max_new)
        meta_run = {"model": MODEL_ID, "revision": REVISION, "dataset": K.DATASET, "dataset_revision": dataset_rev,
                    "seed": a.seed, "n": len(items), "instruction": K.INSTRUCTION, "torch": torch.__version__,
                    "decoding": "greedy, repetition_penalty=1.0, eos=[2,11]"}
        (out_dir / "meta.json").write_text(json.dumps({**meta_run, "gpu_start": gpu.snapshot()}, ensure_ascii=False, indent=1))
        t0 = time.time()
        for mode in modes:
            thinking = MODES[mode]
            prompts = [{"id": it["id"], "prompt": G.render_prompt(tok, K.build_prompt(it["question"]), thinking=thinking)}
                       for it in items]
            for p in prompts:
                p["n_prompt"] = len(tok(p["prompt"], add_special_tokens=False)["input_ids"])
            print(f"== modo {mode}: prompts de {min(p['n_prompt'] for p in prompts)} a {max(p['n_prompt'] for p in prompts)} tokens", flush=True)
            G.run_batches(prompts, gen_fn, out_dir / f"{mode}.jsonl", max_batch=a.batch, token_budget=a.budget,
                          max_new_tokens=a.max_new, meta={"mode": mode, "enable_thinking": thinking,
                                                          "dataset_revision": dataset_rev, "seed": a.seed})
        meta = json.loads((out_dir / "meta.json").read_text())
        meta |= {"gpu_end": gpu.snapshot(), "wall_seconds": round(time.time() - t0, 1)}
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    summary = {m: score(out_dir, items, m) for m in modes}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    main()

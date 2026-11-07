"""GSM8K-PT no vLLM: mesmos itens, prompts e pontuação do piloto em HF (scripts/run_gsm8k.py).

Roda no ambiente separado:  PYTHONPATH=src .venv-vllm/bin/python scripts/run_gsm8k_vllm.py --n 256
O vLLM tem batching contínuo: uma saída que não termina não segura as outras, e o lote não precisa ter
tamanho nem ordem fixos (a ordem de agendamento muda de uma execução para outra; `--repeat 2` mede o
efeito disso). Greedy, até 4.096 tokens, `stop_token_ids=[2, 11]`, sem penalidade de repetição.
"""
import argparse
import json
import time
from pathlib import Path

from lumi_ft.eval import generate as G
from lumi_ft.eval import gsm8k as K
from lumi_ft.eval import parse as P
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION

MODES = {"thinking": True, "nothinking": False}


def run_mode(llm, sp, items, tok, thinking: bool):
    prompts = [G.render_prompt(tok, K.build_prompt(it["question"]), thinking=thinking) for it in items]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    gens = []
    for it, o in zip(items, outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        while ids and ids[-1] in EOS_IDS:  # o token de parada vem incluído em token_ids
            ids.pop()
        gens.append({"id": it["id"], "n_prompt": len(o.prompt_token_ids), "text": c.text, "n_gen": len(ids),
                     "finished": c.finish_reason == "stop", "gen_len": len(ids), "finish_reason": c.finish_reason})
    return gens, wall


def agreement(a: list[dict], b: list[dict]) -> dict:
    """Concordância entre duas execuções (mesma ordem de ids)."""
    n = len(a)
    same_text = sum(x["text"] == y["text"] for x, y in zip(a, b))
    return {"n": n, "identical_text": round(same_text / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="thinking,nothinking")
    ap.add_argument("--max-new", type=int, default=G.MAX_NEW_TOKENS)
    ap.add_argument("--max-seqs", type=int, default=128, help="max_num_seqs (sequências simultâneas)")
    ap.add_argument("--max-len", type=int, default=8192, help="max_model_len")
    ap.add_argument("--mem", type=float, default=0.88, help="gpu_memory_utilization")
    ap.add_argument("--repeat", type=int, default=1, help="repete cada modo para medir a reprodutibilidade")
    ap.add_argument("--compare", default="runs/pilot_gsm8k", help="pasta do piloto em HF para comparar")
    ap.add_argument("--out", default="runs/pilot_gsm8k_vllm")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    from lumi_ft.modeling.load import load_tokenizer

    items, rev = K.load_items(a.n or None, a.seed)
    tok = load_tokenizer()
    t0 = time.time()
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16", max_model_len=a.max_len,
              gpu_memory_utilization=a.mem, max_num_seqs=a.max_seqs, enable_prefix_caching=True, seed=0,
              generation_config="vllm", trust_remote_code=a.trust_remote_code)
    init_s = round(time.time() - t0, 1)
    print(f"engine pronto em {init_s} s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)

    summary, by_mode = {"init_seconds": init_s, "max_num_seqs": a.max_seqs, "n": len(items)}, {}
    for mode in a.modes.split(","):
        thinking = MODES[mode]
        runs = []
        for r in range(a.repeat):
            gens, wall = run_mode(llm, sp, items, tok, thinking)
            runs.append((gens, wall))
            tokens = sum(g["n_gen"] for g in gens)
            print(f"{mode} #{r + 1}: {wall:.1f} s | {tokens} tokens | {tokens / wall:.0f} tok/s | {len(items) / wall:.2f} itens/s", flush=True)
        gens, wall = runs[0]
        (out / f"{mode}.jsonl").write_text(json.dumps({"batch": 0, "size": len(gens), "seconds": round(wall, 2),
                                                       "gen_len": max(g["gen_len"] for g in gens), "items": gens},
                                                      ensure_ascii=False) + "\n")
        by_id = {it["id"]: it for it in items}
        rows = [K.score_row(g, by_id[g["id"]]["answer"], thinking=thinking) for g in gens]
        (out / f"{mode}_scores.json").write_text(json.dumps(rows, ensure_ascii=False))
        s = K.summarize(rows, [], thinking=thinking)
        tokens = sum(g["n_gen"] for g in gens)
        s["vllm"] = {"wall_seconds": round(wall, 1), "items_per_s": round(len(items) / wall, 2),
                     "tokens_per_s": round(tokens / wall), "walls_repeat": [round(w, 1) for _, w in runs]}
        if a.repeat > 1:
            s["vllm"]["run_to_run_agreement"] = agreement(runs[0][0], runs[1][0])
            s["vllm"]["run_to_run_strict_acc"] = [
                round(sum(K.score_row(g, by_id[g["id"]]["answer"], thinking=thinking)["strict_ok"] for g in rr) / len(rr), 4)
                for rr, _ in runs]
        hf = Path(a.compare) / f"{mode}.jsonl"
        if hf.exists():
            hf_items = {it["id"]: it for ln in hf.read_text().splitlines() if '"batch"' in ln
                        for it in json.loads(ln)["items"]}
            if set(hf_items) == {g["id"] for g in gens}:
                hf_gens = [hf_items[g["id"]] for g in gens]
                hf_rows = [K.score_row(g, by_id[g["id"]]["answer"], thinking=thinking) for g in hf_gens]
                s["vs_hf"] = {**agreement(gens, hf_gens),
                              "hf_strict_acc": round(sum(r["strict_ok"] for r in hf_rows) / len(hf_rows), 4),
                              "vllm_strict_acc": s["strict_acc"],
                              "both_right": sum(r["strict_ok"] and v["strict_ok"] for r, v in zip(hf_rows, rows)),
                              "only_hf_right": sum(r["strict_ok"] and not v["strict_ok"] for r, v in zip(hf_rows, rows)),
                              "only_vllm_right": sum(v["strict_ok"] and not r["strict_ok"] for r, v in zip(hf_rows, rows)),
                              "hf_wall_seconds": sum(json.loads(ln)["seconds"] for ln in hf.read_text().splitlines() if '"batch"' in ln)}
        summary[mode] = s
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    main()

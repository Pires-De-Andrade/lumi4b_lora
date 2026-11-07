"""Avalia um modelo (o Lumi ou um adaptador fundido) no vLLM, em tarefas e modos. Retomável por (tarefa, modo).

Saída em runs/eval/<nome>/: <tarefa>_<modo>.jsonl (uma saída por linha), <tarefa>_<modo>_scores.json e summary.json.
Greedy, `stop_token_ids=[2, 11]`, até 4.096 tokens novos nos dois modos (A-04). Rodar no ambiente do vLLM:

  scripts/detach.sh --gpu eval-base "VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py \\
      --name base --tasks validation"
  ... --name viab --model runs/merged/viab --tasks validation

--tasks: nomes separados por vírgula ou os grupos `validation`, `suite` (Nota de raciocínio), `card`, `all`.
"""
import argparse
import json
import time
from pathlib import Path

from lumi_ft.eval import generate as G
from lumi_ft.eval import tasks as T
from lumi_ft.modeling.load import EOS_IDS, MODEL_ID, REVISION, load_tokenizer

GROUPS = {"validation": T.VALIDATION, "suite": T.SUITE, "card": T.CARD, "all": T.SUITE + T.CARD}
MODES = {"thinking": True, "nothinking": False}


def expand(tasks: str) -> list[str]:
    out = []
    for t in tasks.split(","):
        out += GROUPS.get(t, [t])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=MODEL_ID, help="id do Hub (usa a revisão fixada) ou pasta com o modelo fundido")
    ap.add_argument("--tasks", default="validation")
    ap.add_argument("--modes", default="thinking,nothinking")
    ap.add_argument("--limit", type=int, default=0, help="só os N primeiros itens de cada tarefa (smoke)")
    ap.add_argument("--max-new", type=int, default=G.MAX_NEW_TOKENS)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", default="runs/eval")
    a = ap.parse_args()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    sum_path = out / "summary.json"
    summary = json.loads(sum_path.read_text()) if sum_path.exists() else {}
    names = expand(a.tasks)
    todo = [(t, m) for t in names for m in a.modes.split(",") if f"{t}|{m}" not in summary]
    if not todo:
        print("nada a fazer: tudo já está em", sum_path)
        return

    from vllm import LLM, SamplingParams

    tok = load_tokenizer()
    t0 = time.time()
    kw = {"revision": REVISION} if a.model == MODEL_ID else {}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, gpu_memory_utilization=a.mem, max_num_seqs=a.max_seqs,
              enable_prefix_caching=True, seed=0, generation_config="vllm", **kw)
    summary["_engine"] = {"model": a.model, "init_seconds": round(time.time() - t0, 1), "max_num_seqs": a.max_seqs,
                          "max_len": a.max_len, "decoding": "greedy, 4096 tokens, stop_token_ids=[2, 11], repetition_penalty=1.0"}
    loaded: dict[str, T.Task] = {}
    for tname, mode in todo:
        thinking = MODES[mode]
        task = loaded.get(tname) or loaded.setdefault(tname, T.get_task(tname, a.limit))
        prompts = [G.render_prompt(tok, task.prompt(it), thinking=thinking) for it in task.items]
        n_prompt = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        params = [SamplingParams(temperature=0.0, max_tokens=max(1, min(a.max_new, a.max_len - n)), stop_token_ids=EOS_IDS,
                                 skip_special_tokens=False) for n in n_prompt]
        capped = sum(n > a.max_len - a.max_new for n in n_prompt)
        t1 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t1
        gens = []
        for it, o in zip(task.items, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in EOS_IDS:
                ids.pop()
            gens.append({"id": it["id"], "n_prompt": len(o.prompt_token_ids), "text": c.text, "n_gen": len(ids),
                         "finished": c.finish_reason == "stop", "finish_reason": c.finish_reason})
        rows = [task.score(g, it, thinking) for g, it in zip(gens, task.items)]
        (out / f"{tname}_{mode}.jsonl").write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gens) + "\n")
        (out / f"{tname}_{mode}_scores.json").write_text(json.dumps(rows, ensure_ascii=False))
        s = task.summarize(rows, thinking)
        tokens = sum(g["n_gen"] for g in gens)
        s["run"] = {"wall_seconds": round(wall, 1), "items_per_s": round(len(gens) / wall, 2), "tokens_per_s": round(tokens / wall),
                    "max_prompt_tokens": max(n_prompt), "prompts_with_capped_max_tokens": capped}
        summary[f"{tname}|{mode}"] = s
        sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"{tname} {mode}: n={s['n']} strict={s['strict_acc']} flex={s['flexible_acc']} trunc={s['truncated_rate']} "
              f"| {wall:.0f} s, {tokens / wall:.0f} tok/s, max_prompt={max(n_prompt)}", flush=True)
    print("EXIT_OK")


if __name__ == "__main__":
    main()

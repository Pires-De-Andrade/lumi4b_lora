"""GRPO, passo 1: gera `--group` respostas por prompt com o vLLM e calcula recompensa e vantagem (roda no ambiente do vLLM).

Os prompts da iteração vêm de `lumi_ft.train.grpo.select_prompts` (metade matemática, metade múltipla escolha por padrão).
Modo thinking, sem começo forçado: o modelo (já com o adaptador do SFT fundido) começa o raciocínio sozinho. Grava uma linha por
resposta: ids do prompt e da resposta (exatos, sem retokenizar), recompensa, vantagem. Estatísticas em <out>.stats.json.
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/grpo_rollout.py --model runs/grpo/x/merged --iter 0 --out runs/grpo/x/it000/rollouts.jsonl
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from lumi_ft.eval import generate as G
from lumi_ft.modeling.load import EOS_IDS, load_tokenizer
from lumi_ft.train import grpo as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pasta do modelo fundido")
    ap.add_argument("--prompts", default="data/grpo_prompts.jsonl")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--frac-mcq", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--adv", default="std", choices=["std", "mean"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    sel = R.select_prompts(prompts, a.iter, a.n_prompts, a.frac_mcq, a.seed)
    tok = load_tokenizer()
    texts = [G.render_prompt(tok, p["prompt"], thinking=True) for p in sel]

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096 + 1024, gpu_memory_utilization=a.mem, max_num_seqs=a.max_seqs,
              enable_prefix_caching=True, seed=a.seed * 1000 + a.iter, generation_config="vllm")
    sp = SamplingParams(n=a.group, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    st = {"iter": a.iter, "prompts": len(sel), "group": a.group, "gen_seconds": round(dt), "whys": {}, "appended_stop_token": 0}
    by_kind: dict[str, list[float]] = {}
    n_signal, n_tok = Counter(), 0
    with out.with_suffix(".tmp").open("w") as f:
        for p, o in zip(sel, outs):
            outputs = []
            for c in o.outputs:
                ids, added = R.complete_ids(list(c.token_ids), c.finish_reason, c.stop_reason)
                st["appended_stop_token"] += int(added)
                n_tok += len(c.token_ids)
                outputs.append({"ids": ids, "text": c.text, "finished": c.finish_reason == "stop"})
            samples, whys = R.build_samples(p, list(o.prompt_token_ids), outputs, a.adv)
            for k, v in whys.items():
                st["whys"][f"{p['kind']}:{k}"] = st["whys"].get(f"{p['kind']}:{k}", 0) + v
            by_kind.setdefault(p["kind"], []).extend(s["reward"] for s in samples)
            n_signal[p["kind"]] += any(s["advantage"] != 0 for s in samples)
            for s in samples:
                f.write(json.dumps(s) + "\n")
    out.with_suffix(".tmp").replace(out)
    st["reward_mean"] = {k: round(sum(v) / len(v), 4) for k, v in by_kind.items()}
    st["groups_with_signal"] = dict(n_signal)
    st["gen_tokens"], st["tok_per_s"] = n_tok, round(n_tok / dt)
    out.with_suffix(".stats.json").write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(json.dumps(st, ensure_ascii=False))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())

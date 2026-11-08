"""Benchmark de velocidade da GERAÇÃO do RL (roda no ambiente do vLLM). Mesmo modelo, mesmos prompts, mesma amostragem do `grpo_rollout.py`;
muda só a configuração do motor: prefix caching, backend Mamba (triton ou flashinfer) e fração de memória da GPU. Liga o log de estatísticas
do vLLM (`disable_log_stats=False`) para ver quantas respostas rodam ao mesmo tempo e quanto do cache de KV está em uso.

Não grava rollouts nem mexe em nada do RL. Três modos:
  rodar : PYTHONPATH=src VLLM_USE_FLASHINFER_SAMPLER=0 .venv-vllm/bin/python scripts/bench_rollout.py --model M --label base [--prefix-caching 0] [--mamba-backend flashinfer] [--mem 0.93]
  parse : scripts/bench_rollout.py --parse-log runs/bench_rollout/base.log --label base --max-seqs 128   (lê o log do vLLM e anexa as estatísticas)
  tabela: scripts/bench_rollout.py --report [--out runs/bench_rollout/results.jsonl]
Resultados (uma linha JSON por execução ou por parse) em runs/bench_rollout/results.jsonl.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median

STAT_RUN = re.compile(r"Running: (\d+) reqs")
STAT_WAIT = re.compile(r"Waiting: (\d+) reqs")
STAT_KV = re.compile(r"GPU KV cache usage: ([\d.]+)%")
STAT_GEN = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s")


def llm_kwargs(model: str, mem: float, max_seqs: int, prefix_caching: bool, backend: str, seed: int, max_len: int = 4096 + 1024) -> dict:
    """Argumentos do `LLM(...)`. Com o backend padrão (triton) NÃO passa `mamba_config`: a base é idêntica à do `grpo_rollout.py`."""
    kw = dict(model=model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=mem, max_num_seqs=max_seqs,
              enable_prefix_caching=bool(prefix_caching), seed=seed, generation_config="vllm", disable_log_stats=False)
    if backend != "triton":
        kw["mamba_config"] = {"backend": backend}
    return kw


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def parse_log_stats(text: str, max_seqs: int) -> dict:
    """Estatísticas das linhas periódicas do vLLM ("Running: N reqs, Waiting: M reqs, GPU KV cache usage: Z%").
    Só conta as linhas com ao menos 1 resposta rodando (ignora a carga e o fim)."""
    run, wait, kv, gen = [], [], [], []
    for line in text.splitlines():
        r = STAT_RUN.search(line)
        if not r or int(r.group(1)) == 0:
            continue
        run.append(int(r.group(1)))
        w, k, g = STAT_WAIT.search(line), STAT_KV.search(line), STAT_GEN.search(line)
        wait.append(int(w.group(1)) if w else 0)
        if k:
            kv.append(float(k.group(1)))
        if g:
            gen.append(float(g.group(1)))
    if not run:
        return {"linhas": 0}
    return {"linhas": len(run), "rodando_medio": round(mean(run), 1), "rodando_mediana": median(run), "rodando_p10": _pct(run, 0.1),
            "rodando_max": max(run), "fracao_no_maximo": round(sum(x >= max_seqs for x in run) / len(run), 3),
            "esperando_medio": round(mean(wait), 1), "kv_medio_pct": round(mean(kv), 1) if kv else None, "kv_max_pct": max(kv) if kv else None,
            "tok_s_medio_do_log": round(mean(gen)) if gen else None}


def report(path: Path) -> str:
    """Tabela (markdown) das execuções do arquivo, juntando as linhas de `parse` pelo rótulo."""
    rows, stats = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        (stats if "vllm_stats" in d else rows)[d["label"]] = d
    out = ["| config | geração (s) | tokens | tok/s | trunc. | recompensa mcq/math | rodando (média/máx) | % no máximo | KV médio/máx % |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lab, d in rows.items():
        s = stats.get(lab, {}).get("vllm_stats", {})
        rew = d.get("reward_mean", {})
        out.append(f"| {lab} | {d['gen_seconds']} | {d['gen_tokens']} | {d['tok_per_s']} | {d['truncated']:.3f} | {rew.get('mcq', '-')}/{rew.get('math', '-')} | "
                   f"{s.get('rodando_medio', '-')}/{s.get('rodando_max', '-')} | {s.get('fracao_no_maximo', '-')} | {s.get('kv_medio_pct', '-')}/{s.get('kv_max_pct', '-')} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--label", default="")
    ap.add_argument("--prompts", default="data/grpo_prompts_gold.jsonl")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--frac-mcq", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.88)
    ap.add_argument("--prefix-caching", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mamba-backend", default="triton", choices=["triton", "flashinfer"])
    ap.add_argument("--out", default="runs/bench_rollout/results.jsonl")
    ap.add_argument("--parse-log")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.report:
        print(report(out))
        return 0
    if a.parse_log:
        st = parse_log_stats(Path(a.parse_log).read_text(errors="replace"), a.max_seqs)
        with out.open("a") as f:
            f.write(json.dumps({"label": a.label, "vllm_stats": st}) + "\n")
        print(json.dumps({"label": a.label, "vllm_stats": st}))
        return 0
    if not a.model or not a.label:
        ap.error("faltou --model e --label")

    from lumi_ft.eval import generate as G
    from lumi_ft.modeling.load import EOS_IDS, load_tokenizer
    from lumi_ft.train import grpo as R

    prompts = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    sel = R.select_prompts(prompts, a.iter, a.n_prompts, a.frac_mcq, a.seed)
    tok = load_tokenizer()
    texts = [G.render_prompt(tok, p["prompt"], thinking=True) for p in sel]

    from vllm import LLM, SamplingParams

    t_load = time.perf_counter()
    llm = LLM(**llm_kwargs(a.model, a.mem, a.max_seqs, bool(a.prefix_caching), a.mamba_backend, a.seed * 1000 + a.iter))
    t_load = time.perf_counter() - t_load
    sp = SamplingParams(n=a.group, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new, stop_token_ids=EOS_IDS, skip_special_tokens=False)
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    lengths, finished, by_kind = [], [], {}
    for p, o in zip(sel, outs):
        outputs = [{"ids": R.complete_ids(list(c.token_ids), c.finish_reason, c.stop_reason)[0], "text": c.text, "finished": c.finish_reason == "stop"}
                   for c in o.outputs]
        lengths += [len(c.token_ids) for c in o.outputs]
        finished += [c.finish_reason == "stop" for c in o.outputs]
        samples, _ = R.build_samples(p, list(o.prompt_token_ids), outputs, "std")
        by_kind.setdefault(p["kind"], []).extend(s["reward"] for s in samples)
    n_tok = sum(lengths)
    rec = {"label": a.label, "model": a.model, "prefix_caching": a.prefix_caching, "mamba_backend": a.mamba_backend, "mem": a.mem, "max_seqs": a.max_seqs,
           "prompts": len(sel), "group": a.group, "max_new": a.max_new, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
           "load_seconds": round(t_load), "gen_seconds": round(dt), "gen_tokens": n_tok, "tok_per_s": round(n_tok / dt),
           "len_mediana": int(median(lengths)), "len_p90": int(_pct(lengths, 0.9)), "truncated": round(1 - sum(finished) / len(finished), 4),
           "reward_mean": {k: round(sum(v) / len(v), 4) for k, v in by_kind.items()}}
    with out.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))
    print("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

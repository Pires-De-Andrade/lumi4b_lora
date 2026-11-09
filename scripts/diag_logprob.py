"""Diagnóstico de log-probabilidades do RL: descompasso vLLM x treinador, ruído do corte do GSPO e cabeça em FP32 (ver lumi_ft/train/logprob_diag.py).

Dois estágios, em ambientes diferentes, sobre as mesmas N respostas de um arquivo de rollouts (sorteio fixo):
  1) vLLM (.venv-vllm): log-probabilidade, temperatura 1, de cada token da resposta, pelo modelo fundido que gerou os rollouts.
     VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/diag_logprob.py vllm \
        --model runs/merged/thor_pt_e1 --rollouts runs/grpo/gspo_thor_e1/it000/rollouts.jsonl --out runs/diag/lp_vllm.json
  2) HF (.venv): a mesma conta no treinador (nativo + adaptador, BF16, como o grpo_train.py), com e sem cabeça FP32, em duas composições
     de batch (a da passada "antiga" e a de um minibatch de treino), e a checagem "adaptador desligado = Lumi".
     PYTHONPATH=src python scripts/diag_logprob.py hf --adapter runs/train/thor_pt_e1/adapter \
        --rollouts runs/grpo/gspo_thor_e1/it000/rollouts.jsonl --vllm-json runs/diag/lp_vllm.json --out runs/diag/lp_report.json
O adaptador e o modelo fundido do vLLM têm de ser os mesmos que geraram o arquivo de rollouts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from lumi_ft.train import logprob_diag as D


def stage_vllm(a) -> int:
    from vllm import LLM, SamplingParams

    picked = D.pick_samples(a.rollouts, a.n, a.seed, a.max_len)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096 + 1024, gpu_memory_utilization=a.mem, max_num_seqs=16,
              enable_prefix_caching=False, seed=0, generation_config="vllm")
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    prompts = [{"prompt_token_ids": s["prompt_ids"] + s["completion_ids"]} for s in picked]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    lps = {}
    for i, (s, o) in enumerate(zip(picked, outs)):
        ids, npr = s["prompt_ids"] + s["completion_ids"], len(s["prompt_ids"])
        pl = o.prompt_logprobs
        vals = []
        for t in range(npr, len(ids)):
            entry = pl[t]
            assert entry is not None and ids[t] in entry, f"sem logprob do token na posição {t} da sequência {i}"
            vals.append(float(entry[ids[t]].logprob))
        lps[str(i)] = vals
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "rollouts": a.rollouts, "n": len(picked), "seed": a.seed, "signature": D.signature(picked),
                               "seconds": round(time.perf_counter() - t0, 1), "logprobs": lps}))
    print(f"vLLM: {len(picked)} sequências, {sum(len(v) for v in lps.values())} tokens -> {out}")
    print("EXIT_OK")
    return 0


def stage_hf(a) -> int:
    import torch
    from peft import PeftModel

    from lumi_ft.modeling.load import load_model
    from lumi_ft.train import grpo as R
    from lumi_ft.train import sft

    picked = D.pick_samples(a.rollouts, a.n, a.seed, a.max_len)
    ref = json.loads(Path(a.vllm_json).read_text())
    assert ref["signature"] == D.signature(picked), "o vLLM e o HF não avaliaram as mesmas sequências (arquivo, N ou semente diferentes)"
    vllm_lp = {int(k): v for k, v in ref["logprobs"].items()}
    rows = [{"input_ids": s["prompt_ids"] + s["completion_ids"], "n_prompt": len(s["prompt_ids"]),
             "loss_end": len(s["prompt_ids"]) + len(s["completion_ids"]), "kind": s.get("kind", "")} for s in picked]
    idx_all = list(range(len(rows)))

    model = load_model("native")
    model.config.use_cache = False

    def run(mod, groups):
        out: dict[int, list[float]] = {}
        for mb in groups:
            b = sft.collate(rows, mb)
            ids, mask = b["input_ids"].cuda(non_blocking=True), b["attention_mask"].cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mod(input_ids=ids, attention_mask=mask).logits
            for k, i in enumerate(mb):
                out[i] = R.completion_logprobs(logits[k], ids[k], rows[i]["n_prompt"], rows[i]["loss_end"]).cpu().tolist()
            del logits
        return out

    old_groups = sft.micro_batches(rows, idx_all, a.token_budget)  # composição da passada "antiga": todas as respostas, por comprimento
    new_groups = [g for chunk in D.shuffled_chunks(len(rows), a.chunk, a.seed) for g in sft.micro_batches(rows, chunk, a.token_budget)]

    model.eval()
    raw_lp = run(model, old_groups)  # modelo base (Lumi), sem adaptador, antes de embrulhar com o PEFT
    pm = PeftModel.from_pretrained(model, a.adapter, is_trainable=False)
    pm.eval()
    report = {"adapter": a.adapter, "rollouts": a.rollouts, "n": len(rows), "chunk": a.chunk, "token_budget": a.token_budget}

    with pm.disable_adapter():
        off_lp = run(pm, old_groups)
    report["adapter_off_vs_base"] = D.summarize(D.diff_lists(off_lp, raw_lp))  # esperado: zero (mesmos núcleos, mesma composição)

    old_bf16 = run(pm, old_groups)
    new_bf16 = run(pm, new_groups)
    report["bf16"] = {
        "vllm_mismatch": D.summarize(D.diff_lists(old_bf16, vllm_lp)),  # treinador - vLLM (temperatura 1)
        "noise_old_vs_new": D.summarize(D.diff_lists(new_bf16, old_bf16)),  # o `log s` do primeiro passo, só por composição de batch
    }

    inner = pm.base_model.model
    assert hasattr(inner, "lm_head"), "o modelo não tem lm_head"
    original = D.fp32_head(inner)
    try:
        old_fp32 = run(pm, old_groups)
        new_fp32 = run(pm, new_groups)
    finally:
        inner.lm_head = original
    report["fp32_head"] = {
        "vllm_mismatch": D.summarize(D.diff_lists(old_fp32, vllm_lp)),
        "noise_old_vs_new": D.summarize(D.diff_lists(new_fp32, old_fp32)),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("EXIT_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("vllm", "hf"):
        p = sub.add_parser(name)
        p.add_argument("--rollouts", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--n", type=int, default=96)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-len", type=int, default=3072)
        if name == "vllm":
            p.add_argument("--model", required=True, help="pasta do modelo fundido que gerou os rollouts")
            p.add_argument("--mem", type=float, default=0.80)
        else:
            p.add_argument("--adapter", required=True, help="adaptador que, fundido, gerou os rollouts")
            p.add_argument("--vllm-json", required=True)
            p.add_argument("--token-budget", type=int, default=3072)
            p.add_argument("--chunk", type=int, default=24, help="sequências por 'minibatch de treino' na composição nova")
    a = ap.parse_args()
    return stage_vllm(a) if a.stage == "vllm" else stage_hf(a)


if __name__ == "__main__":
    sys.exit(main())

"""GRPO, passo 2: atualiza o adaptador com as respostas e vantagens de `grpo_rollout.py` (roda no ambiente do treino).

Perda por token estilo PPO com corte assimétrico (0,2 / 0,28), sem KL, normalizada pelo total de tokens de resposta do
minibatch (como no SFT, a acumulação de gradiente dá o mesmo gradiente de um batch único). As log-probabilidades "antigas"
vêm de uma passada sem gradiente do próprio treino antes de qualquer atualização, então a razão começa em 1 (não depende da
diferença numérica entre vLLM e Transformers). Só treina nas respostas com vantagem diferente de zero. Um passo do otimizador
por minibatch de `--mb-seqs` respostas, uma passada pelos dados. AdamW com o estado guardado entre iterações.
  PYTHONPATH=src python scripts/grpo_train.py --rollouts runs/grpo/x/it000/rollouts.jsonl --adapter-in runs/train/adalberto/adapter --out runs/grpo/x/it000
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from lumi_ft.train import grpo as R
from lumi_ft.train import sft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--adapter-in", required=True)
    ap.add_argument("--ckpt-in", default="", help="estado do AdamW da iteração anterior (opcional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mb-seqs", type=int, default=64, help="respostas por passo do otimizador")
    ap.add_argument("--token-budget", type=int, default=3072)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--ckpt", default="full", choices=["full", "none"],
                    help="gradient checkpointing (padrão: full). Sem ele, uma só resposta de ~2.350 tokens estoura os 24 GiB no backward (OOM no ciclo 0 do grpo1)")
    ap.add_argument("--ratio", default="token", choices=["token", "seq"], help="razão de importância por token (GRPO/PPO) ou da sequência (GSPO)")
    ap.add_argument("--eps-low", type=float, default=float(os.environ["GRPO_EPS_LOW"]) if "GRPO_EPS_LOW" in os.environ else None,
                    help="padrão: variável GRPO_EPS_LOW; senão 0,2 (token) ou 3e-4 (seq)")
    ap.add_argument("--eps-high", type=float, default=float(os.environ["GRPO_EPS_HIGH"]) if "GRPO_EPS_HIGH" in os.environ else None,
                    help="padrão: variável GRPO_EPS_HIGH; senão 0,28 (token) ou 4e-4 (seq)")
    ap.add_argument("--old-same-batches", action="store_true", default=os.environ.get("GRPO_OLD_SAME_BATCHES", "0") == "1",
                    help="calcula as log-probabilidades antigas nos mesmos micro-batches que o treino vai usar (mesma composição e padding), "
                         "para o ruído numérico do primeiro passo sumir; padrão: variável GRPO_OLD_SAME_BATCHES=1 (ou desligado = como antes)")
    ap.add_argument("--entropy-only", action="store_true", help="só mede a entropia do modelo nas respostas (sem treinar) e grava <out>/entropy.json")
    ap.add_argument("--max-samples", type=int, default=0, help="com --entropy-only: usa só N respostas (sorteio fixo pela semente)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--positives-only", action="store_true",
                    help="RFT/RAFT: treina só nas respostas acima da média do grupo (vantagem 1 e razão do PPO), sem gradiente negativo")
    ap.add_argument("--mask-truncated", action="store_true", help="não treina nas respostas truncadas (padrão: treina, com vantagem negativa)")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("GRPO_SEED", "0")), help="padrão: variável GRPO_SEED (ou 0)")
    ap.add_argument("--logit-temp", type=float, default=float(os.environ.get("GRPO_LOGIT_TEMP", "1.0")),
                    help="divide os logits por T no cálculo das log-probabilidades (use a temperatura da amostragem); padrão: variável GRPO_LOGIT_TEMP (ou 1,0 = como antes)")
    a = ap.parse_args()
    if a.eps_low is None:
        a.eps_low = 0.2 if a.ratio == "token" else 3e-4
    if a.eps_high is None:
        a.eps_high = 0.28 if a.ratio == "token" else 4e-4

    import torch

    from lumi_ft import gpu
    from lumi_ft.modeling import peft_patch
    from lumi_ft.modeling.load import load_model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in Path(a.rollouts).read_text().splitlines() if line.strip()]
    if a.entropy_only:  # todas as respostas, com ou sem sinal; um sorteio fixo se pedido
        use = [s for s in samples if len(s["prompt_ids"]) + len(s["completion_ids"]) <= a.max_seq_len]
        if a.max_samples and len(use) > a.max_samples:
            use = random.Random(a.seed).sample(use, a.max_samples)
    else:
        use = [s for s in samples if (s["advantage"] > 0 if a.positives_only else s["advantage"] != 0) and (s["finished"] or not a.mask_truncated)
               and len(s["prompt_ids"]) + len(s["completion_ids"]) <= a.max_seq_len]
    rows = [{"input_ids": s["prompt_ids"] + s["completion_ids"], "n_prompt": len(s["prompt_ids"]),
             "loss_end": len(s["prompt_ids"]) + len(s["completion_ids"]), "kind": s["kind"]} for s in use]
    adv = [1.0 if a.positives_only else s["advantage"] for s in use]
    log = out / "train_log.jsonl"
    info = {"logit_temp": a.logit_temp, "seed": a.seed, "eps": [a.eps_low, a.eps_high], "old_same_batches": a.old_same_batches, "samples": len(samples), "used": len(use), "reward_mean_all": round(sum(s["reward"] for s in samples) / max(1, len(samples)), 4)}
    print(info, flush=True)
    if not use:  # nenhum grupo com sinal: o adaptador fica como está
        import shutil
        shutil.copytree(a.adapter_in, out / "adapter", dirs_exist_ok=True)
        gpu.append_jsonl(log, {"event": "sem_sinal", **info})
        print("nenhuma resposta com vantagem diferente de zero; adaptador copiado sem mudança")
        return 0

    cfg = json.loads((Path(a.adapter_in) / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        peft_patch.apply()
    from peft import PeftModel

    torch.manual_seed(a.seed)
    model = load_model("native")
    model.config.use_cache = False
    if a.ckpt == "full":  # só age no modo train: a passada das log-probabilidades antigas (eval, sem gradiente) não muda
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = PeftModel.from_pretrained(model, a.adapter_in, is_trainable=True)
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert params and all(p.dtype == torch.float32 for p in params.values()), "o adaptador deve estar em FP32"
    opt = torch.optim.AdamW(list(params.values()), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if a.ckpt_in and Path(a.ckpt_in).exists():
        state = torch.load(a.ckpt_in, map_location="cpu")
        opt.load_state_dict(state["opt"])
    gpu.reset_peak()

    def forward_lp(mb: list[int], entropy: bool = False):
        b = sft.collate(rows, mb)
        ids, mask = b["input_ids"].cuda(non_blocking=True), b["attention_mask"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = pm(input_ids=ids, attention_mask=mask).logits
        lps = [R.completion_logprobs(logits[k], ids[k], rows[i]["n_prompt"], rows[i]["loss_end"], a.logit_temp) for k, i in enumerate(mb)]
        if not entropy:
            return lps
        ents = [R.completion_entropy(logits[k], rows[i]["n_prompt"], rows[i]["loss_end"]) for k, i in enumerate(mb)]
        return lps, ents

    # 1) log-probabilidades antigas (sem gradiente) e a entropia do modelo nas respostas
    t0 = time.time()
    old: dict[int, torch.Tensor] = {}
    ent_sum, ent_n = {}, {}
    pm.eval()
    with torch.no_grad():
        mini = R.minibatch_indices(len(rows), a.mb_seqs, a.seed)
        old_groups = ([mb for idxs in mini for mb in sft.micro_batches(rows, idxs, a.token_budget)] if a.old_same_batches
                      else sft.micro_batches(rows, list(range(len(rows))), a.token_budget))
        for mb in old_groups:
            lps, ents = forward_lp(mb, entropy=True)
            for i, lp, en in zip(mb, lps, ents):
                old[i] = lp.cpu()
                k = rows[i]["kind"]
                ent_sum[k] = ent_sum.get(k, 0.0) + float(en.sum())
                ent_n[k] = ent_n.get(k, 0) + en.numel()
    pm.train()
    t_old = time.time() - t0
    entropy = {k: round(ent_sum[k] / ent_n[k], 4) for k in ent_sum}
    entropy["all"] = round(sum(ent_sum.values()) / max(1, sum(ent_n.values())), 4)
    entropy["mean_logp"] = round(sum(float(v.sum()) for v in old.values()) / max(1, sum(v.numel() for v in old.values())), 4)
    info["entropy"] = entropy
    print({"entropy": entropy}, flush=True)
    if a.entropy_only:
        (out / "entropy.json").write_text(json.dumps({"adapter": a.adapter_in, "rollouts": a.rollouts, **info, "samples_used": len(rows)}, indent=1))
        print("EXIT_OK")
        return 0

    # 2) um passo do otimizador por minibatch
    t1, steps, clipped_all, ratio_all, n_all = time.time(), 0, 0.0, 0.0, 0
    tot, first_abs_log_s = Counter(), None  # diagnóstico do corte (só registro; a perda não muda)
    for idxs in mini:
        N = sum(rows[i]["loss_end"] - rows[i]["n_prompt"] for i in idxs)
        loss_total = 0.0
        mbs = Counter()  # contadores deste passo do otimizador
        for mb in sft.micro_batches(rows, idxs, a.token_budget):
            lps = forward_lp(mb)
            loss_fn = R.ppo_clip_loss_sum if a.ratio == "token" else R.gspo_loss_sum
            loss = sum(loss_fn(lp, old[i].cuda(), adv[i], a.eps_low, a.eps_high) for i, lp in zip(mb, lps)) / N
            loss.backward()
            loss_total += float(loss.detach())
            with torch.no_grad():
                for i, lp in zip(mb, lps):
                    o_i = old[i].cuda()
                    r = torch.exp(lp.detach() - o_i)
                    ratio_all += float(r.sum())
                    # LEGADO: por token contra os limites do modo em uso. No GSPO (corte da SEQUÊNCIA) isto NÃO mede o corte da perda.
                    clipped_all += float(((r < 1 - a.eps_low) | (r > 1 + a.eps_high)).sum())
                    n_all += r.numel()
                    if a.ratio == "seq":  # corte real do GSPO: por sequência (A > 0 e s > 1+eps_alto, ou A < 0 e s < 1-eps_baixo)
                        cs = R.seq_clip_stats(lp, o_i, adv[i], a.eps_low, a.eps_high)
                        mbs["n"] += 1
                        mbs["outside"] += cs["outside"]
                        mbs["active"] += cs["active"]
                        mbs["abs_log_s"] += abs(cs["log_s"])
                        mbs["tok"] += cs["n"]
                        mbs["active_tok"] += cs["n"] * cs["active"]
                    else:  # modo por token: corte ativo (gradiente zero) do PPO
                        na, nt = R.token_clip_active(lp, o_i, adv[i], a.eps_low, a.eps_high)
                        mbs["n"] += 1
                        mbs["tok"] += nt
                        mbs["active_tok"] += na
            del lps, loss
        gnorm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), a.clip))
        if not math.isfinite(loss_total) or not math.isfinite(gnorm):
            raise RuntimeError(f"loss ou gradiente não finito: loss={loss_total} gnorm={gnorm}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
        rec = {"step": steps, "loss": round(loss_total, 5), "grad_norm": round(gnorm, 3), "tokens": N, **gpu.peak_gib()}
        if a.ratio == "seq":
            rec |= {"seq_clip_active": round(mbs["active"] / max(1, mbs["n"]), 4), "mean_abs_log_s": round(mbs["abs_log_s"] / max(1, mbs["n"]), 7)}
            first_abs_log_s = rec["mean_abs_log_s"] if first_abs_log_s is None else first_abs_log_s  # 1º passo: ruído numérico entre as duas passadas
        else:
            rec |= {"tok_clip_active": round(mbs["active_tok"] / max(1, mbs["tok"]), 4)}
        tot.update(mbs)
        gpu.append_jsonl(log, rec)
        print(rec, flush=True)
    summary = {"event": "end", **info, "ratio": a.ratio, "mask_truncated": a.mask_truncated, "positives_only": a.positives_only, "steps": steps, "old_logp_seconds": round(t_old), "train_seconds": round(time.time() - t1),
               "mean_ratio": round(ratio_all / max(1, n_all), 4), "clip_fraction": round(clipped_all / max(1, n_all), 4), **gpu.peak_gib()}
    # `clip_fraction` e `mean_ratio` acima são POR TOKEN (legado). O diagnóstico do corte da perda vem abaixo, com os denominadores:
    if a.ratio == "seq":
        summary |= {"seq_clip_active_frac": round(tot["active"] / max(1, tot["n"]), 4),      # sequências com gradiente zerado pelo corte / sequências
                    "seq_outside_frac": round(tot["outside"] / max(1, tot["n"]), 4),         # sequências com s fora do intervalo / sequências
                    "seq_clip_active_token_frac": round(tot["active_tok"] / max(1, tot["tok"]), 4),  # tokens dessas sequências / tokens
                    "mean_abs_log_s": round(tot["abs_log_s"] / max(1, tot["n"]), 7), "first_step_abs_log_s": first_abs_log_s, "n_seq": int(tot["n"])}
    else:
        summary |= {"tok_clip_active_frac": round(tot["active_tok"] / max(1, tot["tok"]), 4), "n_seq": int(tot["n"])}
    gpu.append_jsonl(log, summary)
    pm.save_pretrained(out / "adapter")
    tmp = out / "ckpt.tmp"
    torch.save({"opt": opt.state_dict()}, tmp)
    tmp.replace(out / "ckpt.pt")
    print(json.dumps(summary))
    print("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fase anti-laço, passo 2: treina o adaptador com a perda de preferência no token que inicia o laço (roda no ambiente do treino).

Parte de um adaptador (o do melhor candidato) e continua treinando os mesmos pesos LoRA. Para cada evento de `antidoom_mine.py`: passada do prefixo
(prompt + resposta até o início da repetição), logits do último token; a referência são os logits do adaptador de partida, calculados uma vez antes
de treinar. Parada antecipada: a cada `--eval-every` passos mede, nos eventos de validação (fora do treino), a fração em que o melhor escolhido passa
o rejeitado (`chosen_win`); para quando chega a `--target-win` (o Antidoom relata que treinar além disso degrada o modelo).
  PYTHONPATH=src python scripts/antidoom_train.py --events runs/antidoom/r1/mine/events.jsonl --adapter-in runs/grpo/gspo1/it007/adapter --out runs/antidoom/r1/train
Saída em --out: adapter/ (o da parada), adapter_stepN/ (a cada avaliação), train_log.jsonl, resumo em summary.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, nargs="+")
    ap.add_argument("--adapter-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8, help="eventos por passo do otimizador")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=8, help="passos entre as avaliações na validação")
    ap.add_argument("--target-win", type=float, default=0.35)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam-rest", type=float, default=5.0)
    ap.add_argument("--lam-cr", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-chosen", type=int, default=20)
    ap.add_argument("--max-rank", type=int, default=3, help="descarta o evento se o rejeitado não está entre os N maiores logits da referência (descompasso vLLM x treino)")
    ap.add_argument("--max-seq-len", type=int, default=3584)
    ap.add_argument("--smoke", action="store_true", help="teste de fumaça: 6 eventos, 2 passos, sem parada antecipada")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    from lumi_ft import gpu
    from lumi_ft.modeling import peft_patch
    from lumi_ft.modeling.load import EOS_IDS, load_model, load_tokenizer
    from lumi_ft.train import antidoom as A

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "train_log.jsonl"
    events = [json.loads(line) for p in a.events for line in Path(p).read_text().splitlines() if line.strip()]
    events = [e for e in events if len(e["prompt_ids"]) + len(e["prefix_ids"]) <= a.max_seq_len]
    if a.smoke:
        events = events[:6]
        a.max_steps, a.eval_every, a.batch = 2, 1, 3
    print({"eventos": len(events)}, flush=True)
    if len(events) < 4:
        print("poucos eventos de laço; nada a treinar")
        return 1

    cfg = json.loads((Path(a.adapter_in) / "adapter_config.json").read_text())
    if cfg.get("use_dora"):
        peft_patch.apply()
    from peft import PeftModel

    torch.manual_seed(a.seed)
    tok = load_tokenizer()
    model = load_model("native")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = PeftModel.from_pretrained(model, a.adapter_in, is_trainable=True)
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert params and all(p.dtype == torch.float32 for p in params.values()), "o adaptador deve estar em FP32"
    opt = torch.optim.AdamW(list(params.values()), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    w32 = model.lm_head.weight.detach().float()  # cabeça em FP32 (uma cópia): a diferença z - z0 é pequena e o BF16 a esconderia
    gpu.reset_peak()

    def last_logits(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.backbone(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)[0]
        return h[0, -1].float() @ w32.T

    # 1) referência: logits do adaptador de partida e os escolhidos de cada evento
    special = set(EOS_IDS) | set(tok.all_special_ids)
    pm.eval()
    data, dropped = [], {"rejeitado_fora_do_topo": 0, "sem_escolhidos": 0}
    t0 = time.time()
    with torch.no_grad():
        for e in events:
            ids = e["prompt_ids"] + e["prefix_ids"]
            z0 = last_logits(ids)
            rank = int((z0 > z0[e["rejected"]]).sum())
            if rank >= a.max_rank:
                dropped["rejeitado_fora_do_topo"] += 1
                continue
            chosen = A.choose_candidates(z0, e["rejected"], tok.decode, top_k=a.top_k, max_chosen=a.max_chosen, exclude=special)
            if not chosen:
                dropped["sem_escolhidos"] += 1
                continue
            data.append({"ids": ids, "rejected": e["rejected"], "chosen": chosen, "z0": z0.half().cpu(), "rank": rank})
    print({"referencia_s": round(time.time() - t0), "usados": len(data), "descartados": dropped}, flush=True)
    if len(data) < 4:
        print("eventos de menos depois do filtro")
        return 1
    rng = random.Random(a.seed)
    rng.shuffle(data)
    n_val = 2 if a.smoke else max(8, int(len(data) * a.val_frac))
    val, train = data[:n_val], data[n_val:]

    def loss_of(d: dict):
        z = last_logits(d["ids"])
        return A.ftpo_loss(z, d["z0"].cuda().float(), d["chosen"], d["rejected"], beta=a.beta, lam_rest=a.lam_rest, lam_cr=a.lam_cr, tau=a.tau)

    def mean_metrics(ms: list[dict]) -> dict:
        return {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in ms[0]}

    def evaluate() -> dict:
        pm.eval()
        with torch.no_grad():
            ms = [loss_of(d)[1] for d in val]
        pm.train()
        return mean_metrics(ms)

    base = evaluate()
    info = {"events": len(events), "treino": len(train), "validacao": len(val), "descartados": dropped, "inicio_validacao": base,
            "lr": a.lr, "batch": a.batch, "beta": a.beta, "lam_rest": a.lam_rest, "lam_cr": a.lam_cr, "tau": a.tau, "target_win": a.target_win,
            "rank_do_rejeitado": {str(r): sum(d["rank"] == r for d in data) for r in range(a.max_rank)}}
    gpu.append_jsonl(log, {"event": "inicio", **info})
    print(info, flush=True)

    # 2) treino com parada antecipada na validação
    pm.train()
    order: list[int] = []
    step, stopped, hist = 0, None, []
    t1 = time.time()
    while step < a.max_steps and stopped is None:
        if len(order) < a.batch:
            order = list(range(len(train)))
            rng.shuffle(order)
        batch, order = order[:a.batch], order[a.batch:]
        opt.zero_grad(set_to_none=True)
        ms = []
        for i in batch:
            loss, m = loss_of(train[i])
            (loss / len(batch)).backward()
            ms.append(m)
        gn = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0))
        opt.step()
        step += 1
        row = {"step": step, "grad_norm": round(gn, 4), **mean_metrics(ms)}
        if step % a.eval_every == 0:
            v = evaluate()
            row["val"] = v
            hist.append((step, v))
            pm.save_pretrained(out / f"adapter_step{step}")
            if not a.smoke and v["chosen_win"] >= a.target_win:
                stopped = step
        gpu.append_jsonl(log, row)
        if step % 4 == 0 or "val" in row:
            print(row, flush=True)
    pm.save_pretrained(out / "adapter")
    summary = {**info, "passos": step, "parou_em": stopped, "treino_s": round(time.time() - t1), "validacao_por_passo": hist,
               "fim_validacao": evaluate(), "pico_gib": gpu.peak_gib()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

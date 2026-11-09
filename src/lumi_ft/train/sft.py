"""SFT com LoRA ou DoRA sobre o Lumi: laço próprio, retomável, micro-batches por orçamento de tokens.

- Dados: `train.parquet` da fase 1 (`input_ids`, `n_prompt`, `loss_end`); os labels saem daí.
- Batch efetivo de `eff_batch` sequências (embaralhadas por época, semente fixa). Dentro dele, os exemplos são
  ordenados por comprimento e agrupados em micro-batches de no máximo `token_budget` tokens (com padding).
- A loss é a soma sobre os tokens-alvo dividida pelo total de tokens-alvo do batch efetivo, então a acumulação de
  gradiente dá o mesmo gradiente de um batch único (o resultado não depende de como o batch foi fatiado).
- Sem gradient checkpointing (PRD); autocast BF16; adaptador em FP32; AdamW; cosseno com warmup.
- Retomável: `ckpt/last.pt` a cada `save_every` passos (adaptador, otimizador, passo); a ordem dos dados depende só
  de (semente, época), então retomar no meio de uma época pula os batches já consumidos.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

PAD_ID = 0


# ---------------------------------------------------------------------------------- dados
def load_rows(path: str, *, mix: str = "75/25", frac: float | None = None, viab: bool = False) -> list[dict]:
    """Lê o parquet da fase 1. `mix` '100/0' fica só com raciocínio; `viab` pega o subconjunto de viabilidade;
    `frac` pega os exemplos com bucket < frac * 10.000 (20% = 0.2; contém a mesma proporção de cada grupo)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if mix == "100/0":
        rows = [r for r in rows if r["mode"] == "think"]
    elif mix != "75/25":
        raise ValueError(f"mix desconhecido: {mix}")
    if viab:
        rows = [r for r in rows if r["viab"]]
    if frac is not None:
        rows = [r for r in rows if r["bucket"] < frac * 10_000]
    return rows


def labels_for(row: dict) -> list[int]:
    ids = row["input_ids"]
    return [-100] * row["n_prompt"] + ids[row["n_prompt"]: row["loss_end"]] + [-100] * (len(ids) - row["loss_end"])


def epoch_batches(n: int, eff_batch: int, seed: int, epoch: int) -> list[list[int]]:
    """Índices de cada batch efetivo da época. Só depende de (n, eff_batch, seed, epoch)."""
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return [order[i: i + eff_batch] for i in range(0, n, eff_batch)]


def micro_batches(rows: list[dict], idxs: list[int], token_budget: int) -> list[list[int]]:
    """Agrupa por comprimento (decrescente) em micro-batches com `n * maior_comprimento <= token_budget`."""
    order = sorted(idxs, key=lambda i: (-len(rows[i]["input_ids"]), i))
    groups: list[list[int]] = []
    for i in order:
        if groups and (len(groups[-1]) + 1) * len(rows[groups[-1][0]]["input_ids"]) <= token_budget:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def collate(rows: list[dict], idxs: list[int]) -> dict:
    """Padding à direita (a atenção é causal e o padding fica fora da loss)."""
    import torch

    seqs = [rows[i]["input_ids"] for i in idxs]
    labs = [labels_for(rows[i]) for i in idxs]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    labels = torch.full((len(seqs), L), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, (s, l) in enumerate(zip(seqs, labs)):
        ids[k, : len(s)] = torch.tensor(s)
        labels[k, : len(l)] = torch.tensor(l)
        mask[k, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def n_targets(rows: list[dict], idxs: list[int]) -> int:
    """Tokens com loss: os labels deslocados (o primeiro token nunca é previsto, e é sempre do prompt)."""
    return sum(max(0, r["loss_end"] - r["n_prompt"]) for r in (rows[i] for i in idxs))


def batch_loss_sum(logits, labels, loss_type: str = "nll"):
    """Soma da cross-entropy sobre os tokens-alvo, sem copiar os logits (desloca os labels, não os logits).

    `loss_type="dft"` (Dynamic Fine-Tuning, o `loss_type="dft"` do SFTConfig do TRL): cada token pesa a própria probabilidade, sem gradiente
    nesse peso: `-sg(p) * log p`. O padrão `nll` é a cross-entropy de sempre e não muda nenhuma rodada antiga."""
    import torch
    import torch.nn.functional as F

    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    flat, tgt = logits.reshape(-1, logits.shape[-1]).float(), shifted.reshape(-1)
    if loss_type == "nll":
        return F.cross_entropy(flat, tgt, ignore_index=-100, reduction="sum")
    if loss_type == "dft":
        nll = F.cross_entropy(flat, tgt, ignore_index=-100, reduction="none")  # 0 nos tokens ignorados
        return (torch.exp(-nll).detach() * nll).sum()
    raise ValueError(f"loss_type desconhecido: {loss_type!r} (nll ou dft)")


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float = 0.03) -> float:
    """Warmup linear e cosseno até zero. `step` conta de 0."""
    warm = max(1, round(total * warmup_frac))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


# --------------------------------------------------------------------------------- treino
@dataclass
class TrainConfig:
    arm: str = "lora"  # lora | dora
    r: int = 16
    lr: float = 1e-4
    epochs: int = 1
    eff_batch: int = 32
    token_budget: int = 0  # 0 = padrão do braço (2.048 no LoRA, 1.024 no DoRA)
    warmup: float = 0.03
    clip: float = 1.0
    seed: int = 0
    mix: str = "75/25"
    frac: float | None = None
    viab: bool = False
    data: str = "data/train.parquet"
    valid: str = "data/valid.parquet"
    out: str = "runs/train/run"
    save_every: int = 200
    log_every: int = 10
    max_steps: int = 0  # >0: para depois de N passos (depuração)
    targets: str = "all"  # all = os 71 módulos do PRD | lista de tipos: mamba, mlp, attn (ex.: "mlp" ou "mamba,attn")
    # Estudo de capacidade (padrões = PRD: nada muda nas rodadas antigas)
    alpha: float | None = None  # None = 2 * r (escala alpha/r = 2)
    rslora: bool = False  # escala alpha/raiz(r)
    layers: str = ""  # faixa fechada de camadas, "0-20" (vazio = todas)
    rank_pattern: str = ""  # JSON {regex do final do nome do módulo: rank}, ex.: '{"in_proj": 4, "up_proj|down_proj": 32}'
    alpha_pattern: str = ""  # JSON idem para alpha (vazio = mantém a escala alpha/r de cada módulo)
    lr_mult: str = ""  # JSON {regex do nome do parâmetro: multiplicador do lr}; vale o primeiro que casar, senão 1
    loss: str = "nll"  # nll | dft (a perda de validação continua sendo a nll)


def default_budget(arm: str) -> int:
    return 1024 if arm == "dora" else 2048


def lr_multiplier(name: str, lr_mult: dict[str, float]) -> float:
    """Multiplicador do lr de um parâmetro: o do primeiro regex de `lr_mult` que casa (re.search) com o nome, senão 1."""
    for rx, m in lr_mult.items():
        if re.search(rx, name):
            return float(m)
    return 1.0


def param_groups(params: dict, lr: float, lr_mult: dict[str, float]) -> list[dict]:
    """Um grupo do otimizador por multiplicador distinto (ordem de primeira aparição, estável entre execuções)."""
    by: dict[float, list] = {}
    for n, p in params.items():
        by.setdefault(lr_multiplier(n, lr_mult), []).append(p)
    return [{"params": ps, "lr": lr * m, "mult": m} for m, ps in by.items()]


def _save_ckpt(path: Path, params: dict, opt, step: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "opt": opt.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def evaluate_loss(model, rows: list[dict], token_budget: int) -> float:
    """Loss média por token-alvo (sem gradiente) numa lista de exemplos."""
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for mb in micro_batches(rows, list(range(len(rows))), token_budget):
            b = collate(rows, mb)
            logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
            total += float(batch_loss_sum(logits, b["labels"].cuda()))
            n += n_targets(rows, mb)
            del logits
    model.train()
    return total / max(1, n)


def train(cfg: TrainConfig) -> dict:
    import torch

    from lumi_ft import gpu
    from lumi_ft.modeling.load import load_model
    from lumi_ft.modeling.peft_setup import add_adapter

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = cfg.token_budget or default_budget(cfg.arm)
    rows = load_rows(cfg.data, mix=cfg.mix, frac=cfg.frac, viab=cfg.viab)
    valid = load_rows(cfg.valid, mix=cfg.mix) if cfg.valid and Path(cfg.valid).exists() else []
    steps_per_epoch = math.ceil(len(rows) / cfg.eff_batch)
    total = steps_per_epoch * cfg.epochs
    tokens = sum(r["n_tokens"] for r in rows)
    meta = {**cfg.__dict__, "token_budget": budget, "n_train": len(rows), "n_valid": len(valid), "train_tokens": tokens,
            "steps_per_epoch": steps_per_epoch, "total_steps": total}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log = out / "train_log.jsonl"

    torch.manual_seed(cfg.seed)
    model = load_model("native")
    model.config.use_cache = False  # sem gradient checkpointing (PRD): memória medida na fase 0
    pm = add_adapter(model, r=cfg.r, use_dora=(cfg.arm == "dora"), targets=cfg.targets, alpha=cfg.alpha, layers=cfg.layers,
                     rslora=cfg.rslora, rank_pattern=json.loads(cfg.rank_pattern) if cfg.rank_pattern else None,
                     alpha_pattern=json.loads(cfg.alpha_pattern) if cfg.alpha_pattern else None)
    pm.train()
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(param_groups(params, cfg.lr, json.loads(cfg.lr_mult) if cfg.lr_mult else {}),
                            lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    step, ckpt = 0, out / "ckpt" / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        for n, p in params.items():
            p.data.copy_(state["params"][n])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"retomando do passo {step}", flush=True)
    gpu.append_jsonl(log, {"event": "start", "resume_step": step, "trainable_params": sum(p.numel() for p in params.values()),
                           **meta, **gpu.snapshot()})
    gpu.reset_peak()
    t_begin, tok_seen, last = time.time(), 0, None
    for epoch in range(cfg.epochs):
        for bi, idxs in enumerate(epoch_batches(len(rows), cfg.eff_batch, cfg.seed, epoch)):
            gstep = epoch * steps_per_epoch + bi
            if gstep < step:
                continue
            lr = lr_at(gstep, total, cfg.lr, cfg.warmup)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("mult", 1.0)
            N = n_targets(rows, idxs)
            if N == 0:
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_total, padded = 0.0, 0
            for mb in micro_batches(rows, idxs, budget):
                b = collate(rows, mb)
                ids, mask, labels = (b[k].cuda(non_blocking=True) for k in ("input_ids", "attention_mask", "labels"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = pm(input_ids=ids, attention_mask=mask).logits
                loss = batch_loss_sum(logits, labels, cfg.loss) / N
                loss.backward()
                loss_total += float(loss.detach())
                padded += ids.numel()
                del logits, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), cfg.clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            step = gstep + 1
            tok_seen += padded
            if not math.isfinite(loss_total) or not math.isfinite(gnorm):
                raise RuntimeError(f"loss ou gradiente não finito no passo {step}: loss={loss_total} gnorm={gnorm}")
            last = {"step": step, "epoch": epoch + 1, "loss": round(loss_total, 4), "lr": lr, "grad_norm": round(gnorm, 3),
                    "step_s": round(dt, 2), "tok_per_s": round(padded / dt), "target_tokens": N, **gpu.peak_gib()}
            if step % cfg.log_every == 0 or step == total:
                gpu.append_jsonl(log, last)
                print(last, flush=True)
            if cfg.save_every and step % cfg.save_every == 0:
                _save_ckpt(ckpt, params, opt, step)
            if cfg.max_steps and step >= cfg.max_steps:
                break
        if cfg.max_steps and step >= cfg.max_steps:
            break
        if not (out / f"adapter_epoch{epoch + 1}").exists():
            vloss = evaluate_loss(pm, valid, budget) if valid else None
            pm.save_pretrained(out / f"adapter_epoch{epoch + 1}")
            _save_ckpt(ckpt, params, opt, step)
            gpu.append_jsonl(log, {"event": "epoch_end", "epoch": epoch + 1, "step": step, "valid_loss": vloss,
                                   "train_loss_last": last["loss"] if last else None})
            print(f"fim da época {epoch + 1}: valid_loss={vloss}", flush=True)
    pm.save_pretrained(out / "adapter")
    summary = {"event": "end", "steps": step, "wall_seconds": round(time.time() - t_begin), "padded_tokens_seen": tok_seen,
               **gpu.peak_gib(), **gpu.snapshot()}
    gpu.append_jsonl(log, summary)
    return summary

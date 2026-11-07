"""Fase 0 (A-14): pico de memória, maior batch viável e tokens/s do treino, por braço.

Mede passo completo (forward com loss, backward, AdamW) com gradient checkpointing não reentrante,
`use_cache=False`, pesos BF16, autocast BF16 (como o Trainer com bf16=True) e adaptador em FP32.
Usa sequências de comprimento cheio, sem padding, com loss em todas as posições: é o pior caso.
Para cada (braço, comprimento) sobe a escada de batch até a primeira falta de memória.
Retomável: linhas já gravadas em runs/phase0/train_resources.jsonl são puladas.

Uso: scripts/detach.sh --gpu measure-train PYTHONPATH=src python scripts/measure_train.py
     (smoke: ... --arms lora:16 --lengths 256 --ladder 1,2 --steps 1 --out runs/phase0/smoke_train.jsonl)
"""
import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from lumi_ft import gpu
from lumi_ft.modeling.load import free_gpu, load_model
from lumi_ft.modeling.peft_setup import add_adapter

VOCAB = 131072


def one_config(model, params, b: int, L: int, steps: int, seed: int) -> dict:
    """Roda `1 + steps` passos (o primeiro é aquecimento e não entra no tempo). Pode levantar OOM."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(1000, 100000, (b, L), device="cuda", generator=g)
    mask = torch.ones_like(ids)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    gpu.reset_peak()
    times, loss_v = [], None
    for i in range(1 + steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i > 0:
            times.append(time.perf_counter() - t)
        loss_v = float(loss.detach())
    assert loss_v == loss_v and abs(loss_v) < 1e4, f"loss inválida: {loss_v}"
    med = statistics.median(times)
    return {"status": "ok", "step_s": [round(x, 3) for x in times], "tok_per_s": round(b * L / med),
            "loss": round(loss_v, 3), **gpu.peak_gib()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="lora:16,lora:32,dora:16,dora:32")
    ap.add_argument("--lengths", default="1024,2048")
    ap.add_argument("--ladder", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="runs/phase0/train_resources.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    lengths = [int(x) for x in a.lengths.split(",")]
    ladder = [int(x) for x in a.ladder.split(",")]
    done = {(r["arm"], r["r"], r["L"], r["b"]): r for r in gpu.read_jsonl(out) if "arm" in r}
    gpu.append_jsonl(out, {"event": "start", **gpu.snapshot(), "torch": torch.__version__})

    for spec in a.arms.split(","):
        arm, r = spec.split(":")
        r = int(r)
        todo = [(L, b) for L in lengths for b in ladder if (arm, r, L, b) not in done]
        if not todo:
            continue
        model = load_model("native")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm = add_adapter(model, r=r, use_dora=(arm == "dora"))
        pm.train()
        params = [p for p in pm.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        dtypes = sorted({str(p.dtype) for p in params})
        print(f"== {arm} r={r}: {n_train/1e6:.1f} M parâmetros treináveis {dtypes}", flush=True)
        for L in lengths:
            oom = False
            for b in ladder:
                if (arm, r, L, b) in done:
                    oom = oom or done[(arm, r, L, b)]["status"] == "oom"
                    continue
                if oom:
                    break
                row = {"arm": arm, "r": r, "L": L, "b": b, "tokens": b * L, "trainable_params": n_train,
                       "adapter_dtypes": dtypes}
                try:
                    row |= one_config(pm, params, b, L, a.steps, seed=0)
                except torch.OutOfMemoryError:
                    row |= {"status": "oom", **gpu.peak_gib()}
                    oom = True
                gc.collect()
                torch.cuda.empty_cache()
                gpu.append_jsonl(out, row)
                print(row, flush=True)
        del pm, model, params
        free_gpu()
    gpu.append_jsonl(out, {"event": "end", **gpu.snapshot()})
    print("EXIT_OK")


if __name__ == "__main__":
    main()

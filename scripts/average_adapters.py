"""Média de pesos de adaptadores LoRA treinados a partir do mesmo ponto de partida (por exemplo, rodadas de RL com sementes diferentes).

Confere que os adaptadores têm as mesmas chaves, formas e configuração (r, alpha, módulos), faz a média tensor a tensor das matrizes A e B
(em float32, gravando no tipo do primeiro) e escreve um adaptador novo em --out, com o `adapter_config.json` do primeiro e um
`soup_report.json` (sha256 das entradas, distância relativa de cada adaptador à média). Como as rodadas partem do mesmo adaptador e se afastam
pouco dele (passos de RL pequenos), a média de A e de B separadas aproxima a média dos produtos B·A com erro de segunda ordem.
Uso: PYTHONPATH=src python scripts/average_adapters.py --adapters runs/grpo/gspo1/it003/adapter runs/grpo/gspo1_s1/it003/adapter --out runs/train/gspo1_soup2/adapter
Depois: scripts/merge_adapter.py --adapter <out> --out runs/merged/<nome>
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CONFIG_KEYS = ("r", "lora_alpha", "target_modules", "use_dora", "use_rslora", "base_model_name_or_path")


def average(states: list[dict], weights: list[float] | None = None) -> dict:
    """Média ponderada tensor a tensor; as chaves e as formas têm de ser iguais. Devolve no tipo do primeiro adaptador."""
    if not states:
        raise ValueError("nenhum adaptador")
    w = weights or [1.0 / len(states)] * len(states)
    if len(w) != len(states) or abs(sum(w) - 1.0) > 1e-6:
        raise ValueError("os pesos têm de somar 1 e ter um por adaptador")
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise ValueError(f"chaves diferentes: {sorted(set(s) ^ keys)[:3]}")
    out = {}
    for k in sorted(keys):
        shapes = {tuple(s[k].shape) for s in states}
        if len(shapes) != 1:
            raise ValueError(f"formas diferentes em {k}: {shapes}")
        acc = sum(wi * s[k].float() for wi, s in zip(w, states))
        out[k] = acc.to(states[0][k].dtype).contiguous()
    return out


def relative_distance(a: dict, mean: dict) -> float:
    """||a - média|| / ||média|| sobre todos os tensores."""
    num = sum(float(((a[k].float() - mean[k].float()) ** 2).sum()) for k in mean)
    den = sum(float((mean[k].float() ** 2).sum()) for k in mean)
    return (num / den) ** 0.5


def approximation_error(states: list[dict], mean: dict) -> dict:
    """Erro relativo (Frobenius) de média(B)·média(A) contra a média dos produtos B_i·A_i, módulo a módulo: mede o que a média de A e B
    separadas perde em relação à média dos deltas efetivos."""
    errs = []
    for ka in sorted(k for k in mean if "lora_A" in k):
        kb = ka.replace("lora_A", "lora_B")
        true = sum(s[kb].float() @ s[ka].float() for s in states) / len(states)
        approx = mean[kb].float() @ mean[ka].float()
        errs.append(float((approx - true).norm() / true.norm().clamp_min(1e-12)))
    errs.sort()
    return {"modulos": len(errs), "maximo": round(errs[-1], 6), "mediana": round(errs[len(errs) // 2], 6)} if errs else {"modulos": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapters", required=True, nargs="+", help="pastas de adaptador (com adapter_model.safetensors e adapter_config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    a = ap.parse_args()
    dirs = [Path(d) for d in a.adapters]
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    for c in cfgs[1:]:
        for k in CONFIG_KEYS:
            if c.get(k) != cfgs[0].get(k):
                raise ValueError(f"configuração diferente em {k}: {cfgs[0].get(k)} contra {c.get(k)}")
    states = [load_file(str(d / "adapter_model.safetensors")) for d in dirs]
    mean = average(states, a.weights)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(mean, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    shutil.copy(dirs[0] / "adapter_config.json", out / "adapter_config.json")
    report = {"inputs": {str(d): hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest() for d in dirs},
              "weights": a.weights or [1.0 / len(dirs)] * len(dirs),
              "distancia_relativa_a_media": {str(d): round(relative_distance(s, mean), 6) for d, s in zip(dirs, states)},
              "erro_da_aproximacao_BA": approximation_error(states, mean),
              "tensores": len(mean)}
    (out / "soup_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

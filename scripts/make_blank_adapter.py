"""Cria um adaptador LoRA em branco (B = 0) sobre um modelo fundido, para começar um RL a partir dele.

O RL (`scripts/run_grpo.sh`) treina um adaptador em cima de uma base e funde o resultado para o vLLM. Para partir do T6-Lumi (o `thor_pt_antidoom_of`, que já é
SFT + Antidoom fundidos) e não do Lumi original, a base passa a ser a pasta do T6-Lumi (`--base` aqui; `GRPO_BASE` no treino e na fusão) e o adaptador de partida é
este: com B = 0 a fusão devolve a base sem mudar um bit, então as primeiras respostas do RL são as do T6-Lumi. Mesma configuração do SFT (r 16, alpha 32, os 71
módulos, `peft_setup.add_adapter`).

  PYTHONPATH=src python scripts/make_blank_adapter.py --base runs/merged/thor_pt_antidoom_of --out runs/rl_t6/init_adapter
  GRPO_BASE=runs/merged/thor_pt_antidoom_of scripts/run_grpo.sh NOME --init runs/rl_t6/init_adapter ...
"""
import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pasta de um modelo fundido (config e pesos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    from lumi_ft.modeling.peft_setup import add_adapter

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, trust_remote_code=False, device_map={"": "cpu"})
    pm = add_adapter(model, r=a.r, alpha=a.alpha)
    n_mod = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A.default"))
    zeros = all(bool((p == 0).all()) for n, p in pm.named_parameters() if "lora_B" in n)
    assert n_mod == 71, f"esperava 71 módulos LoRA, veio {n_mod}"
    assert zeros, "o B do LoRA deveria começar em zero"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(out)
    print(f"adaptador em branco gravado em {out}: {n_mod} módulos, r={a.r}, alpha={a.alpha}, sobre {a.base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

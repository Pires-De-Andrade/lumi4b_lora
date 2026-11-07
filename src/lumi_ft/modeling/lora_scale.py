"""Multiplica o efeito de um adaptador LoRA antes de fundir (interpolação entre o Lumi original e o ajustado).

`scale = 1` é o adaptador como treinado; `0` é o Lumi original. Só para LoRA: no DoRA a magnitude é recalculada da soma
dos pesos, então multiplicar o escalonamento não interpola de forma simples.
"""

from __future__ import annotations

import torch.nn as nn


def scale_lora(model: nn.Module, scale: float) -> int:
    """Multiplica o `scaling` (alpha/r) de cada módulo LoRA por `scale`. Devolve quantos módulos mudaram."""
    n = 0
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and hasattr(m, "lora_A") and sc:
            for k in sc:
                sc[k] = sc[k] * scale
            n += 1
    return n

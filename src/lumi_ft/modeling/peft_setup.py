"""Alvos e configuração do adaptador (LoRA ou DoRA), conforme o PRD."""
import re

import torch
from peft import LoraConfig, get_peft_model

from lumi_ft.modeling import peft_patch

TARGET_REGEX = r".*\.mixer\.(in_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"
EXPECTED_TARGETS = {"in_proj": 21, "up_proj": 17, "down_proj": 17,
                    "q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4}  # 71 no total


def lora_config(r: int = 16, use_dora: bool = False, alpha: int | None = None) -> LoraConfig:
    return LoraConfig(
        r=r, lora_alpha=alpha if alpha is not None else 2 * r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_REGEX, use_dora=use_dora,
    )


def target_names(model) -> list[str]:
    rx = re.compile(TARGET_REGEX)
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and rx.fullmatch(n)]


def count_by_type(names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in names:
        k = n.rsplit(".", 1)[-1]
        out[k] = out.get(k, 0) + 1
    return out


def check_targets(model) -> None:
    """Falha se o regex não casar exatamente com os módulos esperados, ou se casar com os proibidos."""
    names = target_names(model)
    assert count_by_type(names) == EXPECTED_TARGETS, count_by_type(names)
    for n in names:
        assert not any(x in n for x in ("out_proj", "conv1d", "embed", "lm_head")), n


def add_adapter(model, r: int = 16, use_dora: bool = False):
    check_targets(model)
    if use_dora:
        peft_patch.apply()  # linhas zeradas da in_proj: ver peft_patch.py
    return get_peft_model(model, lora_config(r=r, use_dora=use_dora))

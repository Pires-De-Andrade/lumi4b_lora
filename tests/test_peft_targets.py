"""Conjuntos de alvos do adaptador (all = 71 módulos do PRD, mlp = só up_proj e down_proj), sem carregar o Lumi."""

import re

import pytest
import torch.nn as nn

from lumi_ft.modeling import peft_setup as P


def fake_lumi() -> nn.Module:
    """Mesma contagem de módulos do Lumi: 21 Mamba (in_proj, out_proj), 17 MLP (up, down) e 4 atenção (q, k, v, o)."""
    layers = nn.ModuleDict()
    for i in range(21):
        layers[f"m{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({"in_proj": nn.Linear(2, 2), "out_proj": nn.Linear(2, 2)})})
    for i in range(17):
        layers[f"f{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({"up_proj": nn.Linear(2, 2), "down_proj": nn.Linear(2, 2)})})
    for i in range(4):
        layers[f"a{i}"] = nn.ModuleDict({"mixer": nn.ModuleDict({k: nn.Linear(2, 2) for k in ("q_proj", "k_proj", "v_proj", "o_proj")})})
    return nn.ModuleDict({"backbone": nn.ModuleDict({"layers": layers})})


def test_all_targets_unchanged():
    m = fake_lumi()
    assert len(P.target_names(m)) == 71 and P.count_by_type(P.target_names(m)) == P.EXPECTED_TARGETS
    P.check_targets(m)
    assert P.lora_config().target_modules == P.TARGET_REGEX


def test_mlp_targets_are_only_up_and_down():
    m = fake_lumi()
    names = P.target_names(m, "mlp")
    assert len(names) == 34 and P.count_by_type(names) == {"up_proj": 17, "down_proj": 17}
    P.check_targets(m, "mlp")
    rx = re.compile(P.lora_config(targets="mlp").target_modules)
    assert rx.fullmatch("backbone.layers.3.mixer.up_proj") and rx.fullmatch("backbone.layers.3.mixer.down_proj")
    assert not any(rx.fullmatch(f"backbone.layers.3.mixer.{k}") for k in ("in_proj", "out_proj", "q_proj", "o_proj"))


def test_wrong_counts_fail():
    m = fake_lumi()
    del m["backbone"]["layers"]["f0"]
    with pytest.raises(AssertionError):
        P.check_targets(m, "mlp")
    with pytest.raises(AssertionError):
        P.check_targets(m)


def test_train_config_and_cli_carry_targets():
    from lumi_ft.train import sft

    assert sft.TrainConfig().targets == "all" and sft.TrainConfig(targets="mlp").targets == "mlp"

"""Patch do DoRA para linhas zeradas. CPU, rápido, sem o modelo."""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from lumi_ft.modeling import peft_patch

ZERO_ROWS = (2, 5)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(16, 12, bias=False)
        with torch.no_grad():
            self.proj.weight[list(ZERO_ROWS)] = 0.0

    def forward(self, x):
        return self.proj(x)


def make(use_dora=True):
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["proj"], use_dora=use_dora)
    return get_peft_model(Tiny(), cfg)


@pytest.fixture(autouse=True)
def _restore():
    peft_patch.revert()
    yield
    peft_patch.revert()


def test_unpatched_dora_gives_nan_on_zero_rows():  # documenta o defeito do PEFT que o patch corrige
    out = make()(torch.randn(3, 16))
    assert torch.isnan(out).any()


def test_patched_fresh_adapter_keeps_output_and_zero_rows_stay_zero():
    x = torch.randn(3, 16)
    base = Tiny()(x)
    peft_patch.apply()
    out = make()(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, base, atol=1e-6)
    assert (out[:, list(ZERO_ROWS)] == 0).all()
    mag = make().base_model.model.proj.lora_magnitude_vector["default"].weight
    assert (mag[list(ZERO_ROWS)] == 0).all(), "magnitude inicial das linhas zeradas deve continuar 0"


def test_patch_does_not_change_models_without_zero_rows():
    class NoZero(Tiny):
        def __init__(self):
            super().__init__()
            nn.init.normal_(self.proj.weight)
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["proj"], use_dora=True)
    x = torch.randn(3, 16)

    def run():
        torch.manual_seed(1); m = get_peft_model(NoZero(), cfg)
        with torch.no_grad():
            m.base_model.model.proj.lora_B["default"].weight.normal_(0, 0.1)
        return m(x)
    a = run(); peft_patch.apply(); b = run()
    assert torch.equal(a, b)


def test_patched_grads_finite_and_merge_matches_active():
    peft_patch.apply()
    m = make()
    x = torch.randn(8, 16)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad(); m(x).pow(2).mean().backward(); opt.step()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
    with torch.no_grad():
        m.base_model.model.proj.lora_B["default"].weight.normal_(0, 0.1)
        active = m(x)
        merged = m.merge_and_unload()(x)
    assert torch.isfinite(merged).all()
    assert torch.allclose(active, merged, atol=1e-5), (active - merged).abs().max()
    assert (merged[:, list(ZERO_ROWS)] == 0).all(), "linhas mortas continuam mortas após a fusão"

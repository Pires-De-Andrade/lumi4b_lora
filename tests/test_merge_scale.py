"""Escala do adaptador LoRA (interpolação com o Lumi original) e limite de tokens do construtor (CPU, sem o Lumi)."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from lumi_ft.data import teacher as T
from lumi_ft.modeling.lora_scale import scale_lora


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.lin(x)


def _merged_weight(scale: float):
    torch.manual_seed(0)
    pm = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["lin"], init_lora_weights=False))
    w0 = pm.base_model.model.lin.base_layer.weight.detach().clone()
    a = pm.base_model.model.lin.lora_A["default"].weight.detach().clone()
    b = pm.base_model.model.lin.lora_B["default"].weight.detach().clone()
    delta = (b @ a) * (4 / 2)
    n = scale_lora(pm, scale)
    assert n == 1
    w = pm.merge_and_unload().lin.weight.detach()
    return w0, delta, w


def test_scale_half_is_halfway():
    w0, delta, w = _merged_weight(0.5)
    assert torch.allclose(w, w0 + 0.5 * delta, atol=1e-6)


def test_scale_one_is_trained_adapter():
    w0, delta, w = _merged_weight(1.0)
    assert torch.allclose(w, w0 + delta, atol=1e-6)


def test_scale_zero_is_original():
    w0, _, w = _merged_weight(0.0)
    assert torch.equal(w, w0)


class FakeTok:  # um token por caractere; o suficiente para o limite de tamanho
    pass


def test_build_rows_respects_max_len(monkeypatch):
    sizes = {"a": 100, "b": 3000}

    def fake_render(tok, system, user, answer, thought=None):
        n = sizes[answer]
        return {"n_tokens": n, "n_prompt": 10, "loss_end": n, "input_ids": list(range(n))}

    monkeypatch.setattr(T.GV, "render_example", fake_render)
    recs = [{"id": k, "problem_id": k, "question": "q", "thought": "t", "answer": k, "source": "s"} for k in sizes]
    rows, st = T.build_rows(None, recs, [], max_len=0)
    assert [r["id"] for r in rows] == ["a"] and st["too_long"] == 1 and st["max_len"] == T.GV.MAX_LEN
    rows, st = T.build_rows(None, recs, [], max_len=4096)
    assert sorted(r["id"] for r in rows) == ["a", "b"] and st["too_long"] == 0

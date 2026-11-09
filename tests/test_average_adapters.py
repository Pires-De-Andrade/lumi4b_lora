"""Média de pesos de adaptadores LoRA (scripts/average_adapters.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_spec = importlib.util.spec_from_file_location("average_adapters", Path(__file__).resolve().parents[1] / "scripts" / "average_adapters.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def _state(v, dtype=torch.bfloat16):
    return {"a.lora_A.weight": torch.full((2, 3), v, dtype=dtype), "a.lora_B.weight": torch.full((3, 2), 2 * v, dtype=dtype)}


def test_media_simples_e_tipo_do_primeiro():
    out = M.average([_state(1.0), _state(3.0)])
    assert torch.allclose(out["a.lora_A.weight"].float(), torch.full((2, 3), 2.0))
    assert torch.allclose(out["a.lora_B.weight"].float(), torch.full((3, 2), 4.0))
    assert out["a.lora_A.weight"].dtype == torch.bfloat16


def test_media_ponderada():
    out = M.average([_state(0.0), _state(4.0)], [0.75, 0.25])
    assert torch.allclose(out["a.lora_A.weight"].float(), torch.full((2, 3), 1.0))


def test_recusa_pesos_que_nao_somam_1():
    with pytest.raises(ValueError, match="somar 1"):
        M.average([_state(1.0), _state(2.0)], [0.5, 0.6])


def test_recusa_chaves_e_formas_diferentes():
    b = _state(1.0)
    b["extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="chaves diferentes"):
        M.average([_state(1.0), b])
    c = _state(1.0)
    c["a.lora_A.weight"] = torch.zeros((4, 3), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="formas diferentes"):
        M.average([_state(1.0), c])


def test_distancia_relativa_zero_para_iguais():
    s = _state(1.0)
    assert M.relative_distance(s, M.average([s, s])) == 0.0
    assert M.relative_distance(_state(1.0), M.average([_state(1.0), _state(3.0)])) == pytest.approx(0.5, rel=1e-3)


def test_erro_da_aproximacao_zero_para_iguais_e_positivo_para_diferentes():
    s = _state(1.0)
    assert M.approximation_error([s, s], M.average([s, s]))["maximo"] == 0.0
    a = {"m.lora_A.weight": torch.tensor([[1.0, 0.0]]), "m.lora_B.weight": torch.tensor([[1.0], [0.0]])}
    b = {"m.lora_A.weight": torch.tensor([[0.0, 1.0]]), "m.lora_B.weight": torch.tensor([[0.0], [1.0]])}
    err = M.approximation_error([a, b], M.average([a, b]))
    assert err["modulos"] == 1 and err["maximo"] > 0.1


def _write(d: Path, v, r=16):
    d.mkdir(parents=True)
    save_file(_state(v), str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": 32, "target_modules": ["x"], "use_dora": False}))


def test_main_grava_adaptador_e_relatorio(tmp_path, monkeypatch):
    _write(tmp_path / "a", 1.0)
    _write(tmp_path / "b", 3.0)
    out = tmp_path / "soup"
    monkeypatch.setattr(sys, "argv", ["x", "--adapters", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)])
    assert M.main() == 0
    got = load_file(str(out / "adapter_model.safetensors"))
    assert torch.allclose(got["a.lora_A.weight"].float(), torch.full((2, 3), 2.0))
    rep = json.loads((out / "soup_report.json").read_text())
    assert rep["tensores"] == 2 and len(rep["inputs"]) == 2
    assert json.loads((out / "adapter_config.json").read_text())["r"] == 16


def test_main_recusa_configuracao_diferente(tmp_path, monkeypatch):
    _write(tmp_path / "a", 1.0, r=16)
    _write(tmp_path / "b", 3.0, r=32)
    monkeypatch.setattr(sys, "argv", ["x", "--adapters", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "s")])
    with pytest.raises(ValueError, match="configuração diferente"):
        M.main()

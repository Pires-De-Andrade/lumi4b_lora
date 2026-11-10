"""Diferença exata T6-Lumi menos Lumi (handoff/spark1/delta.py): ida e volta reproduz o arquivo byte a byte, inclusive BF16 e FP32."""
import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("delta", Path(__file__).resolve().parents[1] / "handoff" / "spark1" / "delta.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _par(tmp_path):
    g = torch.Generator().manual_seed(0)
    base = {
        "a.weight": torch.randn(64, 48, generator=g).to(torch.bfloat16),
        "b.weight": torch.randn(33, 17, generator=g).to(torch.bfloat16),
        "c.A_log": torch.randn(7, generator=g),  # FP32
        "d.same": torch.randn(5, 5, generator=g).to(torch.bfloat16),
    }
    model = dict(base)
    model["a.weight"] = (base["a.weight"].float() + 1e-3 * torch.randn(64, 48, generator=g)).to(torch.bfloat16)
    model["b.weight"] = base["b.weight"].clone()
    model["b.weight"][0, 0] = -base["b.weight"][0, 0] - 1  # muda sinal e expoente: a diferença dos bits dá a volta
    model["c.A_log"] = base["c.A_log"] + 1e-6
    pb, pm = tmp_path / "base.safetensors", tmp_path / "model.safetensors"
    save_file(base, str(pb), metadata={"format": "pt"})
    save_file(model, str(pm), metadata={"format": "pt"})
    return pb, pm


def test_ida_e_volta_identica_byte_a_byte(tmp_path):
    pb, pm = _par(tmp_path)
    r = D.pack(str(pb), str(pm), str(tmp_path / "t6.delta"))
    assert r["alterados"] == 3 and r["total"] == 4  # d.same fica de fora
    out = tmp_path / "out.safetensors"
    assert D.unpack(str(tmp_path / "t6.delta"), str(out), str(pb)) == {"alterados": 3, "total": 4}
    assert out.read_bytes() == pm.read_bytes()


def test_arquivo_truncado_e_recusado(tmp_path):
    import pytest

    pb, pm = _par(tmp_path)
    D.pack(str(pb), str(pm), str(tmp_path / "t6.delta"))
    dados = (tmp_path / "t6.delta").read_bytes()
    (tmp_path / "cortado.delta").write_bytes(dados[: len(dados) // 2])
    with pytest.raises((SystemExit, Exception)):
        D.unpack(str(tmp_path / "cortado.delta"), str(tmp_path / "x.safetensors"), str(pb))

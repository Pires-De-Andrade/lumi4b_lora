"""Interpolação de pesos (scripts/interpolate_models.py): extremos, ponto médio, tipos e checagens de chaves."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interpolate_models import interpolate  # noqa: E402


def _modelo(d: Path, seed: int, extra: dict | None = None):
    g = torch.Generator().manual_seed(seed)
    t = {"w.a": torch.randn(4, 3, generator=g).to(torch.bfloat16), "w.b": torch.randn(5, generator=g).to(torch.bfloat16),
         "idx": torch.arange(3)}
    t.update(extra or {})
    d.mkdir(parents=True, exist_ok=True)
    save_file(t, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}")
    return t


def _rodar(a, b, alpha, out):
    return subprocess.run([sys.executable, str(ROOT / "scripts/interpolate_models.py"), "--a", str(a), "--b", str(b), "--alpha", str(alpha),
                           "--out", str(out), "--no-sha"], capture_output=True, text=True)


def test_extremos_e_ponto_medio(tmp_path):
    ta, tb = _modelo(tmp_path / "a", 1), _modelo(tmp_path / "b", 2)
    for alpha, ref in ((1.0, ta), (0.0, tb)):
        r = _rodar(tmp_path / "a", tmp_path / "b", alpha, tmp_path / f"o{alpha}")
        assert r.returncode == 0, r.stderr
        o = load_file(str(tmp_path / f"o{alpha}" / "model.safetensors"))
        assert all(torch.equal(o[k], ref[k]) for k in ref)
    r = _rodar(tmp_path / "a", tmp_path / "b", 0.5, tmp_path / "m")
    assert r.returncode == 0, r.stderr
    o = load_file(str(tmp_path / "m" / "model.safetensors"))
    esperado = ((ta["w.a"].float() + tb["w.a"].float()) / 2).to(torch.bfloat16)
    assert o["w.a"].dtype == torch.bfloat16 and torch.equal(o["w.a"], esperado)
    assert torch.equal(o["idx"], ta["idx"])
    assert (tmp_path / "m" / "config.json").exists()
    rep = json.loads((tmp_path / "m" / "interp_report.json").read_text())
    assert rep["alpha"] == 0.5 and rep["keys"] == 3 and rep["tensores_diferentes_entre_a_e_b"] == 2


def test_recusa_chaves_ou_formas_diferentes(tmp_path):
    _modelo(tmp_path / "a", 1)
    _modelo(tmp_path / "b", 2, {"extra": torch.zeros(2)})
    assert _rodar(tmp_path / "a", tmp_path / "b", 0.5, tmp_path / "o").returncode != 0
    with pytest.raises(ValueError):
        interpolate(torch.zeros(2, 3), torch.zeros(3, 2), 0.5)
    with pytest.raises(ValueError):
        interpolate(torch.arange(3), torch.arange(3) + 1, 0.5)

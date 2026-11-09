"""Junção de parquets de SFT (scripts/concat_parquets.py)."""
import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_spec = importlib.util.spec_from_file_location("concat_parquets", Path(__file__).resolve().parents[1] / "scripts" / "concat_parquets.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _tab(ids, mode="think", group="g", n=5):
    return pa.table({"id": ids, "mode": [mode] * len(ids), "group": [group] * len(ids), "n_tokens": [n] * len(ids),
                     "input_ids": [[1, 2, 3]] * len(ids)})


def test_concat_junta_linhas_e_preserva_ordem():
    out = C.concat([_tab(["a", "b"]), _tab(["c"], group="h")])
    assert out.column("id").to_pylist() == ["a", "b", "c"]
    assert out.column("group").to_pylist() == ["g", "g", "h"]


def test_concat_recusa_id_repetido():
    with pytest.raises(ValueError, match="ids repetidos"):
        C.concat([_tab(["a", "b"]), _tab(["b"])])


def test_concat_recusa_esquema_diferente():
    outro = pa.table({"id": ["z"], "mode": ["think"], "group": ["g"], "n_tokens": pa.array([5], pa.int32()), "input_ids": [[1]]})
    with pytest.raises(ValueError, match="esquemas diferentes"):
        C.concat([_tab(["a"]), outro])


def test_main_grava_parquet_e_manifest(tmp_path, monkeypatch):
    a, b, out = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "c.parquet"
    pq.write_table(_tab(["a1", "a2"]), a)
    pq.write_table(_tab(["b1"], group="h", n=7), b)
    monkeypatch.setattr(sys, "argv", ["x", "--inputs", str(a), str(b), "--out", str(out)])
    assert C.main() == 0
    assert pq.read_table(out).num_rows == 3
    import json
    m = json.loads((tmp_path / "c.manifest.json").read_text())
    assert m["rows"] == {"think/g": 2, "think/h": 1}
    assert m["total_tokens"] == 17
    assert set(m["inputs"]) == {str(a), str(b)}

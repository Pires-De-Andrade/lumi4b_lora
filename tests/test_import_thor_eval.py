"""Importação da devolução da Thor (scripts/import_thor_eval.py): junta sem sobrescrever e recusa conflitos."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_thor_eval import merge_eval_dir  # noqa: E402


def _pasta(d: Path, tarefas: dict, engine="motor"):
    d.mkdir(parents=True, exist_ok=True)
    summ = {"_engine": {"model": engine}}
    for t, acc in tarefas.items():
        (d / f"{t}_thinking_scores.json").write_text(json.dumps([{"id": "x", "strict_ok": True}]))
        (d / f"{t}_thinking.jsonl").write_text("{}\n")
        summ[f"{t}|thinking"] = {"n": 1, "strict_acc": acc}
    (d / "summary.json").write_text(json.dumps(summ))
    return d


def test_junta_a_um_destino_que_ja_tem_a_suite(tmp_path):
    dst = _pasta(tmp_path / "base_thor", {"gsm8k_test": 0.89})
    src = _pasta(tmp_path / "base_novas_thor", {"oab_novas": 0.2})
    r = merge_eval_dir(src, dst)
    assert r == {"arquivos": 2, "chaves": 1}
    s = json.loads((dst / "summary.json").read_text())
    assert set(k for k in s if "|" in k) == {"gsm8k_test|thinking", "oab_novas|thinking"}
    assert "_engine_base_novas_thor" in s and (dst / "oab_novas_thinking.jsonl").exists()
    assert merge_eval_dir(src, dst) == {"arquivos": 0, "chaves": 0}  # idempotente


def test_recusa_arquivo_ou_chave_em_conflito(tmp_path):
    dst = _pasta(tmp_path / "d", {"enem": 0.8})
    src = _pasta(tmp_path / "s", {"enem": 0.7})
    with pytest.raises(SystemExit):
        merge_eval_dir(src, dst)
    (src / "enem_thinking_scores.json").write_text((dst / "enem_thinking_scores.json").read_text())
    (src / "enem_thinking.jsonl").write_text((dst / "enem_thinking.jsonl").read_text())
    with pytest.raises(SystemExit):  # arquivos iguais, mas a chave do summary tem outro resultado
        merge_eval_dir(src, dst)

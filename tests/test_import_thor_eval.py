"""Importação da devolução da Thor (scripts/import_thor_eval.py): junta sem sobrescrever e recusa conflitos."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_thor_eval import chave_do_arquivo, destinos, merge_eval_dir  # noqa: E402


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


def test_chave_do_arquivo():
    assert chave_do_arquivo("oab_novas_thinking.jsonl") == "oab_novas|thinking"
    assert chave_do_arquivo("mmlu_pt_enade_nothinking_scores.json") == "mmlu_pt_enade|nothinking"
    assert chave_do_arquivo("summary.json") is None and chave_do_arquivo("notas.txt") is None


def test_entrega_parcial_ignora_o_que_esta_em_andamento_e_a_seguinte_completa(tmp_path):
    src = _pasta(tmp_path / "s_spark", {"enem": 0.8})
    (src / "oab_thinking.jsonl").write_text("{}\n")  # tarefa sem chave no summary: em andamento
    dst = tmp_path / "d"
    assert merge_eval_dir(src, dst) == {"arquivos": 2, "chaves": 1}
    assert not (dst / "oab_thinking.jsonl").exists()
    # a entrega seguinte traz a prova inteira (conteúdo diferente do parcial) e a importação aceita
    src2 = _pasta(tmp_path / "s2_spark", {"enem": 0.8, "oab": 0.5})
    (src2 / "oab_thinking.jsonl").write_text("{}\n{}\n")
    assert merge_eval_dir(src2, dst) == {"arquivos": 2, "chaves": 1}


def test_origem_spark_pega_toda_pasta_spark_da_devolucao(tmp_path):
    for n in ("base_spark", "thor_pt_antidoom_of_spark", "base_novas_thor"):
        _pasta(tmp_path / "runs" / "eval" / n, {"enem": 0.8})
    assert [d for d, _ in destinos("spark", tmp_path)] == ["base_spark", "thor_pt_antidoom_of_spark"]
    assert destinos("thor", tmp_path)[0][0] == "base_thor"

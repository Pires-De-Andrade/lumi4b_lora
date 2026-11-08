"""Conversor de questões traduzidas em prompts de RL (scripts/make_questions_rl_prompts.py)."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("make_questions_rl_prompts", Path(__file__).resolve().parents[1] / "scripts" / "make_questions_rl_prompts.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def rec(**kw):
    base = {"id": "q1", "source": "medmcqa", "category": "Medicine", "verdict": "ok", "ref": "b", "question_pt": "Qual é a resposta?",
            "options_pt": ["um", "dois", "três", "quatro"], "question_en": "What?", "options_en": ["one", "two", "three", "four"]}
    base.update(kw)
    return base


def conv(r, seen=None, pt=lambda t: False, en=lambda t: False, ntok=lambda t: 10, maxtok=100):
    bp = lambda item: item["question"] + "|" + ",".join(item["options"])  # noqa: E731
    return Q.convert(r, set() if seen is None else seen, bp, pt, en, ntok, maxtok)


def test_questao_valida_vira_linha_de_rl():
    row, why = conv(rec())
    assert why is None
    assert row["id"] == "mcq:q1" and row["kind"] == "mcq" and row["gold"] == "B" and row["n_options"] == 4
    assert row["prompt"] == "Qual é a resposta?|um,dois,três,quatro" and row["source"] == "medmcqa"


def test_recusada_na_traducao_sai():
    assert conv(rec(verdict="q:numeros_divergentes"))[1] == "recusada_na_traducao"


def test_esquema_invalido():
    for bad in (rec(ref="E"), rec(ref="AB"), rec(ref=None), rec(options_pt=["um"]), rec(options_pt=["um", "", "três"]), rec(question_pt=" "),
                rec(options_pt=["x"] * 11)):
        assert conv(bad)[1] == "esquema_invalido"


def test_gabarito_dentro_do_intervalo_de_alternativas():
    assert conv(rec(ref="C", options_pt=["a", "b", "c"]))[0]["gold"] == "C"
    assert conv(rec(ref="D", options_pt=["a", "b", "c"]))[1] == "esquema_invalido"


def test_id_repetido_so_na_segunda():
    seen = set()
    assert conv(rec(), seen)[1] is None
    assert conv(rec(), seen)[1] == "id_repetido"


def test_contaminacao_em_portugues_e_em_ingles():
    assert conv(rec(), pt=lambda t: True)[1] == "8grama_provas_pt"
    assert conv(rec(), en=lambda t: True)[1] == "8grama_mmlu_prox_en"


def test_contaminacao_olha_enunciado_e_alternativas():
    vistos = []
    conv(rec(), pt=lambda t: vistos.append(t) or False)
    assert "Qual é a resposta?" in vistos[0] and "quatro" in vistos[0]


def test_prompt_longo_sai_e_nao_marca_o_id_como_visto():
    seen = set()
    assert conv(rec(), seen, ntok=lambda t: 101)[1] == "prompt_longo"
    assert "q1" not in seen


def test_leitura_tolera_linha_quebrada(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": 1}) + "\n" + '{"id": 2, "verd' + "\n")
    rows, bad = Q.read_jsonl(str(p))
    assert rows == [{"id": 1}] and bad == 1

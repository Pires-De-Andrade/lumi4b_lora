"""Provas novas (src/lumi_ft/eval/novas.py): conversão de cada fonte, descontaminação, arquivos e registro dos carregadores."""
import pytest

from lumi_ft.data.decontaminate import build_index
from lumi_ft.eval import mcq as M
from lumi_ft.eval import novas as N


def _oab(i, exam="2019-01", key=1, nullified=False, texts=("a um", "b dois", "c tres", "d quatro")):
    return {"id": f"{exam}_{i}", "exam_id": exam, "nullified": nullified, "question": f"Questão {i} sobre direito",
            "choices": {"label": ["A", "B", "C", "D"], "text": list(texts)}, "answerKey": key}


def test_oab_novas_exclui_edicoes_da_suite_anuladas_e_gabarito_invalido():
    rows = [_oab(1), _oab(2, exam="2017-24"), _oab(3, nullified=True), _oab(4, key="C"), _oab(5, key="x"), _oab(6, key=9), _oab(7, texts=("a", "", "c", "d"))]
    items, drops = N.parse_oab_novas(rows, {"2017-24"})
    assert [i["id"] for i in items] == ["oab_novas:2019-01_1", "oab_novas:2019-01_4"]
    assert [i["gold"] for i in items] == ["B", "C"]  # índice 1 e rótulo "C"
    assert drops == {"edicao_na_suite": 1, "anulada": 1, "gabarito_invalido": 1, "formato": 2}


def _enem(i, q, alts, label, descs=()):
    return {"id": f"questao_{i:02d}", "question": q, "alternatives": list(alts), "label": label, "description": list(descs), "figures": []}


def test_enem_2024_troca_placeholders_na_ordem_e_tira_anulada_e_figura_sem_descricao():
    alts = ["x", "y", "z", "w", "v"]
    rows = [
        _enem(1, "Texto simples?", alts, "C"),
        _enem(2, "[[placeholder]]\nO que a figura mostra?", alts, "A", ["Descrição da imagem: um gráfico."]),
        _enem(3, "Qual figura?", ["[[placeholder]]"] * 5, "B", [f"Descrição {k}" for k in range(5)]),
        _enem(4, "Anulada", alts, "Anulado"),
        _enem(5, "[[placeholder]] sem descrição", alts, "D"),
    ]
    items, drops = N.parse_enem_2024(rows)
    assert [i["id"] for i in items] == ["enem_2024:questao_01", "enem_2024:questao_02", "enem_2024:questao_03"]
    assert items[1]["question"].startswith("Descrição da imagem: um gráfico.\nO que")
    assert items[2]["options"] == [f"Descrição {k}" for k in range(5)] and items[2]["gold"] == "B"
    assert drops == {"anulada": 1, "figura_sem_descricao": 1}


def _mm(i, exam, n=4, ans=0, choices=None):
    return {"exam": exam, "annotation_id": f"{i:032x}", "question": f"Pergunta {i}", "answer": ans,
            "choices": choices if choices is not None else [f"op{k}" for k in range(n)]}


def test_mmlu_pt_filtra_por_exame_e_formato():
    rows = [_mm(1, "OBI"), _mm(2, "ENADE"), _mm(3, "IME", n=5, ans=4), _mm(4, "AFA", ans=7), _mm(5, "POSCOMP", choices=["a", "", "c", "d"]), _mm(6, "BNDES", n=3)]
    items, drops = N.parse_mmlu_pt(rows, N.MMLU_PT_SLICES["mmlu_pt_exatas"], "mmlu_pt_exatas")
    assert [i["gold"] for i in items] == ["A", "E"] and items[0]["id"] == "mmlu_pt_exatas:" + f"{1:032x}"[:16]
    assert drops == {"formato": 2}  # resposta fora e alternativa vazia; BNDES e ENADE nem entram na conta


def test_descontaminacao_por_fracao_e_conjunto_estrito():
    suite = build_index([{"q": "Qual das alternativas abaixo descreve corretamente o princípio da legalidade no direito administrativo brasileiro"}])
    treino = build_index([{"q": "O presidente da república sanciona ou veta o projeto de lei aprovado pelo congresso nacional em quinze dias úteis"}])
    comum = "As asserções I e II são proposições verdadeiras e a II é uma justificativa correta da I"
    suite.add(comum)
    novos = [
        {"id": "a", "question": "Qual das alternativas abaixo descreve corretamente o princípio da legalidade no direito administrativo brasileiro?", "options": ["x", "y"], "gold": "A"},
        {"id": "b", "question": "O presidente da república sanciona ou veta o projeto de lei aprovado pelo congresso nacional em quinze dias úteis, correto?", "options": ["x", "y"], "gold": "A"},
        {"id": "c", "question": "Uma pergunta inteiramente diferente sobre fotossíntese em plantas aquáticas de água doce", "options": ["x", "y"], "gold": "B"},
        # só a alternativa-padrão coincide: poucos 8-gramas diante dos do enunciado, então fica, mas fora do conjunto estrito
        {"id": "d", "question": "Considere o texto " + " ".join(f"palavra{k}" for k in range(150)) + " e avalie as relações propostas",
         "options": [comum, "As asserções I e II são proposições falsas"], "gold": "A"},
    ]
    kept, removed, strict = N.decontaminate(novos, {"suite": suite, "treino": treino})
    assert [i["id"] for i in kept] == ["c", "d"] and removed == {"suite": 1, "treino": 1}
    assert strict == {"c"}


def test_drop_same_text_e_arquivos(tmp_path, monkeypatch):
    its = [{"id": "oab_novas:1", "question": "Texto  IGUAL, sim", "options": ["a", "b"], "gold": "A"},
           {"id": "oab_novas:2", "question": "Outro texto", "options": ["a", "b"], "gold": "B"}]
    kept, n = N.drop_same_text(its, {N.norm_text("texto igual sim")})
    assert [i["id"] for i in kept] == ["oab_novas:2"] and n == 1
    N.write_task(tmp_path, "oab_novas", kept)
    assert N.read_task("oab_novas", tmp_path) == kept
    assert (tmp_path / "oab_novas_ids.json").read_text() == '["oab_novas:2"]'
    with pytest.raises(FileNotFoundError):
        N.read_task("enem_2024", tmp_path)
    monkeypatch.setattr(N, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(N, "read_task", lambda name, root=str(tmp_path): N.read_task.__wrapped__(name, root) if hasattr(N.read_task, "__wrapped__") else kept)
    assert M.LOADERS["oab_novas"]() == kept


def test_loaders_registrados():
    for name in N.TASKS:
        assert name in M.LOADERS

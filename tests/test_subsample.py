"""Amostra fixa de uma prova (src/lumi_ft/eval/mcq.py: subsample): tamanho, determinismo, ordem e ids preservados."""
from lumi_ft.eval import mcq as M


def _itens(n):
    return [{"id": f"x:{i}", "question": "q", "options": ["a", "b"], "gold": "A"} for i in range(n)]


def test_tamanho_determinismo_e_ordem():
    it = _itens(2210)
    a, b = M.subsample(it, 800), M.subsample(it, 800)
    assert len(a) == 800 and [x["id"] for x in a] == [x["id"] for x in b]
    idx = [int(x["id"].split(":")[1]) for x in a]
    assert idx == sorted(idx) and len(set(idx)) == 800
    assert [x["id"] for x in M.subsample(it, 800, seed=1)] != [x["id"] for x in a]


def test_prova_menor_que_n_fica_inteira():
    it = _itens(50)
    assert M.subsample(it, 800) is it

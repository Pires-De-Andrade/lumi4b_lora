"""Filtro por tamanho do raciocínio (scripts/subset_think_by_length.py): só descarta, nunca altera, e mantém o sem thinking."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("subset_think_by_length", Path(__file__).resolve().parents[1] / "scripts" / "subset_think_by_length.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

O, C = S.THINK_OPEN, S.THINK_CLOSE


def row(mode, ids):
    return {"id": f"{mode}{len(ids)}", "mode": mode, "input_ids": ids}


def test_think_tokens_conta_entre_marcadores():
    assert S.think_tokens([5, O, 7, 8, 9, C, 3]) == 3
    assert S.think_tokens([5, O, C, 3]) == 0


def test_think_tokens_sem_marcadores_ou_sem_fechar():
    assert S.think_tokens([5, 6, 7]) is None
    assert S.think_tokens([5, O, 6, 7]) is None
    assert S.think_tokens([5, C, 6]) is None


def test_filtra_so_o_think_longo_e_mantem_o_nothink():
    curto = row("think", [1, O] + [9] * 10 + [C, 2])
    limite = row("think", [1, O] + [9] * 20 + [C, 2])
    longo = row("think", [1, O] + [9] * 21 + [C, 2])
    sem = row("think", [1, 2, 3])
    nt = row("nothink", [1, O, C, 2] + [9] * 5000)  # nothink nunca é filtrado, mesmo longo
    kept, counts = S.subset_by_think_len([curto, longo, limite, sem, nt], 20)
    assert kept == [curto, limite, nt]
    assert counts == {"think_mantidos": 2, "think_removidos_longos": 1, "think_sem_marcadores": 1, "nothink_mantidos": 1}


def test_nao_altera_as_linhas_mantidas():
    r = row("think", [1, O, 9, C, 2])
    kept, _ = S.subset_by_think_len([r], 100)
    assert kept[0] is r

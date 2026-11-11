"""Cálculo dos números dos gráficos das provas novas do card (scripts/make_card_figures_novas.py), com itens sintéticos: acerto, fechamento e os
quadrantes pareados (a soma deles tem de dar a diferença de acerto)."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("card_figures_novas", SCRIPTS / "make_card_figures_novas.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def rows(spec: list[tuple[bool, bool]]) -> dict[str, dict]:
    """[(fechou, acertou)] -> {id: linha}. Acerto estrito sem fechar o `</think>` não existe (vale 0)."""
    return {f"i{k}": {"closed": c, "strict_ok": bool(a and c)} for k, (c, a) in enumerate(spec)}


REF = rows([(True, True), (True, True), (True, False), (False, False), (False, False), (True, True), (False, False), (True, False)])
CUR = rows([(True, True), (True, False), (True, False), (True, True), (True, True), (False, False), (True, False), (True, True)])


def test_accuracy_and_closure():
    assert N.accuracy(REF) == 100 * 3 / 8
    assert N.closure_share(REF) == 100 * 5 / 8
    assert N.accuracy(CUR) == 100 * 4 / 8
    assert N.closure_share(CUR) == 100 * 7 / 8


def test_quadrants_add_up_to_the_paired_difference():
    both, only_ref, only_cur = N.quadrants(REF, CUR)
    assert abs((both + only_ref + only_cur) - (N.accuracy(CUR) - N.accuracy(REF))) < 1e-9
    assert only_ref <= 0 <= only_cur  # items that only the reference closes cannot help the model; the reverse cannot hurt


def test_both_close_can_be_positive_or_negative():
    # na suíte o quadrante "ambos fecham" era sempre negativo; nas provas novas pode ser positivo, e o gráfico trata os dois sinais
    better = N.quadrants(rows([(True, False)] * 4), rows([(True, True)] * 4))[0]
    worse = N.quadrants(rows([(True, True)] * 4), rows([(True, False)] * 4))[0]
    assert better == 100.0 and worse == -100.0


def test_missing_quadrant_counts_as_zero():
    same = rows([(True, True), (True, False)])
    assert N.quadrants(same, same) == (0.0, 0.0, 0.0)

"""Consistência dos números do card do T6-Lumi (scripts/make_card_figures.py): médias, diferenças pareadas, quadrantes e ordem dos modelos.

Os números vêm dos registros (discovery.md §20.64, §20.66 e §20.68). Estes testes não medem nada: pegam erro de digitação, comparando cada número com
os outros que o mesmo registro permite recalcular.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("card_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_card_figures.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

T6, LT, LN = "T6-Lumi (thinking)", "Lumi-4B-v0.2 (thinking)", "Lumi-4B-v0.2 (no thinking)"


def test_reasoning_score_matches_the_record():
    for m, rec in F.REASONING_SCORE_RECORD.items():
        assert F.fmt(F.mean_score(m)) == F.fmt(rec), m


def test_paired_difference_matches_the_accuracy_columns():
    # estimate in the record vs the difference of the two accuracies shown in the figure (each rounded to 0.01-0.1)
    for exam in F.EXAMS:
        assert abs((F.ACC[T6][exam] - F.ACC[LT][exam]) - F.DIFF_VS_THINKING[exam][0]) <= 0.15, exam
        assert abs((F.ACC[T6][exam] - F.ACC[LN][exam]) - F.DIFF_VS_NO_THINKING[exam][0]) <= 0.15, exam
    for model, diff in ((LT, F.DIFF_VS_THINKING), (LN, F.DIFF_VS_NO_THINKING)):
        assert abs((F.mean_score(T6) - F.mean_score(model)) - diff["Reasoning score"][0]) <= 0.15


def test_confidence_intervals_contain_the_estimate_and_are_ordered():
    for diff in (F.DIFF_VS_THINKING, F.DIFF_VS_NO_THINKING):
        assert set(diff) == set(F.EXAMS) | {"Reasoning score"}
        for est, lo, hi in diff.values():
            assert lo <= est <= hi


def test_quadrants_add_up_to_the_paired_difference():
    for exam, (both, only_lumi, only_t6) in F.QUADRANTS.items():
        assert both <= 0 and only_lumi <= 0 and only_t6 >= 0, exam
        assert abs((both + only_lumi + only_t6) - F.DIFF_VS_THINKING[exam][0]) <= 0.15, exam


def test_closure_is_higher_for_t6_and_the_gain_is_where_lumi_truncates():
    for exam in F.EXAMS:
        assert F.CLOSURE[T6][exam] > F.CLOSURE[LT][exam], exam
    # the exam where Lumi closes least is the exam with the largest paired gain
    worst = min(F.EXAMS, key=lambda e: F.CLOSURE[LT][e])
    assert worst == max(F.EXAMS, key=lambda e: F.DIFF_VS_THINKING[e][0]) == "OAB Exams"


def test_stage_values_are_percentages_and_end_at_t6():
    for metric, vals in F.VALIDATION.items():
        assert len(vals) == len(F.STAGES) == 3 and all(0 <= v <= 100 for v in vals), metric
    for loops in F.LOOPS.values():
        assert loops[1] < loops[0]


def test_gsm8k_t6_is_1170_of_1295():
    assert round(F.ACC[T6]["GSM8K-PT"], 2) == 90.35  # the record's rounding of 1,170/1,295
    assert F.fmt(F.ACC[T6]["GSM8K-PT"]) == "90.3"

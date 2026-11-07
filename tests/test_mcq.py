"""Testes do módulo de múltipla escolha e do registro de tarefas (CPU, sem rede)."""

from lumi_ft.eval import mcq as M
from lumi_ft.eval import tasks as T

ITEM = {"id": "x:1", "question": "Qual é a capital da França?", "options": ["Berlim", "Paris", "Roma", "Madri"], "gold": "B"}


def test_prompt_lists_options_with_letters_and_asks_for_final_line():
    p = M.build_prompt(ITEM)
    assert "Resposta: <letra>" in p and "Questão: Qual é a capital da França?" in p
    assert "A) Berlim\nB) Paris\nC) Roma\nD) Madri" in p and p.endswith("D) Madri")


def test_prompt_with_ten_options_uses_up_to_j():
    item = {"id": "y", "question": "q", "options": [str(i) for i in range(10)], "gold": "J"}
    assert "J) 9" in M.build_prompt(item) and "K)" not in M.build_prompt(item)


def _gen(text, finished=True):
    return {"id": "x:1", "text": text, "n_gen": 30, "finished": finished}


def test_score_row_strict_correct_with_thinking():
    r = M.score_row(_gen("A França... Paris.</think>\nParis é a capital.\nResposta: B"), ITEM, thinking=True)
    assert r["strict_ok"] and r["flexible_ok"] and r["status"] == "ok" and r["closed"]


def test_score_row_wrong_letter():
    r = M.score_row(_gen("t</think>Resposta: C"), ITEM, thinking=True)
    assert not r["strict_ok"] and r["pred"] == "C"


def test_score_row_letter_outside_options_is_an_error():
    r = M.score_row(_gen("t</think>Resposta: E"), ITEM, thinking=True)
    assert r["status"] == "invalid_option" and not r["strict_ok"]


def test_score_row_answer_only_in_trace_does_not_count():
    r = M.score_row(_gen("Resposta: B\n</think>\nNão sei dizer."), ITEM, thinking=True)
    assert not r["strict_ok"] and r["status"] == "no_option"


def test_score_row_flexible_differs_from_strict_when_format_is_loose():
    r = M.score_row(_gen("t</think>A resposta correta é a letra B."), ITEM, thinking=True)
    assert not r["strict_ok"] and r["flexible_ok"]


def test_score_row_no_thinking():
    r = M.score_row(_gen("Paris.\nResposta: B"), ITEM, thinking=False)
    assert r["strict_ok"] and r["trace_language"] is None


def test_summarize_rates():
    rows = [M.score_row(_gen("t</think>Resposta: B"), ITEM, thinking=True),
            M.score_row(_gen("t</think>Resposta: A"), ITEM, thinking=True),
            M.score_row(_gen("sem fim", finished=False), ITEM, thinking=True)]
    s = M.summarize(rows, thinking=True)
    assert s["n"] == 3 and abs(s["strict_acc"] - 1 / 3) < 1e-3 and abs(s["truncated_rate"] - 1 / 3) < 1e-3
    assert s["status_rates"]["no_close"] == 1 / 3 and s["closed_and_answer_rate"] == round(2 / 3, 4)


def test_registry_groups_and_unknown_task():
    import pytest

    assert T.SUITE == ["gsm8k_test", "mmlu_prox_test", "enem", "bluex", "oab"]
    assert set(M.LOADERS) >= {"mmlu_prox_test", "enem", "bluex", "oab", "global_mmlu_test", "mmmlu_test", "include_test"}
    with pytest.raises(KeyError):
        T.get_task("nao_existe")


def test_prompt_en_keeps_marker_and_options():
    from lumi_ft.eval import mcq as M

    p = M.build_prompt_en({"question": "What is 2+2?", "options": ["3", "4"]})
    assert p.startswith("Answer the multiple-choice") and "`Resposta: <letter>`" in p and "Question: What is 2+2?" in p and "B) 4" in p
    assert "Questão:" not in p

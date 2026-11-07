"""Testes de extração de resposta com casos sintéticos (A-05 e A-06). Só CPU."""

import pytest

from lumi_ft.eval import parse as P

ABCD, ABCDE = "ABCD", "ABCDE"


# ------------------------------------------------------------------ separar traço e resposta
def test_thinking_splits_trace_and_answer():
    p = P.split_response("passo 1\npasso 2</think>\n\nResposta: B<|im_end|>", thinking=True)
    assert (p.thought, p.answer, p.closed, p.n_close) == ("passo 1\npasso 2", "Resposta: B", True, 1)


def test_thinking_without_close_has_no_answer():
    p = P.split_response("raciocínio que não terminou...", thinking=True)
    assert p.answer is None and not p.closed and p.answer_or_empty == ""


def test_thinking_ignores_text_after_im_end():
    p = P.split_response("t</think>Resposta: A<|im_end|>Resposta: C", thinking=True)
    assert p.answer == "Resposta: A"


def test_thinking_uses_first_close_and_flags_repeats():
    p = P.split_response("t</think>Resposta: A</think>Resposta: C", thinking=True)
    assert p.answer == "Resposta: A</think>Resposta: C" and p.n_close == 2


def test_no_thinking_answer_is_the_whole_output():
    p = P.split_response("Resposta: C<|im_end|>", thinking=False)
    assert p.answer == "Resposta: C" and not p.closed


def test_no_thinking_with_spontaneous_trace_uses_text_after_close():
    p = P.split_response("<think>hmm</think>Resposta: D", thinking=False)
    assert p.thought == "hmm" and p.answer == "Resposta: D"


def test_no_thinking_with_open_trace_never_closed_is_an_error():
    r = P.extract_choice("<think>hmm... Resposta: D", thinking=False, letters=ABCD)
    assert r.status == P.NO_CLOSE and r.pred is None


def test_im_end_stripped_whether_or_not_decoder_kept_it():
    a = P.extract_choice("t</think>Resposta: B<|im_end|>", thinking=True, letters=ABCD)
    b = P.extract_choice("t</think>Resposta: B", thinking=True, letters=ABCD)
    assert a.pred == b.pred == "B"


# ------------------------------------------------------------------ múltipla escolha: estrita
@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Resposta: B", "B"),
        ("resposta: b", "B"),
        ("**Resposta: B**", "B"),
        ("**Resposta:** B", "B"),
        ("**Resposta**: B", "B"),
        ("Resposta: (C)", "C"),
        ("Resposta: D.", "D"),
        ("Explicação longa.\nMais uma linha.\nResposta: A", "A"),
        ("Resposta: A\nrevisando...\nResposta: C", "C"),  # vale a última
    ],
)
def test_strict_choice_accepts(answer, expected):
    assert P.strict_choice(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "A resposta é B",  # frase, não a linha pedida
        "Resposta B",  # sem dois pontos
        "Resposta: B porque a soma dá 10",  # texto depois da letra
        "Resposta: Não sei",
        "B",
        "",
    ],
)
def test_strict_choice_rejects(answer):
    assert P.strict_choice(answer) is None


# ----------------------------------------------------------------- múltipla escolha: flexível
@pytest.mark.parametrize(
    "answer, expected",
    [
        ("A resposta é B", "B"),
        ("A resposta correta é a alternativa C.", "C"),
        ("A resposta correta é a letra E, pois...", "E"),
        ("Gabarito: C", "C"),
        ("Resposta: B porque a soma dá 10", "B"),
        ("Resposta: A resposta correta é B", "B"),  # `A` do artigo não é a alternativa
        ("B", "B"),
        ("(B)", "B"),
        ("B.", "B"),
    ],
)
def test_flexible_choice_accepts(answer, expected):
    assert P.flexible_choice(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "As alternativas A e B estão erradas.",  # `A` e `E` são palavras em português
        "Alternativa B está correta. Alternativa C está incorreta.",  # sem afirmação ("é"/":")
        "a resposta é b",  # letra minúscula: ambígua com artigo, não vale
        "Não consigo responder.",
        "",
    ],
)
def test_flexible_choice_rejects(answer):
    assert P.flexible_choice(answer) is None


# -------------------------------------------------------------------------- status da questão
def test_choice_ok():
    r = P.extract_choice("pensando...</think>\nResposta: C", thinking=True, letters=ABCD)
    assert (r.status, r.pred, r.flexible) == (P.OK, "C", "C")


def test_choice_answer_inside_trace_does_not_count():
    """O risco do BELEBELE: a resposta aparece no traço, mas não na resposta final."""
    r = P.extract_choice("Resposta: A\n</think>\nNão tenho certeza.", thinking=True, letters=ABCD)
    assert r.status == P.NO_OPTION and r.pred is None and r.flexible is None


def test_choice_without_close_is_an_error_even_if_trace_has_the_answer():
    r = P.extract_choice("analisando... Resposta: B", thinking=True, letters=ABCD)
    assert r.status == P.NO_CLOSE and r.pred is None


def test_choice_empty_answer():
    r = P.extract_choice("t</think>   <|im_end|>", thinking=True, letters=ABCD)
    assert r.status == P.EMPTY_ANSWER


def test_choice_no_option():
    r = P.extract_choice("t</think>Depende do contexto.", thinking=True, letters=ABCD)
    assert r.status == P.NO_OPTION


def test_choice_letter_outside_the_options_is_invalid():
    r = P.extract_choice("t</think>Resposta: E", thinking=True, letters=ABCD)
    assert (r.status, r.pred) == (P.INVALID_OPTION, None)
    assert P.extract_choice("t</think>Resposta: E", thinking=True, letters=ABCDE).pred == "E"


def test_choice_flexible_can_succeed_where_strict_fails():
    r = P.extract_choice("t</think>A resposta correta é a alternativa D.", thinking=True, letters=ABCD)
    assert (r.status, r.pred, r.flexible) == (P.NO_OPTION, None, "D")


def test_choice_without_thinking():
    r = P.extract_choice("Resposta: B<|im_end|>", thinking=False, letters=ABCD)
    assert (r.status, r.pred) == (P.OK, "B")


# -------------------------------------------------------------------------------- GSM8K-PT
def _num(text, thinking=True):
    return P.extract_number(text, thinking=thinking)


def test_gsm8k_target_from_dataset_answer():
    assert P.gsm8k_target("Leva 2/2=<<2/2=1>>1 fio.\n#### 3") == "3"
    assert P.gsm8k_target("... 130.000\n#### 70000") == "70000"


def test_gsm8k_strict_basic():
    r = _num("conta...</think>\nPortanto, 7 maçãs.\n#### 7")
    assert (r.status, r.strict, r.flexible) == (P.OK, "7", "7")
    assert P.gsm8k_correct(r.strict, "7")


@pytest.mark.parametrize(
    "answer, target",
    [
        ("#### 18", "18"),
        ("#### 18.", "18"),  # ponto final da frase
        ("#### 5,000", "5000"),  # `,` é removida pelo harness
        ("#### -3", "-3"),
        ("#### 1.5", "1.5"),
    ],
)
def test_gsm8k_correct_forms(answer, target):
    assert P.gsm8k_correct(_num("t</think>" + answer).strict, target)


def test_gsm8k_strict_misses_dollar_but_flexible_gets_it():
    r = _num("t</think>#### $18")
    assert r.status == P.NO_NUMBER and r.strict is None
    assert P.gsm8k_correct(r.flexible, "18")


def test_gsm8k_without_marker_only_flexible():
    r = _num("t</think>A resposta é 42.")
    assert r.status == P.NO_NUMBER and r.strict is None
    assert r.flexible == "42." and P.gsm8k_correct(r.flexible, "42")


def test_gsm8k_strict_takes_first_marker_flexible_takes_last_number():
    r = _num("t</think>#### 5\nrevisando, na verdade 7")
    assert r.strict == "5" and r.flexible == "7"


def test_gsm8k_marker_inside_trace_is_ignored():
    r = _num("chuto #### 99</think>Conta final.\n#### 7")
    assert r.strict == "7"


def test_gsm8k_no_close_is_an_error_even_with_marker_in_trace():
    r = _num("conta... #### 7")
    assert r.status == P.NO_CLOSE and r.strict is None and r.flexible is None


def test_gsm8k_empty_answer():
    assert _num("t</think>").status == P.EMPTY_ANSWER


def test_gsm8k_without_thinking():
    r = _num("Somando, dá 12.\n#### 12<|im_end|>", thinking=False)
    assert (r.status, r.strict) == (P.OK, "12")


def test_gsm8k_portuguese_thousands_separator_is_wrong_for_the_harness():
    """`70.000` é 70000 em português, mas o harness só remove `,`. A nota estrita o conta como erro.

    A diagnóstica `gsm8k_correct_ptbr` mostra quantos erros vêm só do formato (A-06).
    """
    r = _num("t</think>#### 70.000")
    assert r.strict == "70.000"
    assert not P.gsm8k_correct(r.strict, "70000")
    assert P.gsm8k_correct_ptbr(r.strict, "70000")


def test_gsm8k_ptbr_diagnostic_is_not_too_lenient():
    assert not P.gsm8k_correct_ptbr("70.001", "70000")
    assert not P.gsm8k_correct_ptbr(None, "7")
    assert P.gsm8k_correct_ptbr("3,5", "3.5")
    assert P.gsm8k_correct_ptbr("1.234,56", "1234.56")


@pytest.mark.parametrize("reference_answer, target", [("... 2125 blocos.\n#### 2.125", "2.125"), ("...\n#### 114.200", "114.200")])
def test_gsm8k_reference_with_thousands_dot_penalizes_the_plain_integer(reference_answer, target):
    """Dataset real: 9 de 1.295 alvos usam `.` de milhar (índices 143, 196, 495, 598, 803, 813, 977, 988, 1182).

    Quem responde o inteiro correto perde na nota estrita; o diagnóstico PT-BR recupera.
    """
    assert P.gsm8k_target(reference_answer) == target
    plain = target.replace(".", "")
    assert not P.gsm8k_correct(plain, P.gsm8k_target(reference_answer))
    assert P.gsm8k_correct_ptbr(plain, P.gsm8k_target(reference_answer))
    assert P.gsm8k_correct(target, P.gsm8k_target(reference_answer))  # imitar o formato do dataset acerta


@pytest.mark.parametrize("reference_answer", ["...6 mph\n#### 6mph", "...\n#### R$ 276.000", "...\n\n#### 25%"])
def test_gsm8k_malformed_reference_is_unreachable(reference_answer):
    """Dataset real: 3 de 1.295 alvos (18, 225, 1072) trazem unidade ou símbolo; nenhuma extração bate."""
    target = P.gsm8k_target(reference_answer)
    for plain in ("6", "276000", "276.000", "25"):
        assert not P.gsm8k_correct(plain, target)
        assert not P.gsm8k_correct_ptbr(plain, target)


def test_gsm8k_flexible_regex_keeps_the_harness_quirk():
    """O `flexible-extract` aceita `...` como número (`[$0-9.,]{2,}`). Replicamos para comparar notas."""
    r = _num("t</think>#### 18\n...")
    assert r.strict == "18" and r.flexible == "..."


def test_gsm8k_trace_last_number_for_shape_metric():
    r = _num("12+5=17 maçãs no total</think>#### 17")
    assert P.gsm8k_correct(r.trace_last_number(), "17")
    r2 = _num("então fica 17, mas Maria deu 2</think>#### 15")
    assert not P.gsm8k_correct(r2.trace_last_number(), "15")


# -------------------------------------------------------------------------------- taxas
def test_rates():
    out = P.rates([P.OK, P.OK, P.NO_CLOSE, P.NO_OPTION])
    assert out["n"] == 4 and out[P.OK] == 0.5 and out[P.NO_CLOSE] == 0.25 and out[P.EMPTY_ANSWER] == 0


def test_rates_rejects_unknown_status():
    with pytest.raises(ValueError):
        P.rates(["ok", "talvez"])

"""Separação das questões de múltipla escolha do Nemotron em português (CPU)."""

from lumi_ft.data import nemotron_pt as N

INSTR = "Responda à seguinte questão de múltipla escolha. A última linha da sua resposta deve estar no seguinte formato: 'Resposta: \\boxed{A/B/C/D}'."


def _row(prompt_body, letter="B", thought="We need to answer. " * 5, final="É a B.\n\nResposta: \\boxed{B}", uuid="u1"):
    return {"uuid": uuid, "metadata": {"source category": "RQA synthetics"},
            "messages": [{"role": "user", "content": f"{INSTR}\n\n{prompt_body}"}, {"role": "assistant", "content": final, "reasoning_content": thought}]}


BODY = "Qual é a cor do céu?\n\nA. verde\nB. azul\nC. vermelho\nD. preto"


def test_parse_prompt_dot_and_paren_and_multiline():
    assert N.parse_prompt(f"{INSTR}\n\n{BODY}") == ("Qual é a cor do céu?", ["verde", "azul", "vermelho", "preto"])
    stem, opts = N.parse_prompt(f"{INSTR}\n\nPergunta?\n\nA) um\nB) dois\ncontinua\nC) três\nD) quatro")
    assert stem == "Pergunta?" and opts == ["um", "dois\ncontinua", "três", "quatro"]
    stem, opts = N.parse_prompt(INSTR + "\n\nQ?\n\n" + "\n".join(f"{c}. op{c}" for c in "ABCDEFGHIJ"))
    assert len(opts) == 10 and opts[9] == "opJ"


def test_parse_prompt_rejects():
    assert N.parse_prompt("Sem instrução\n\nA. a\nB. b\nC. c\nD. d") is None  # sem "múltipla escolha" no cabeçalho
    assert N.parse_prompt(f"{INSTR}\n\nQ?\n\nA. a\nB. b") is None  # menos de 4 alternativas
    assert N.parse_prompt(f"{INSTR}\n\nQ?\n\nsem alternativas") is None
    assert N.parse_prompt(f"{INSTR}\n\nA. a\nB. b\nC. c\nD. d") is None  # sem enunciado


def test_select_filters_and_final_cleanup():
    rows = [_row(BODY, uuid="ok"), _row(BODY.replace("céu", "mar"), final="Resposta: \\boxed{B}", uuid="so_letra"),
            _row(BODY.replace("céu", "sol"), final="É a E.\n\nResposta: \\boxed{E}", uuid="letra_fora"),
            _row(BODY.replace("céu", "lua"), thought="word " * 2000, uuid="longo"), _row(BODY, uuid="repetida"),
            _row(BODY.replace("céu", "rio"), final="Sem caixa", uuid="sem_boxed")]
    out, st = N.select(rows, max_words=1000)
    assert [r["id"] for r in out] == ["ok", "so_letra"]
    ok = out[0]
    assert ok["final"] == "É a B." and ok["letter"] == "B" and ok["ready_pt"] and ok["options"][1] == "azul" and ok["category"] == "RQA synthetics"
    assert out[1]["final"] == ""  # a resposta era só a linha final; o treino monta `Resposta: B` depois
    assert st["sem_boxed_valido"] == 2 and st["traco_vazio_ou_longo"] == 1 and st["enunciado_repetido"] == 1 and st["mantidas"] == 2

"""Seleção das questões de ciência do Mixture-of-Thoughts (CPU)."""

from lumi_ft.data import mot_science as M

Q = "Qual é a cor do céu?A: verde\nB: azul\nC: vermelho\nD: preto"


def _row(letter: str, think: str = "Hmm.", tokens: int = 500, q: str = Q):
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\nÉ a B.\n\n\\boxed{{{letter}}}"}], "num_tokens": tokens}


def test_split_options_glued_and_lines():
    stem, opts = M.split_options(Q)
    assert stem == "Qual é a cor do céu?" and opts == ["verde", "azul", "vermelho", "preto"]
    stem, opts = M.split_options("Pergunta?\nA: x\nB: y\nC: z")
    assert stem == "Pergunta?" and opts == ["x", "y", "z"]


def test_split_options_rejects_without_options():
    assert M.split_options("Sem alternativas aqui") is None
    assert M.split_options("Enunciado A: só uma\nB: duas") is None  # menos de 3 alternativas
    assert M.split_options("Tabela A: x\nPergunta?") is None


def test_parse_trace():
    t, f, ltr = M.parse_trace(_row("B")["messages"][1]["content"])
    assert (t, ltr) == ("Hmm.", "B") and f == "É a B." and "boxed" not in f
    assert M.parse_trace("<think>\nx</think>\nsem caixa") is None
    assert M.parse_trace("sem think \\boxed{A}") is None
    assert M.parse_trace("<think>a</think>b</think>\\boxed{A}") is None


def test_select_majority_and_filters():
    rows = [_row("B"), _row("B"), _row("B"), _row("C")]
    out, st = M.select(rows)
    assert len(out) == 2 and {r["letter"] for r in out} == {"B"} and st["perguntas_usadas"] == 1
    assert out[0]["options"][1] == "azul" and out[0]["question"] == "Qual é a cor do céu?"
    out, st = M.select([_row("B"), _row("C"), _row("A"), _row("D")])
    assert out == [] and st["sem_maioria"] == 1
    out, st = M.select([_row("B"), _row("B")])  # poucos traços
    assert out == [] and st["poucos_tracos_validos"] == 1
    out, st = M.select([_row("B", tokens=5000)] * 3)
    assert out == [] and st["traco_longo"] == 3 and st["sem_traco_aproveitavel"] == 1

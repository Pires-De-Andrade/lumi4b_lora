"""Testes da fase 1: filtros, renderização com máscara e descontaminação (CPU)."""

import pytest

from lumi_ft.data import gigaverbo as GV
from lumi_ft.data.decontaminate import NgramIndex, build_index, flatten_strings, ngram_hashes


@pytest.fixture(scope="module")
def tok():
    from lumi_ft.modeling.load import load_tokenizer

    return load_tokenizer()


# ----------------------------------------------------------------------- mensagens e traço
def test_parse_messages_single_turn_only():
    ok = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    assert GV.parse_messages(ok) == {"system": "s", "user": "u", "assistant": "a"}
    assert GV.parse_messages(ok[1:])["system"] is None
    multi = ok + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    assert GV.parse_messages(multi) is None
    assert GV.parse_messages(ok[:2]) is None


@pytest.mark.parametrize(
    "content, expected",
    [
        ("<think>\npasso 1\npasso 2\n</think>\n\nResposta final.", ("passo 1\npasso 2", "Resposta final.")),
        ("<think>\nx\n</think>\nok", ("x", "ok")),
    ],
)
def test_split_trace_ok(content, expected):
    assert GV.split_trace(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        "sem tags",
        "<think>\nsó abre",
        "texto antes<think>\nx\n</think>\nok",  # não começa com <think>
        "<think>\na\n</think>\nb</think>c",  # dois fechamentos
        "<think>\n<think>\na\n</think>\nb",  # duas aberturas
        "<think>\n\n</think>\nresposta",  # traço vazio
        "<think>\ntraço\n</think>\n   ",  # resposta vazia
    ],
)
def test_split_trace_rejects(content):
    assert GV.split_trace(content) is None


def test_has_think_tags():
    assert GV.has_think_tags("a <think> b") and GV.has_think_tags("a </think>") and not GV.has_think_tags("limpo")


# --------------------------------------------------------------------------------- filtros
@pytest.mark.parametrize(
    "thought, answer, expected",
    [
        ("somando 12 + 5 = 17", "O total é 17.", True),
        ("o custo é 1.500 reais", "Custa R$ 1500", True),  # `.` de milhar
        ("deu 3,5 horas", "São 3.5 horas", True),
        ("então fica 17, mas Maria deu 2", "O total é 15.", False),  # último número do traço (2) não está na resposta
        ("sem números aqui", "Resposta 4", False),
        ("...", "x", False),
    ],
)
def test_last_trace_number_rule(thought, answer, expected):
    assert GV.last_trace_number_in_answer(thought, answer) is expected


def test_residue_filter():
    assert GV.has_residue("com base no contexto fornecido, temos", "ok")
    assert GV.has_residue("ok", "A resposta fornecida está certa")
    assert not GV.has_residue("some 2 e 3", "5")


def test_language_filter_drops_english_keeps_portuguese_and_unknown():
    assert GV.trace_dropped_by_language("We need to parse the problem. So the total is that we have five of them")
    assert not GV.trace_dropped_by_language("Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5")
    assert not GV.trace_dropped_by_language("12 + 5 = 17")  # 'unk': poucas palavras funcionais, mantém


def test_bucket_is_stable_and_uniform():
    assert GV.bucket("reasoning:42") == GV.bucket("reasoning:42")
    b = [GV.bucket(f"x:{i}") for i in range(20000)]
    assert 0.18 < sum(x < 2000 for x in b) / len(b) < 0.22  # ~20%


# ---------------------------------------------------------------------------- renderização
def _labeled_text(tok, r):
    return tok.decode([t for t in r["labels"] if t != -100], skip_special_tokens=False)


def test_render_with_thinking_prefix_and_labels(tok):
    r = GV.render_example(tok, "Você resolve problemas.", "Quanto é 2+2?", "A resposta é 4.", thought="Somo 2 e 2: 4.")
    prompt_ids = r["input_ids"][: r["n_prompt"]]
    assert tok.decode(prompt_ids).endswith("<think>\n")
    assert all(x == -100 for x in r["labels"][: r["n_prompt"]])  # prompt não entra na loss
    assert _labeled_text(tok, r) == "Somo 2 e 2: 4.\n</think>\nA resposta é 4.<|im_end|>"
    assert r["labels"][-1] == -100 and r["input_ids"][-1] != GV.IM_END_ID  # o `\n` depois de <|im_end|> fica de fora
    assert GV.IM_END_ID in r["labels"]
    assert r["n_tokens"] == len(r["input_ids"]) == len(r["labels"])


def test_render_without_thinking_prefix_and_labels(tok):
    r = GV.render_example(tok, None, "Diga olá.", "Olá! Como posso ajudar?")
    assert tok.decode(r["input_ids"][: r["n_prompt"]]).endswith("<think></think>")
    assert _labeled_text(tok, r) == "Olá! Como posso ajudar?<|im_end|>"


def test_render_labels_equal_input_ids_where_not_masked(tok):
    r = GV.render_example(tok, "s", "u", "resposta", thought="traço")
    assert all(l == i for l, i in zip(r["labels"], r["input_ids"]) if l != -100)


def test_render_covers_close_tag_in_loss(tok):
    r = GV.render_example(tok, "s", "u", "resposta", thought="traço")
    assert 13 in [t for t in r["labels"] if t != -100]  # </think> é o token 13


def test_render_matches_what_generation_prompt_produces(tok):
    """O prompt do treino é o mesmo que a avaliação envia ao modelo."""
    from lumi_ft.eval.generate import render_prompt

    r = GV.render_example(tok, None, "Quanto é 2+2?", "4", thought="2+2=4")
    gen_prompt = render_prompt(tok, "Quanto é 2+2?", thinking=True)
    ids = tok(gen_prompt, add_special_tokens=False)["input_ids"]
    # a avaliação usa o template padrão (sem system); o treino sem system renderiza o mesmo prompt
    assert r["input_ids"][: len(ids)] == ids


# ------------------------------------------------------------------------ descontaminação
def test_ngram_8gram_shared_is_a_hit_seven_is_not():
    idx = NgramIndex(8)
    idx.add("Qual é o valor da soma de dois mais três vezes quatro neste caso")
    assert idx.hits("pergunta: qual é o valor da soma de dois mais três e outra coisa")  # 8 palavras seguidas em comum
    assert not idx.hits("qual é o valor da soma de e depois outra coisa")  # só 7 palavras seguidas em comum


def test_ngram_normalizes_case_and_punctuation():
    idx = NgramIndex(8)
    idx.add("A, B; C! D? E. F: G - H")
    assert idx.hits("a b c d e f g h")


def test_ngram_short_texts_never_hit():
    idx = NgramIndex(8)
    idx.add("curto demais")
    assert not idx.hits("curto demais") and len(idx) == 0


def test_flatten_and_build_index_from_nested_rows():
    row = {"question": "pergunta com bastante texto para formar oito palavras no mínimo", "choices": ["a", "uma alternativa"], "id": 3}
    assert "uma alternativa" in list(flatten_strings(row))
    idx = build_index([row], 8)
    assert idx.hits("pergunta com bastante texto para formar oito palavras")
    assert list(ngram_hashes("a b", 8)) == []

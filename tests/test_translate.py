"""Ajudantes da tradução de traços (CPU, sem modelo)."""

from lumi_ft.data import translate as TR


def test_chunk_paragraphs_groups_and_keeps_text():
    text = "\n\n".join(f"parágrafo {i} " + "x" * 300 for i in range(10))
    chunks = TR.chunk_paragraphs(text, max_chars=1000)
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)
    assert TR.join_chunks(chunks) == text
    assert TR.chunk_paragraphs("curto") == ["curto"]
    assert TR.chunk_paragraphs("a" * 5000, 1000) == ["a" * 5000]  # parágrafo maior que o limite fica inteiro


def test_check_chunk():
    src = "Okay, let me think about this problem step by step and see what happens."
    good = "Ok, deixe-me pensar sobre este problema passo a passo e ver o que acontece."
    assert TR.check_chunk(src, good, True) is None
    assert TR.check_chunk(src, good, False) == "nao_terminou"
    assert TR.check_chunk(src, "  ", True) == "vazio"
    assert TR.check_chunk(src, good + " </think>", True) == "marcador_de_raciocinio"
    assert TR.check_chunk(src, "Ok.", True) == "tamanho_incompativel"
    assert TR.check_chunk(src, good * 4, True) == "tamanho_incompativel"


def test_prompt_contains_text():
    assert TR.make_prompt("hello").endswith("Texto:\nhello") and "português do Brasil" in TR.make_prompt("x")

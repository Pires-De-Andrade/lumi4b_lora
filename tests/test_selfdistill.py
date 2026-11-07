"""Testes da aceitação de saídas do Lumi para autodestilação (CPU) e do construtor do professor."""

from lumi_ft.data import selfdistill as S
from lumi_ft.data import teacher as T

PT = "Vou raciocinar em português. Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou"
EN = "First, we can calculate the total of the apples that she has for each day, then it is 5 and the answer is"


def test_accept_think_ok():
    rec, why = S.accept_think(f"{PT}.\n</think>\nTotal 42.\n\n#### 42", True, "42")
    assert why == "ok" and rec["thought"].startswith("Vou raciocinar em português.") and rec["answer"].endswith("#### 42")


def test_accept_think_rejections():
    cases = {
        "truncado": (f"{PT}. mais", False, "42"),
        "sem_um_unico_close": (f"{PT}. #### 42", True, "42"),
        "traco_ou_resposta_vazios": (f"{PT}.\n</think>\n", True, "42"),
        "resposta_errada_ou_sem_marcador": (f"{PT}.\n</think>\nTotal.\n#### 41", True, "42"),
        "traco_nao_pt": (f"{EN}.\n</think>\nTotal.\n#### 42", True, "42"),
    }
    for expected, (text, finished, ref) in cases.items():
        assert S.accept_think(text, finished, ref) == (None, expected), expected
    assert S.accept_think(f"{PT}.\n</think>\nA</think>B\n#### 42", True, "42")[1] == "sem_um_unico_close"


def test_accept_think_brazilian_thousands():
    assert S.accept_think(f"{PT}.\n</think>\nTotal.\n#### 70.000", True, "70000")[1] == "ok"
    assert S.accept_think(f"{PT}.\n</think>\nTotal.\n#### 70.001", True, "70000")[1] != "ok"


def test_accept_nothink():
    assert S.accept_nothink("Soma: 40+2.\n\n#### 42<|im_end|>", True, "42")[0]["answer"].endswith("#### 42")
    assert S.accept_nothink("<think>x</think>#### 42", True, "42")[1] == "tem_tags_think"
    assert S.accept_nothink("#### 41", True, "42")[1] == "resposta_errada_ou_sem_marcador"
    assert S.accept_nothink("#### 42", False, "42")[1] == "truncado"


def test_number_signature_across_translations():
    a = "João tem 3 maçãs e compra mais 1.000 reais de frutas, 0,5 kg cada"
    b = "Joao possui 0.5 kg de frutas por 1000 reais e 3 maçãs"
    assert T.number_signature(a) == T.number_signature(b)
    assert T.number_signature("tem 2 e 5") != T.number_signature("tem 2 e 6")


def test_to_hash_format():
    ans = "Conta feita.\n\n**Resposta final:** 72"
    assert T.to_hash_format(ans, "72").endswith("\n\n#### 72") and "Resposta final" not in T.to_hash_format(ans, "72")
    assert T.to_hash_format("Resposta final: 72\nmais texto depois", "72") is None
    assert T.to_hash_format("sem linha final", "72") is None


def test_plain_number():
    assert T.plain_number("70.000 reais") == "70000" and T.plain_number("3,5") == "3.5" and T.plain_number("sem número") is None


def test_val_duplicate_detection(tmp_path):
    import json

    p = tmp_path / "val.jsonl"
    p.write_text(json.dumps({"id": "v", "question": "Maria tem 3 maçãs verdes e compra 12 maçãs vermelhas no mercado", "answer": "x"}) + "\n")
    v = T.load_val_signatures(str(p))
    assert T.is_val_duplicate("Maria possui 12 maçãs vermelhas e 3 maçãs verdes, comprou no mercado", v)  # outra tradução
    assert not T.is_val_duplicate("Pedro tem 3 livros e 12 canetas para guardar na mochila nova", v)  # mesmos números, outro problema
    assert not T.is_val_duplicate("Maria tem 4 maçãs verdes e compra 12 maçãs vermelhas no mercado", v)

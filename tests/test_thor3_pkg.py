"""Pacote 3 da Thor (handoff/thor3): prompts do MedMCQA e do WebInstruct, tradução com o professor e conversão para o treino."""

import filecmp
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from lumi_ft.data import gigaverbo as GV
from lumi_ft.data import thor3 as T
from lumi_ft.data import translate as TR
from lumi_ft.eval import shape

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("translate_thor", ROOT / "handoff" / "thor3" / "translate_thor.py")
X = importlib.util.module_from_spec(spec)
sys.modules["translate_thor"] = X
spec.loader.exec_module(X)
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_thor_translated as P  # noqa: E402

EN = ("Okay, so we need to find the total and then check that the answer is what the problem asks for. We have 12 apples and 30 "
      "pears, so the total is 42, which is the first value that we can use for the next step of this problem.")


# ------------------------------------------------------------------------------ cópias
def test_shared_files_are_the_tested_copies():
    for f in ("gen_traces.py", "filter_traces_en.py"):
        assert filecmp.cmp(ROOT / "handoff" / "thor2" / f, ROOT / "handoff" / "thor3" / f, shallow=False)


@pytest.mark.parametrize("text", [EN, "Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5", "x = 3", "de the e of"])
def test_vendored_helpers_match_the_repo(text):
    assert X.trace_language(text) == shape.trace_language(text)
    assert X.bucket(text) == GV.bucket(text)
    long = "\n\n".join([text] * 9)
    assert X.chunk_paragraphs(long, 1800) == TR.chunk_paragraphs(long, 1800)


# ------------------------------------------------------------------- conferência
PT_LONG = ("Ok, então precisamos achar o total e depois conferir que a resposta é o que o problema pede. Temos 12 maçãs e 30 "
           "peras, então o total é 42, que é o primeiro valor que podemos usar para o passo seguinte deste problema aqui.")


@pytest.mark.parametrize("src, out, fin, why", [
    (EN, PT_LONG, True, None),
    (EN, PT_LONG, False, "nao_terminou"),
    (EN, "", True, "vazio"),
    (EN, "<think>" + PT_LONG, True, "marcador_de_raciocinio"),
    (EN, "Resposta: \\boxed{D}", True, "tamanho_incompativel"),
    (EN, PT_LONG.replace("42", "quarenta e dois").replace("12", "doze").replace("30", "trinta"), True, "numeros_divergentes"),
    (EN, PT_LONG + " Resposta: B", True, "respondeu_em_vez_de_traduzir"),
    ("sunrise", "nascer do sol", True, None),
    ("The value is 1,000.5", "O valor é 1.000,5", True, None),
])
def test_check_piece(src, out, fin, why):
    assert X.check_piece(src, out, fin) == why


def test_question_block_roundtrip_and_failures():
    q, opts = "Which is right?", ["one", "two", "three", "four"]
    assert X.split_question_block(X.question_block(q, opts), 4) == (q, opts)
    pt = "Qual está certa?\nA) um\nB) dois\nC) três\nD) quatro"
    assert X.split_question_block(pt, 4) == ("Qual está certa?", ["um", "dois", "três", "quatro"])
    assert X.split_question_block(pt + "\nE) cinco", 4) is None
    assert X.split_question_block("Qual?\nA) um\nB) dois\nD) quatro", 4) is None


def test_split_final():
    assert X.split_final("mcq", "It is B because...\n\nResposta: B") == ("It is B because...", "Resposta: B")
    assert X.split_final("math", "Total de 42.\n#### 42\nextra") == ("Total de 42.", "#### 42")


# -------------------------------------------------------------- de ponta a ponta (servidor falso)
WORDS = {"the": "o", "of": "de", "and": "e", "to": "para", "is": "é", "that": "que", "it": "isso", "for": "para", "are": "são",
         "with": "com", "we": "nós", "have": "temos", "this": "este", "from": "de", "or": "ou", "but": "mas", "what": "que",
         "can": "pode", "so": "então", "which": "que", "then": "então", "first": "primeiro", "need": "precisamos", "find": "achar",
         "answer": "resposta", "total": "total", "check": "conferir", "next": "seguinte", "step": "passo", "value": "valor",
         "Which": "Qual", "right": "certa", "one": "um", "two": "dois", "three": "três", "four": "quatro", "It": "Isso", "because": "porque"}


class FakeTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert enable_thinking is False
        return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def fake_post(endpoint, payload, api_key, timeout):
    src = payload["prompt"].split("<texto>\n", 1)[1].split("\n</texto>", 1)[0]
    if "SABOTE" in src:
        text = "Resposta: \\boxed{D}"
    else:
        text = re.sub(r"[A-Za-z]+", lambda m: WORDS.get(m.group(0), m.group(0)), src)
    return {"choices": [{"text": text, "finish_reason": "stop"}], "usage": {"completion_tokens": len(text) // 4}}


def ids_by_bucket(want_think: bool, n: int, prefix: str) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        k = f"{prefix}:{i}"
        if (GV.bucket(k) < 7000) == want_think:
            out.append(k)
        i += 1
    return out


class Args:
    model_name, endpoint, api_key, timeout, max_tokens = "teacher", "x", "", 5, 4096
    concurrency, block, max_chars, think_frac, limit, time_budget_min = 4, 2, 1800, 0.7, 0, 0


def kept_rows():
    t1, t2 = ids_by_bucket(True, 2, "mot_sci")
    (m1,) = ids_by_bucket(False, 1, "gsm8k_train")
    mcq = {"kind": "mcq", "source": "mot_science", "question": "Which is right?", "options": ["one", "two", "three", "four"], "ref": "B", "lang": "en", "completion_tokens": 300}
    return [
        {**mcq, "id": t1, "thought": EN + "\n\n" + EN, "answer": "It is two because...\n\nResposta: B"},
        {**mcq, "id": t2, "thought": EN + " SABOTE", "answer": "It is two.\n\nResposta: B"},
        {"id": m1, "kind": "math", "source": "gsm8k_train", "question": "Quanto é 12 + 30?", "options": None, "ref": "42", "lang": "en",
         "thought": EN, "answer": "O total é 42.\n\n#### 42", "completion_tokens": 200},
    ], (t1, t2, m1)


def test_translate_end_to_end_and_resume(tmp_path):
    rows, (t1, t2, m1) = kept_rows()
    kept = tmp_path / "kept.jsonl"
    kept.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    a = Args()
    a.kept, a.out = [str(kept)], str(tmp_path / "out" / "translated.jsonl")
    rep = X.run(a, FakeTok(), fake_post)
    assert rep["exemplos"] == 3 and rep["com_raciocinio_ok"] == 1
    got = {r["id"]: r for r in map(json.loads, Path(a.out).read_text().splitlines())}
    ok = got[t1]
    assert ok["verdict"] == "ok" and ok["with_thought"] and ok["options_pt"] == ["um", "dois", "três", "quatro"]
    assert ok["question_pt"] == "Qual é certa?" and ok["answer_pt"].endswith("\n\nResposta: B") and X.trace_language(ok["thought_pt"]) == "pt"
    assert ok["thought_en"] == rows[0]["thought"] and "SABOTE" not in ok["thought_pt"]
    assert got[t2]["verdict"].startswith("t0:") and "thought_pt" not in got[t2]
    m = got[m1]  # sem raciocínio (hash >= 7000) e final já em português: nada a traduzir
    assert m["verdict"] == "ok" and not m["with_thought"] and m["question_pt"] == "Quanto é 12 + 30?" and m["answer_pt"] == "O total é 42.\n\n#### 42"
    assert X.run(a, FakeTok(), fake_post)["exemplos"] == 3  # retoma: nada novo
    assert len(Path(a.out).read_text().splitlines()) == 3

    # conversão para o treino
    out = tmp_path / "tr"
    sys.argv = ["x", a.out, "--out", str(out)]
    P.main()
    think = [json.loads(x) for x in (out / "think.jsonl").read_text().splitlines()]
    think_en = [json.loads(x) for x in (out / "think_en.jsonl").read_text().splitlines()]
    nothink = [json.loads(x) for x in (out / "nothink.jsonl").read_text().splitlines()]
    assert [r["id"] for r in think] == [t1] == [r["id"] for r in think_en] and [r["id"] for r in nothink] == [m1]
    assert think[0]["prompt"] == think_en[0]["prompt"] and think[0]["prompt"].split("\n\n")[1].startswith("Questão: Qual é certa?")
    assert think[0]["prompt"].endswith("A) um\nB) dois\nC) três\nD) quatro") and think[0]["answer"] == think_en[0]["answer"]
    assert think[0]["thought"] != think_en[0]["thought"] and think_en[0]["thought"] == rows[0]["thought"]
    assert "prompt" not in nothink[0] and nothink[0]["answer"].endswith("#### 42") and nothink[0]["source"] == "thor_gsm8k_train"


# ------------------------------------------------------------------ fontes do pacote 3
@pytest.mark.parametrize("text, expect", [
    ("What is the scientific name for modern humans?\n \n\na.\nAustralopithecus\nc.\nHomo habilis\nb.\nHomo erectus\nd.\nHomo sapiens",
     ("What is the scientific name for modern humans?", ["Australopithecus", "Homo erectus", "Homo habilis", "Homo sapiens"])),
    ("Pick one:\n(a) 1.35 N/C, down\n(b) 3.84 N/C, up\n(c) 2.99 N/C, up", ("Pick one:", ["1.35 N/C, down", "3.84 N/C, up", "2.99 N/C, up"])),
    ("Q?\nA) x\nB) y\nC) z\nD) w", ("Q?", ["x", "y", "z", "w"])),
])
def test_parse_embedded_options(text, expect):
    stem, opts, letters = T.parse_embedded_options(text)
    assert (stem, opts) == expect and letters == list("ABCD"[: len(opts)])


@pytest.mark.parametrize("text", [
    "The movement sought to end internment.employment discrimination.segregated neighborhoods",  # sem marcas
    "Q?\na) x\nb) y\nd) w",  # pula a letra c
    "Q?\na) x\nB) y\nc) z",  # caixas misturadas
    "Q?\na) x\nb) x\nc) z",  # alternativas repetidas
])
def test_parse_embedded_options_rejects(text):
    assert T.parse_embedded_options(text) is None


def test_webinstruct_and_medmcqa_items():
    row = {"id": "w1", "question": "Value? {eq}x\n= 2{/eq}\n(a) one\n(b) two\n(c) three", "answer": "b", "answer_type": "Multiple Choice", "category": "Physics"}
    it = T.webinstruct_item(row)
    assert it["ref"] == "B" and it["options"] == ["one", "two", "three"] and it["question"] == "Value? $x = 2$" and it["id"] == "webinstruct:w1"
    assert T.webinstruct_item({**row, "answer_type": "Float"}) is None
    assert T.webinstruct_item({**row, "answer": "e"}) is None
    assert T.webinstruct_item({**row, "question": "In the figure below:\n(a) one\n(b) two\n(c) three"}) is None
    med = {"id": "m1", "question": "Drug of choice?", "opa": "A1", "opb": "B1", "opc": "C1", "opd": "D1", "cop": 2, "choice_type": "single", "subject_name": "Pharmacology"}
    assert T.medmcqa_item(med)["ref"] == "C"
    assert T.medmcqa_item({**med, "choice_type": "multi"}) is None
    assert T.medmcqa_item({**med, "question": "Which aery is blocked?"}) is None
    assert T.medmcqa_item({**med, "opd": "a1"}) is None


def test_balanced_sample_caps_categories():
    items = [{"id": f"a{i}", "category": "big"} for i in range(100)] + [{"id": f"b{i}", "category": "small"} for i in range(5)]
    got = T.balanced_sample(items, 20, 0.5, seed=0)  # teto de 10 por categoria: 10 grandes + 5 pequenas
    assert len(got) == 15 and sum(x["category"] == "small" for x in got) == 5 and sum(x["category"] == "big" for x in got) == 10
    assert [x["id"] for x in got] == [x["id"] for x in T.balanced_sample(items, 20, 0.5, seed=0)]
    assert len(T.balanced_sample(items, 20, 0.9, seed=0)) == 20


def test_prompts_file_integrity():
    d = ROOT / "handoff" / "thor3" / "data"
    p = d / "prompts.jsonl"
    if not p.exists():
        pytest.skip("handoff/thor3/data não está no git; rode scripts/make_thor3_prompts.py")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == (d / "prompts.sha256").read_text().split()[0]
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 12000 and {r["kind"] for r in rows} == {"mcq"}
    assert all(len(r["options"]) >= 3 and r["ref"] in "ABCDEFGHIJ"[: len(r["options"])] and r["prompt"].endswith(r["options"][-1]) for r in rows)
    assert all("Resposta: <letter>" in r["prompt"] for r in rows)

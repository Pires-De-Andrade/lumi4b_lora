"""Pacote da Thor para o Nemotron (handoff/thor2): filtro com matemática e múltipla escolha, e integridade dos arquivos."""

import filecmp
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "handoff"
spec = importlib.util.spec_from_file_location("filter_traces_en", ROOT / "thor2" / "filter_traces_en.py")
F = importlib.util.module_from_spec(spec)
sys.modules["filter_traces_en"] = F
spec.loader.exec_module(F)

EN = "Okay, so we need to find the total and then check that the answer is what the problem asks for, since it is the first step"
PT = "Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou muito"


def test_gen_traces_is_the_same_tested_file():
    assert filecmp.cmp(ROOT / "thor" / "gen_traces.py", ROOT / "thor2" / "gen_traces.py", shallow=False)


def test_prompts_file_integrity():
    d = ROOT / "thor2" / "data"
    p = d / "prompts.jsonl"
    if not p.exists():
        pytest.skip("handoff/thor2/data não está no git; rode scripts/make_thor2_prompts.py")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == (d / "prompts.sha256").read_text().split()[0]
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) and {r["kind"] for r in rows} == {"math", "mcq"}
    mcq = [r for r in rows if r["kind"] == "mcq"]
    assert all(len(r["options"]) >= 3 and r["ref"] in "ABCDEFGHIJ"[: len(r["options"])] and "Resposta: <letter>" in r["prompt"] for r in mcq)
    assert all("#### <número>" in r["prompt"] and r["ref"] for r in rows if r["kind"] == "math")


@pytest.mark.parametrize("kind, answer, ref, why", [
    ("math", "Total.\n\n#### 42", "42", None), ("math", "Total.\n#### 4.200", "4200", None), ("math", "Total.\n#### 41", "42", "resposta_diferente_da_referência"),
    ("math", "Total 42.", "42", "sem_####_na_resposta"),
    ("mcq", "Explicação.\n\nResposta: B", "B", None), ("mcq", "Explicação.\n\n**Resposta: B**", "B", None), ("mcq", "Explicação.\n\nResposta: B.", "B", None),
    ("mcq", "Explicação.\n\nResposta: C", "B", "resposta_diferente_da_referência"), ("mcq", "Resposta: B\nmais texto", "B", "sem_Resposta_letra_na_ultima_linha"),
    ("mcq", "A resposta é B", "B", "sem_Resposta_letra_na_ultima_linha"), ("mcq", "", "B", "sem_Resposta_letra_na_ultima_linha"),
])
def test_check_answer(kind, answer, ref, why):
    assert F.check_answer(kind, answer, ref) == why


def test_filter_end_to_end(tmp_path):
    prompts = [{"id": "m1", "kind": "math", "source": "s", "question": "q1", "prompt": "p", "ref": "42"},
               {"id": "m2", "kind": "math", "source": "s", "question": "q2", "prompt": "p", "ref": "42"},
               {"id": "m3", "kind": "math", "source": "s", "question": "q3", "prompt": "p", "ref": "42"},
               {"id": "c1", "kind": "mcq", "source": "s", "question": "q4", "options": ["a", "b", "c"], "prompt": "p", "ref": "B"},
               {"id": "c2", "kind": "mcq", "source": "s", "question": "q5", "options": ["a", "b", "c"], "prompt": "p", "ref": "B"},
               {"id": "c3", "kind": "mcq", "source": "s", "question": "q6", "options": ["a", "b", "c"], "prompt": "p", "ref": "B"},
               {"id": "c4", "kind": "mcq", "source": "s", "question": "q7", "options": ["a", "b", "c"], "prompt": "p", "ref": "B"}]
    def row(i, text, reason="stop", tok=300):
        return {"id": i, "source": "s", "ref": next(p["ref"] for p in prompts if p["id"] == i), "text": text, "finish_reason": reason, "completion_tokens": tok}
    traces = [row("m1", f"{EN}.\n</think>\nTotal.\n\n#### 42"), row("m2", f"{PT}.\n</think>\nTotal.\n#### 41"), row("m3", f"{EN}. sem fechar", "length"),
              row("c1", f"{EN}.\n</think>\nIt is B.\n\nResposta: B"), row("c2", f"{EN}.\n</think>\nIt is C.\n\nResposta: C"),
              row("c3", f"{EN}.\n</think>\nIt is B.\n\nResposta: B", tok=9000), {"id": "c4", "source": "s", "ref": "B", "error": "x"}]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    sys.argv = ["x", "--traces", str(tmp_path / "t.jsonl"), "--prompts", str(tmp_path / "p.jsonl"), "--outdir", str(tmp_path / "o")]
    assert F.main() == 0
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    fun = rep["funil"]
    assert fun["gerados"] == 6 and fun["mantidos"] == 2 and fun["truncado"] == 1 and fun["longo_demais"] == 1
    assert fun["resposta_diferente_da_referência"] == 2 and fun["erro_de_requisição"] == 1
    kept = {k["id"]: k for k in map(json.loads, (tmp_path / "o" / "kept.jsonl").read_text().splitlines())}
    assert set(kept) == {"m1", "c1"} and kept["m1"]["lang"] == "en" and kept["c1"]["options"] == ["a", "b", "c"] and "</think>" not in kept["c1"]["thought"]
    assert rep["por_tipo"] == {"math": {"total": 3, "mantidos": 1}, "mcq": {"total": 4, "mantidos": 1}}
    assert (tmp_path / "o" / "samples.md").exists()

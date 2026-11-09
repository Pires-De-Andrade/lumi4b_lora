"""prepare_thor4.py: kept.jsonl do pacote 4 -> registros de treino (CPU)."""
import json
import subprocess
import sys

KEPT = [
    {"id": "m1", "source": "medmcqa", "category": "x", "prompt": "Responda...\n\nPergunta?\nA. a\nB. b", "thought": "Vou pensar.",
     "answer": "Explicação.\n\nResposta: B", "ref": "B", "lang": "pt", "completion_tokens": 300},
    {"id": "w1", "source": "webinstruct", "category": "y", "prompt": "Responda...\n\nOutra?\nA. a\nB. b", "thought": "Certo.",
     "answer": "Explicação.\n\n**Resposta: A**", "ref": "A", "lang": "pt", "completion_tokens": 500},
    {"id": "bad", "source": "medmcqa", "category": "x", "prompt": "p", "thought": "t", "answer": "sem linha final", "ref": "A", "lang": "pt"},
    {"id": "nothought", "source": "medmcqa", "category": "x", "prompt": "p", "thought": "", "answer": "Resposta: A", "ref": "A", "lang": "pt"},
]


def test_prepare_thor4_converts_and_drops_incomplete(tmp_path):
    kept = tmp_path / "kept.jsonl"
    kept.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in KEPT))
    out = tmp_path / "prep"
    r = subprocess.run([sys.executable, "scripts/prepare_thor4.py", "--kept", str(kept), "--out-dir", str(out)], capture_output=True, text=True,
                       env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    recs = [json.loads(line) for line in (out / "think.jsonl").read_text().splitlines()]
    assert [x["id"] for x in recs] == ["qwen_mcq:m1", "qwen_mcq:w1"]
    a = recs[0]
    assert a["prompt"] == a["question"] == KEPT[0]["prompt"] and a["letter"] == "B" and a["source"] == "qwen_medmcqa"
    assert a["thought"] == "Vou pensar." and a["answer"].endswith("Resposta: B") and a["problem_id"] == a["id"]
    rep = json.loads((out / "prepare_report.json").read_text())
    assert rep["lidos"] == 4 and rep["usados"] == 2 and rep["descartado_sem_resposta_final"] == 2

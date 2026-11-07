"""Testes do pacote da Thor (handoff/thor): cliente de geração e filtro, com um servidor falso de /v1/completions."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "handoff" / "thor"))
import filter_traces as F  # noqa: E402
import gen_traces as G  # noqa: E402

PT_TRACE = " Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou"
EN_TRACE = " First, we can calculate the total of the apples that she has for each day, then it is 5 and the answer is"


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silencioso
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["prompt"]
        if "[ERR]" in prompt:
            self.send_response(500)
            self.end_headers()
            return
        text, reason = f"{PT_TRACE}.\n</think>\nA resposta é 42.\n#### 42", "stop"
        if "[NOCLOSE]" in prompt:
            text = f"{PT_TRACE}. #### 42"
        elif "[WRONG]" in prompt:
            text = f"{PT_TRACE}.\n</think>\nA resposta é 41.\n#### 41"
        elif "[EN]" in prompt:
            text = f"{EN_TRACE}.\n</think>\nThe answer is 42.\n#### 42"
        elif "[LONG]" in prompt:
            text, reason = PT_TRACE + " e continua", "length"
        elif "[DOTS]" in prompt:
            text = f"{PT_TRACE}.\n</think>\nTotal.\n#### 4.200"
        out = json.dumps({"choices": [{"text": text, "finish_reason": reason}], "usage": {"completion_tokens": 123}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1/completions"
    srv.shutdown()


@pytest.fixture(scope="module")
def tok_path():
    from huggingface_hub import snapshot_download

    from lumi_ft.modeling.load import MODEL_ID, REVISION

    return snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)


def _prompts(tmp_path, flags):
    p = tmp_path / "prompts.jsonl"
    rows = [{"id": f"p{i}", "source": "t", "question": f"{f} problema {i}", "prompt": f"Resolva {f} problema {i}", "ref": "42" if f != "[DOTS]" else "4200"}
            for i, f in enumerate(flags)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _run(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["gen_traces.py", *args])
    monkeypatch.setattr(G, "BACKOFF_S", 0.01)
    return G.main()


def test_render_ends_with_think_and_prefill(tok_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path)
    text = G.render(tok, "oi", "Vou raciocinar em português.")
    assert text.endswith("<think>\nVou raciocinar em português.") and text.count("<think>") == 1


def test_generate_then_filter_end_to_end(tmp_path, monkeypatch, server, tok_path):
    prompts = _prompts(tmp_path, ["[OK]", "[OK]", "[NOCLOSE]", "[WRONG]", "[EN]", "[LONG]", "[DOTS]"])
    out = tmp_path / "out" / "traces.jsonl"
    rc = _run(monkeypatch, ["--model-name", "fake", "--tokenizer", tok_path, "--prompts", str(prompts), "--out", str(out),
                            "--endpoint", server, "--concurrency", "4"])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 7 and all("error" not in r for r in rows)
    assert all(r["text"].startswith("Vou raciocinar em português.") for r in rows)  # o prefill entra na saída pontuada
    monkeypatch.setattr(sys, "argv", ["filter_traces.py", "--traces", str(out), "--prompts", str(prompts), "--outdir", str(tmp_path / "f")])
    assert F.main() == 0
    rep = json.loads((tmp_path / "f" / "report.json").read_text())
    fun = rep["funil"]
    assert fun["gerados"] == 7 and fun["mantidos"] == 3  # dois [OK] e o [DOTS] (4.200 = 4200 em português)
    assert fun["truncado"] == 1 and fun["sem_um_unico_</think>"] == 1
    assert fun["resposta_diferente_da_referência"] == 1 and fun["traço_não_está_em_português"] == 1
    kept = [json.loads(line) for line in (tmp_path / "f" / "kept.jsonl").read_text().splitlines()]
    assert {k["id"] for k in kept} == {"p0", "p1", "p6"}
    assert all(k["thought"].startswith("Vou raciocinar em português.") and "</think>" not in k["thought"] for k in kept)
    assert kept[0]["question"].endswith(("problema 0", "problema 1", "problema 6"))
    assert (tmp_path / "f" / "samples.md").exists()


def test_resume_skips_done_and_retries_errors(tmp_path, monkeypatch, server, tok_path):
    prompts = _prompts(tmp_path, ["[OK]", "[ERR]", "[OK]"])
    out = tmp_path / "traces.jsonl"
    args = ["--model-name", "fake", "--tokenizer", tok_path, "--prompts", str(prompts), "--out", str(out), "--endpoint", server, "--concurrency", "2"]
    _run(monkeypatch, args)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["id"] for r in rows if "error" not in r) == ["p0", "p2"] and [r["id"] for r in rows if "error" in r] == ["p1"]
    n = len(rows)
    _run(monkeypatch, args)  # segunda rodada: só o p1 (que falhou) é tentado de novo
    rows2 = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows2) == n + 1 and rows2[-1]["id"] == "p1" and "error" in rows2[-1]


def test_start_and_limit_and_time_budget(tmp_path, monkeypatch, server, tok_path):
    prompts = _prompts(tmp_path, ["[OK]"] * 6)
    out = tmp_path / "t.jsonl"
    _run(monkeypatch, ["--model-name", "fake", "--tokenizer", tok_path, "--prompts", str(prompts), "--out", str(out),
                       "--endpoint", server, "--start", "2", "--limit", "3"])
    assert sorted(json.loads(line)["id"] for line in out.read_text().splitlines()) == ["p2", "p3", "p4"]
    out2 = tmp_path / "t2.jsonl"
    assert _run(monkeypatch, ["--model-name", "fake", "--tokenizer", tok_path, "--prompts", str(prompts), "--out", str(out2),
                              "--endpoint", server, "--time-budget-min", "0.0000001"]) == 0
    assert not out2.exists() or out2.read_text() == ""  # orçamento esgotado antes do primeiro pedido


# ----------------------------------------------------------------------------- filtro
@pytest.mark.parametrize("a, b, equal", [("70.000", "70000", True), ("3,5", "3.5", True), ("1.234,56", "1234.56", True),
                                         ("18.", "18", True), ("$18", "18", True), ("70.001", "70000", False), ("abc", "1", None)])
def test_to_dec_ptbr(a, b, equal):
    x, y = F.to_dec(a), F.to_dec(b)
    assert (x is None) if equal is None else ((x == y) is equal)


def test_split_requires_exactly_one_close():
    assert F.split("a</think>b") == ("a", "b") and F.split("<think>a</think>b") == ("a", "b")
    assert F.split("sem") is None and F.split("a</think>b</think>c") is None

"""`scripts/translate_traces.py` com um vLLM falso: registros `ready_pt` (só o raciocínio é traduzido) e registros em inglês."""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from lumi_ft.data import gigaverbo as GV

ROOT = Path(__file__).resolve().parent.parent


class _Out:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop", token_ids=[1] * 5)]


class FakeLLM:
    def __init__(self, **kw):
        pass

    def generate(self, prompts, sp, use_tqdm=False):
        FakeLLM.calls.append(len(prompts))
        outs = []
        for p in prompts:
            src = p.split("Texto:\n", 1)[1].split("<|im_end|>")[0]
            # "tradução": mantém o texto (sem palavras funcionais em inglês) e acrescenta palavras funcionais do português em cada parágrafo
            outs.append(_Out("\n\n".join(par + " de que não uma para com" for par in src.split("\n\n"))))
        return outs


FakeLLM.calls = []


def _load_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("translate_traces_t", ROOT / "scripts" / "translate_traces.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pid(want_think: bool) -> str:
    i = 0
    while (GV.bucket(f"nemotron_pt:t{i}") < 7000) != want_think:
        i += 1
    return f"nemotron_pt:t{i}"


@pytest.fixture(scope="module")
def tok_ok():
    from huggingface_hub import snapshot_download

    from lumi_ft.modeling.load import MODEL_ID, REVISION

    try:
        snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer do Lumi fora do cache")


def test_ready_pt_translates_only_the_thought(tmp_path, monkeypatch, tok_ok):
    m = _load_script(monkeypatch)
    think_id, nothink_id = _pid(True), _pid(False)
    base = {"question": "Qual é a cor?", "options": ["verde", "azul", "vermelho", "preto"], "letter": "B", "final": "É a B.", "source": "nemotron_pt", "ready_pt": True}
    par = "alpha beta gamma delta epsilon zeta " * 12  # parágrafo longo o bastante para o acréscimo da tradução falsa caber na razão de tamanho
    recs = [{**base, "id": "a", "problem_id": think_id, "thought": f"{par}\n\n{par}"},
            {**base, "id": "b", "problem_id": nothink_id, "thought": par}]
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in recs))
    FakeLLM.calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--n", "2", "--out", str(tmp_path / "o"), "--chunk", "10"])
    m.main()
    assert FakeLLM.calls == [1]  # um pedido: o único raciocínio (um bloco); sem pedidos de enunciado nem de texto final
    think = [json.loads(line) for line in (tmp_path / "o" / "think.jsonl").read_text().splitlines()]
    nothink = [json.loads(line) for line in (tmp_path / "o" / "nothink.jsonl").read_text().splitlines()]
    assert len(think) == 1 and len(nothink) == 1
    t = think[0]
    assert t["id"] == "nemotron_pt:a" and t["source"] == "nemotron_pt" and t["answer"] == "É a B.\n\nResposta: B" and t["options"] == base["options"]
    assert "de que não uma para com" in t["thought"] and t["prompt"].count("A) verde") == 1 and "Qual é a cor?" in t["prompt"]
    assert nothink[0]["answer"] == "É a B.\n\nResposta: B" and "thought" not in nothink[0]
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    assert rep["verdicts"] == {"think:ok": 1, "nothink:ok": 1}

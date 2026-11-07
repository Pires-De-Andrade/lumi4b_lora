"""Testes do harness de avaliação (CPU, sem modelo): lotes, retomada, prompt, forma da saída."""

import json

import pytest

from lumi_ft.eval import generate as G
from lumi_ft.eval import gsm8k as K
from lumi_ft.eval import parse as P
from lumi_ft.eval.shape import length_stats, trace_language


# ------------------------------------------------------------------------------ lotes
def test_plan_orders_by_length_descending_with_stable_ties():
    batches = G.plan_batches([10, 50, 50, 30], max_batch=2, token_budget=10_000)
    assert batches == [[1, 2], [3, 0]]


def test_plan_covers_every_item_once():
    lengths = [(i * 37) % 900 + 20 for i in range(203)]
    flat = [i for b in G.plan_batches(lengths) for i in b]
    assert sorted(flat) == list(range(203))


def test_plan_is_deterministic():
    lengths = [(i * 53) % 1500 + 10 for i in range(100)]
    assert G.plan_batches(lengths) == G.plan_batches(list(lengths))


def test_plan_shrinks_batch_for_long_prompts():
    # 32 x 1.024 = 32.768 cabe; 2.000 tokens pedem 16; acima do orçamento, 1 por lote
    assert len(G.plan_batches([1024] * 32)[0]) == 32
    assert len(G.plan_batches([2000] * 40)[0]) == 16
    assert [len(b) for b in G.plan_batches([40_000, 40_000])] == [1, 1]


def test_plan_never_exceeds_token_budget_or_max_batch():
    lengths = [(i * 91) % 3000 + 5 for i in range(300)]
    for b in G.plan_batches(lengths):
        assert len(b) <= G.MAX_BATCH
        assert len(b) == 1 or len(b) * max(lengths[i] for i in b) <= G.TOKEN_BUDGET


def test_cut_at_eos():
    assert G.cut_at_eos([5, 6, 11, 0, 0]) == ([5, 6], True)
    assert G.cut_at_eos([5, 2, 7]) == ([5], True)
    assert G.cut_at_eos([5, 6, 7]) == ([5, 6, 7], False)
    assert G.cut_at_eos([11]) == ([], True)


# --------------------------------------------------------------------------- retomada
def _items(n=7):
    return [{"id": 100 + i, "prompt": f"p{i}", "n_prompt": 10 * (i + 1)} for i in range(n)]


def _fake(calls):
    def fn(prompts):
        calls.append(list(prompts))
        return [{"text": f"{p}</think>#### 1", "n_gen": len(p), "finished": True, "gen_len": 9} for p in prompts]

    return fn


def test_run_batches_resumes_without_regenerating_or_changing_batches(tmp_path):
    out = tmp_path / "g.jsonl"
    items = _items()
    calls1: list = []
    full = G.run_batches(items, _fake(calls1), out, max_batch=3, log=lambda s: None)
    assert len(full) == 7 and len(calls1) == 3

    # Simula queda depois do 1º lote: mantém cabeçalho + lote 0
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:2]) + "\n")
    calls2: list = []
    resumed = G.run_batches(items, _fake(calls2), out, max_batch=3, log=lambda s: None)
    assert len(calls2) == 2 and calls2 == calls1[1:]  # mesmos lotes, só os que faltavam
    assert [r["id"] for r in resumed] == [r["id"] for r in full]


def test_run_batches_rejects_different_configuration(tmp_path):
    out = tmp_path / "g.jsonl"
    G.run_batches(_items(), _fake([]), out, max_batch=3, log=lambda s: None)
    with pytest.raises(AssertionError):
        G.run_batches(_items(), _fake([]), out, max_batch=4, log=lambda s: None)
    with pytest.raises(AssertionError):
        G.run_batches(_items(5), _fake([]), out, max_batch=3, log=lambda s: None)


def test_run_batches_records_batch_steps_and_item_fields(tmp_path):
    out = tmp_path / "g.jsonl"
    G.run_batches(_items(4), _fake([]), out, max_batch=4, log=lambda s: None)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert "header" in lines[0] and lines[1]["size"] == 4 and lines[1]["gen_len"] == 9
    assert {"id", "n_prompt", "text", "n_gen", "finished"} <= set(lines[1]["items"][0])


# ---------------------------------------------------------------------------- prompt
@pytest.fixture(scope="module")
def tok():
    from lumi_ft.modeling.load import load_tokenizer

    return load_tokenizer()


def test_prompt_ends_per_mode_and_matches_chat_template_tokens(tok):
    q = K.build_prompt("Quanto é 2+2?")
    on = G.render_prompt(tok, q, thinking=True)
    off = G.render_prompt(tok, q, thinking=False)
    assert on.endswith("<think>\n") and off.endswith("<think></think>")
    ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=True)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    assert tok(on, add_special_tokens=False)["input_ids"] == ids  # sem BOS extra


def test_gsm8k_prompt_asks_for_final_marker_and_has_no_examples():
    p = K.build_prompt("Quanto é 2+2?")
    assert "#### <número>" in p and p.count("Problema:") == 1 and p.endswith("Quanto é 2+2?")


def test_sample_ids_fixed_by_seed():
    a = K.sample_ids(256, seed=0)
    assert a == K.sample_ids(256, seed=0) and a != K.sample_ids(256, seed=1)
    assert len(set(a)) == 256 and a == sorted(a) and max(a) < K.N_TEST
    assert K.sample_ids(None) == list(range(K.N_TEST))


# ------------------------------------------------------------------------ pontuação
def _gen(text, finished=True, n=50):
    return {"id": 1, "text": text, "n_gen": n, "finished": finished}


def test_score_row_thinking_ok():
    r = K.score_row(_gen("12+5=17</think>\nSomando, 17.\n#### 17"), "...\n#### 17", thinking=True)
    assert r["strict_ok"] and r["flexible_ok"] and r["status"] == P.OK
    assert r["closed"] and r["trace_nonempty"] and r["trace_last_number_ok"]


def test_score_row_truncated_trace_is_an_error():
    r = K.score_row(_gen("passo 1... #### 17", finished=False, n=4096), "#### 17", thinking=True)
    assert r["status"] == P.NO_CLOSE and not r["strict_ok"] and not r["flexible_ok"] and not r["closed"]


def test_score_row_trace_ends_on_wrong_number_but_answer_right():
    r = K.score_row(_gen("então 17, mas Maria deu 2</think>#### 15"), "#### 15", thinking=True)
    assert r["strict_ok"] and r["trace_last_number_ok"] is False


def test_score_row_nothinking_has_no_trace_fields():
    r = K.score_row(_gen("Somando, 17.\n#### 17"), "#### 17", thinking=False)
    assert r["strict_ok"] and r["trace_language"] is None and r["trace_last_number_ok"] is None


def test_summarize_batches_and_rates():
    gens = [
        {"id": 1, "n_prompt": 9, "text": "a</think>#### 1", "n_gen": 100, "finished": True, "gen_len": 100},
        {"id": 2, "n_prompt": 9, "text": "b sem fechar", "n_gen": 4096, "finished": False, "gen_len": 4096},
    ]
    batches = [{"batch": 0, "size": 2, "seconds": 10.0, "gen_len": 4096, "items": gens}]
    rows = [K.score_row(g, "#### 1", thinking=True) for g in gens]
    s = K.summarize(rows, batches, thinking=True)
    assert s["strict_acc"] == 0.5 and s["truncated_rate"] == 0.5 and s["closed_rate"] == 0.5
    assert s["status_rates"][P.NO_CLOSE] == 0.5
    assert s["batches"]["share_with_truncated_item"] == 1.0
    assert s["batches"]["useful_token_share"] == round((100 + 4096) / (2 * 4096), 3)


# ------------------------------------------------------------------------ idioma e tamanho
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Primeiro, vamos calcular o total de maçãs que ela tem para cada dia, então são 5 porque ela já comprou", "pt"),
        ("First, we can calculate the total of the apples that she has for each day, then it is 5 and the answer is", "en"),
        ("12 + 5 = 17", "unk"),
        ("o total de maçãs que ela tem, and the total of apples that she has", "mixed"),
    ],
)
def test_trace_language(text, expected):
    assert trace_language(text) == expected


def test_length_stats():
    s = length_stats(list(range(1, 101)))
    assert s["n"] == 100 and s["median"] in (50, 51) and s["p95"] in (95, 96) and s["max"] == 100
    assert length_stats([]) == {"n": 0}

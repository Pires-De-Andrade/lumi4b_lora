"""Auxiliares do diagnóstico de log-probabilidades do RL (CPU)."""
import json

import pytest
import torch

from lumi_ft.train import logprob_diag as D


def _write(tmp_path, n=10):
    p = tmp_path / "r.jsonl"
    rows = [{"prompt_ids": list(range(5)), "completion_ids": list(range(10, 10 + 20 + i))} for i in range(n)]
    rows.append({"prompt_ids": [1, 2], "completion_ids": [3]})  # curta demais (< 16 tokens)
    rows.append({"prompt_ids": list(range(4000)), "completion_ids": list(range(100))})  # longa demais
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_pick_samples_is_deterministic_and_filters_length(tmp_path):
    f = _write(tmp_path)
    a, b = D.pick_samples(f, 6, seed=3), D.pick_samples(f, 6, seed=3)
    assert a == b and len(a) == 6
    assert all(len(s["completion_ids"]) >= 16 and len(s["prompt_ids"]) + len(s["completion_ids"]) <= 3072 for s in a)
    assert D.signature(a) == D.signature(b) and D.signature(a) != D.signature(D.pick_samples(f, 6, seed=4))


def test_summarize_reports_per_token_and_per_sequence_statistics():
    diffs = [[1e-4, 1e-4, 1e-4], [-5e-4, -5e-4], [0.0, 0.0, 0.0, 0.0]]
    s = D.summarize(diffs)
    assert s["sequences"] == 3 and s["tokens"] == 9
    assert s["frac_seq_abs_gt_0.0003"] == pytest.approx(1 / 3, abs=1e-4)  # só a segunda passa de 3e-4
    assert s["frac_seq_abs_gt_0.0004"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["seq_abs_p90"] == pytest.approx(5e-4, abs=1e-6)
    assert s["mean_abs_token_diff"] == pytest.approx((3e-4 + 1e-3) / 9, abs=1e-6)


def test_diff_lists_requires_same_lengths():
    assert D.diff_lists({0: [1.0, 2.0]}, {0: [0.5, 0.5]}) == [[0.5, 1.5]]
    with pytest.raises(AssertionError):
        D.diff_lists({0: [1.0]}, {0: [1.0, 2.0]})


def test_shuffled_chunks_cover_every_index_once():
    chunks = D.shuffled_chunks(10, 4, seed=1)
    assert sorted(i for c in chunks for i in c) == list(range(10)) and [len(c) for c in chunks] == [4, 4, 2]
    assert chunks == D.shuffled_chunks(10, 4, seed=1) and chunks != D.shuffled_chunks(10, 4, seed=2)


def test_fp32_head_matches_a_float_matmul_and_can_be_restored():
    torch.manual_seed(0)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(8, 5, bias=False).to(torch.bfloat16)

    m = Inner()
    w = m.lm_head.weight.detach().float().clone()
    original = D.fp32_head(m)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    y = m.lm_head(x)
    assert y.dtype == torch.float32 and torch.allclose(y, x.float() @ w.T, atol=1e-6)
    m.lm_head = original
    assert m.lm_head.weight.dtype == torch.bfloat16

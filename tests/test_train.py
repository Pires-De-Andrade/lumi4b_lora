"""Testes da lógica de treino (CPU): lotes, máscara, normalização da loss, schedule e leitura dos dados."""

import math

import pytest
import torch
import torch.nn.functional as F

from lumi_ft.data import gigaverbo as GV
from lumi_ft.train import sft as S


def _row(n, n_prompt=3, mode="think", bucket=0, viab=False, loss_end=None, i=0):
    ids = list(range(10 + i, 10 + i + n))
    return {"id": f"x:{i}", "mode": mode, "group": "g", "task_type": "t", "bucket": bucket, "viab": viab,
            "n_tokens": n, "n_prompt": n_prompt, "loss_end": loss_end if loss_end is not None else n - 1, "input_ids": ids}


# ----------------------------------------------------------------------------- batches
def test_epoch_batches_cover_every_index_once():
    b = S.epoch_batches(103, 32, seed=0, epoch=0)
    assert [len(x) for x in b] == [32, 32, 32, 7]
    assert sorted(i for x in b for i in x) == list(range(103))


def test_epoch_batches_deterministic_and_vary_by_epoch_and_seed():
    a = S.epoch_batches(100, 32, 0, 0)
    assert a == S.epoch_batches(100, 32, 0, 0)
    assert a != S.epoch_batches(100, 32, 0, 1) and a != S.epoch_batches(100, 32, 1, 0)


def test_micro_batches_respect_token_budget_and_cover_all():
    rows = [_row(n, i=i) for i, n in enumerate([1000, 900, 500, 480, 300, 100, 1024, 700])]
    for budget in (1024, 2048):
        groups = S.micro_batches(rows, list(range(len(rows))), budget)
        assert sorted(i for g in groups for i in g) == list(range(len(rows)))
        for g in groups:
            assert len(g) * max(len(rows[i]["input_ids"]) for i in g) <= budget


def test_micro_batches_pairs_for_lora_budget_single_for_dora_when_long():
    rows = [_row(1000, i=0), _row(990, i=1), _row(980, i=2), _row(970, i=3)]
    assert [len(g) for g in S.micro_batches(rows, [0, 1, 2, 3], 2048)] == [2, 2]  # LoRA: 2 sequências
    assert [len(g) for g in S.micro_batches(rows, [0, 1, 2, 3], 1024)] == [1, 1, 1, 1]  # DoRA: 1 sequência
    short = [_row(400, i=0), _row(300, i=1)]
    assert [len(g) for g in S.micro_batches(short, [0, 1], 1024)] == [2]  # curtas cabem juntas mesmo no DoRA


def test_micro_batches_group_similar_lengths():
    rows = [_row(n, i=i) for i, n in enumerate([100, 900, 110, 880])]
    groups = S.micro_batches(rows, [0, 1, 2, 3], 2048)
    assert sorted(sorted(g) for g in groups) == [[0, 2], [1, 3]]  # pares de comprimento parecido: pouco padding


# ------------------------------------------------------------------------ labels e collate
def test_labels_for_masks_prompt_and_trailing_tokens():
    r = _row(10, n_prompt=3, loss_end=9)
    labs = S.labels_for(r)
    assert labs[:3] == [-100] * 3 and labs[3:9] == r["input_ids"][3:9] and labs[9] == -100 and len(labs) == 10


def test_collate_right_pads_and_masks():
    rows = [_row(5, n_prompt=2, loss_end=4, i=0), _row(8, n_prompt=3, loss_end=7, i=1)]
    b = S.collate(rows, [0, 1])
    assert b["input_ids"].shape == (2, 8) and b["attention_mask"].tolist()[0] == [1] * 5 + [0] * 3
    assert b["input_ids"][0, 5:].tolist() == [S.PAD_ID] * 3
    assert (b["labels"][0, 5:] == -100).all() and (b["labels"][0, :2] == -100).all()
    assert b["labels"][0, 2:4].tolist() == rows[0]["input_ids"][2:4]


def test_labels_for_matches_render_example_labels():
    from lumi_ft.modeling.load import load_tokenizer

    tok = load_tokenizer()
    for thought in ("Somo 2 e 2: 4.", None):
        r = GV.render_example(tok, "Você é útil.", "Quanto é 2+2?", "A resposta é 4.", thought=thought)
        row = {"input_ids": r["input_ids"], "n_prompt": r["n_prompt"], "loss_end": r["loss_end"]}
        assert S.labels_for(row) == r["labels"]
        assert S.n_targets([row], [0]) == sum(x != -100 for x in r["labels"])


# ------------------------------------------------------------------- normalização da loss
def _ref_mean_loss(logits, labels):
    """Referência: cross-entropy média por token-alvo, deslocando os logits (forma textbook)."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)


def test_batch_loss_sum_matches_textbook_shifted_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11)
    labels = torch.randint(0, 11, (3, 7))
    labels[:, :3] = -100
    labels[1, 5:] = -100
    n = int((labels[:, 1:] != -100).sum())
    assert torch.allclose(S.batch_loss_sum(logits, labels) / n, _ref_mean_loss(logits, labels), atol=1e-6)


def test_accumulated_loss_equals_single_batch_loss_however_it_is_sliced():
    """O que garante que LoRA (micro-batches de 2) e DoRA (de 1) otimizam o mesmo objetivo."""
    torch.manual_seed(1)
    logits = torch.randn(6, 9, 13)
    labels = torch.randint(0, 13, (6, 9))
    for i, k in enumerate([2, 3, 4, 3, 5, 2]):  # prompts e preenchimentos de tamanhos diferentes
        labels[i, :k] = -100
    labels[2, 7:] = -100
    N = int((labels[:, 1:] != -100).sum())
    full = _ref_mean_loss(logits, labels)
    for sizes in ([6], [2, 2, 2], [1] * 6, [4, 2], [3, 1, 2]):
        total, pos = 0.0, 0
        for s in sizes:
            total += S.batch_loss_sum(logits[pos: pos + s], labels[pos: pos + s]) / N
            pos += s
        assert torch.allclose(total, full, atol=1e-6), sizes


def test_n_targets_counts_loss_positions():
    rows = [_row(10, n_prompt=3, loss_end=9, i=0), _row(6, n_prompt=2, loss_end=5, i=1)]
    assert S.n_targets(rows, [0, 1]) == 6 + 3


# ------------------------------------------------------------------------------ schedule
def test_lr_schedule_warmup_peak_and_cosine_to_zero():
    total, base = 1000, 1e-4
    lrs = [S.lr_at(s, total, base, 0.03) for s in range(total)]
    assert lrs[0] < lrs[10] < lrs[29] and math.isclose(lrs[30], base, rel_tol=1e-3)  # warmup de 30 passos
    assert all(a >= b for a, b in zip(lrs[30:], lrs[31:]))  # cosseno decrescente
    assert lrs[-1] < base * 1e-3


def test_lr_schedule_tiny_runs_do_not_divide_by_zero():
    assert S.lr_at(0, 1, 1e-4) > 0 and S.lr_at(0, 5, 1e-4, 0.03) > 0


# ----------------------------------------------------------------------------- leitura
@pytest.fixture()
def parquet_path(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_row(8, mode="think", bucket=100, viab=True, i=0), _row(8, mode="think", bucket=5000, viab=False, i=1),
            _row(8, mode="nothink", bucket=1500, viab=True, i=2), _row(8, mode="nothink", bucket=9000, viab=False, i=3)]
    p = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pylist(rows), p)
    return str(p)


def test_load_rows_filters(parquet_path):
    assert len(S.load_rows(parquet_path)) == 4
    assert [r["mode"] for r in S.load_rows(parquet_path, mix="100/0")] == ["think", "think"]
    assert [r["id"] for r in S.load_rows(parquet_path, viab=True)] == ["x:0", "x:2"]
    assert [r["id"] for r in S.load_rows(parquet_path, frac=0.2)] == ["x:0", "x:2"]  # bucket < 2.000
    assert [r["id"] for r in S.load_rows(parquet_path, mix="100/0", frac=0.2)] == ["x:0"]
    with pytest.raises(ValueError):
        S.load_rows(parquet_path, mix="50/50")


def test_default_budget_per_arm():
    assert S.default_budget("lora") == 2048 and S.default_budget("dora") == 1024

"""Auxiliares do diagnóstico de log-probabilidades do RL (`scripts/diag_logprob.py`).

Três perguntas, todas sobre os mesmos tokens de respostas já geradas:
1. Descompasso vLLM x treinador: a log-probabilidade (temperatura 1) que o vLLM dá a cada token da resposta contra a que o treinador (HF nativo
   + adaptador, BF16) calcula. É o que o TRL registra como `sampling/sampling_logp_difference`.
2. Ruído numérico do corte do GSPO: a razão de importância usa log-probabilidades "antigas" (passada sem gradiente sobre todas as respostas) e
   "novas" (minibatch de treino, outra composição de padding). Como o eps do GSPO é 3e-4 e 4e-4, ruído dessa ordem corta sequências sem que o
   modelo tenha mudado. Aqui: a mesma sequência calculada em duas composições de batch e a média por sequência da diferença (o `log s`).
3. Cabeça FP32 (`cast_lm_head_to_fp32` do TRL, receita ScaleRL): a mesma medição com a camada final em FP32.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

GSPO_EPS = (3e-4, 4e-4)


def pick_samples(rollouts: str, n: int, seed: int = 0, max_len: int = 3072, min_completion: int = 16) -> list[dict]:
    """Sorteio fixo (pela semente) de N respostas do arquivo de rollouts; o mesmo nos dois estágios do diagnóstico."""
    samples = [json.loads(line) for line in Path(rollouts).read_text().splitlines() if line.strip()]
    use = [s for s in samples if len(s["prompt_ids"]) + len(s["completion_ids"]) <= max_len and len(s["completion_ids"]) >= min_completion]
    random.Random(seed).shuffle(use)
    return use[:n]


def signature(samples: list[dict]) -> list[int]:
    """Impressão digital de cada amostra (para conferir que o vLLM e o HF avaliaram as mesmas sequências)."""
    return [hash((tuple(s["prompt_ids"]), tuple(s["completion_ids"]))) % (2**31) for s in samples]


def _pct(sorted_vals: list[float], q: float) -> float:
    """Percentil por posto mais próximo (conservador nas caudas, bom também com poucas sequências)."""
    return sorted_vals[min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))]


def summarize(diffs: list[list[float]]) -> dict:
    """`diffs[i]`: diferenças de log-probabilidade (a - b) por token da resposta i.
    Devolve o erro por token e, por sequência, a média da diferença (o `log s` do GSPO) e a fração acima dos eps do GSPO."""
    flat = [d for seq in diffs for d in seq]
    seq_mean = [sum(seq) / len(seq) for seq in diffs if seq]
    abs_seq = sorted(abs(x) for x in seq_mean)
    out = {
        "sequences": len(seq_mean),
        "tokens": len(flat),
        "mean_abs_token_diff": round(sum(abs(x) for x in flat) / max(1, len(flat)), 6),
        "mean_token_diff": round(sum(flat) / max(1, len(flat)), 6),
        "seq_mean_diff": round(sum(seq_mean) / max(1, len(seq_mean)), 6),
        "seq_abs_mean": round(sum(abs_seq) / max(1, len(abs_seq)), 6),
        "seq_abs_p50": round(_pct(abs_seq, 0.5), 6) if abs_seq else None,
        "seq_abs_p90": round(_pct(abs_seq, 0.9), 6) if abs_seq else None,
        "seq_abs_p99": round(_pct(abs_seq, 0.99), 6) if abs_seq else None,
    }
    for eps in GSPO_EPS:
        out[f"frac_seq_abs_gt_{eps:g}"] = round(sum(x > eps for x in abs_seq) / max(1, len(abs_seq)), 4)
    return out


def diff_lists(a: dict[int, list[float]], b: dict[int, list[float]]) -> list[list[float]]:
    """Diferença token a token (a - b) para as sequências presentes nos dois dicionários (mesmo índice e mesmo comprimento)."""
    out = []
    for i in sorted(a):
        assert len(a[i]) == len(b[i]), f"comprimentos diferentes na sequência {i}: {len(a[i])} e {len(b[i])}"
        out.append([x - y for x, y in zip(a[i], b[i])])
    return out


def shuffled_chunks(n: int, chunk: int, seed: int) -> list[list[int]]:
    """Composição "de treino": índices embaralhados e repartidos em minibatches de `chunk` sequências."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return [order[i: i + chunk] for i in range(0, n, chunk)]


def fp32_head(inner):
    """Troca `inner.lm_head` por uma versão que calcula em FP32, fora do autocast (como o `cast_lm_head_to_fp32` do TRL).
    Exige embeddings não amarrados (o Lumi tem `tie_word_embeddings=False`). Devolve o módulo original, para restaurar."""
    import torch

    original = inner.lm_head

    class _FP32Head(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.weight32 = head.weight.detach().float()

        def forward(self, x):
            with torch.autocast(device_type=x.device.type, enabled=False):
                return torch.nn.functional.linear(x.float(), self.weight32)

    inner.lm_head = _FP32Head(original)
    return original

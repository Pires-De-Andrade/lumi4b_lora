"""GSM8K-PT: itens, prompt e pontuação (A-04, A-05, A-06, A-08).

Zero-shot, com a linha final `#### <número>` pedida no prompt (A-06). A nota é o `strict-match`; o
`flexible-extract` e a leitura PT-BR são registrados ao lado, sem mudar a nota.
"""

from __future__ import annotations

import random
from collections import Counter

from lumi_ft.eval import parse as P
from lumi_ft.eval.shape import length_stats, trace_language

DATASET = "Polygl0t/gsm8k-pt"
N_TEST = 1295

INSTRUCTION = (
    "Resolva o problema abaixo. Mostre o raciocínio e termine a resposta com uma única linha no formato "
    "`#### <número>`, em que <número> é somente o resultado final, sem unidades nem símbolos."
)


def build_prompt(question: str) -> str:
    return f"{INSTRUCTION}\n\nProblema: {question}"


def sample_ids(n: int | None, seed: int = 0, total: int = N_TEST) -> list[int]:
    """Índices do `test`: todos, ou uma amostra fixa por semente (ordem crescente)."""
    if n is None or n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def load_items(n: int | None, seed: int = 0) -> tuple[list[dict], str]:
    """`[{id, question, answer}]` e o commit do dataset no cache local."""
    import glob
    import os

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    assert len(ds) == N_TEST, len(ds)
    snap = sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", ""), "hub/datasets--Polygl0t--gsm8k-pt/snapshots/*")))
    rev = os.path.basename(snap[-1]) if snap else "desconhecido"
    return [{"id": i, "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in sample_ids(n, seed)], rev


def score_row(gen: dict, target_answer: str, *, thinking: bool) -> dict:
    """Pontua uma saída. `gen` é uma linha de `run_batches` (`text`, `n_gen`, `finished`)."""
    target = P.gsm8k_target(target_answer)
    ex = P.extract_number(gen["text"], thinking=thinking)
    t = ex.parsed.thought
    return {
        "id": gen["id"],
        "status": ex.status,
        "strict": ex.strict,
        "flexible": ex.flexible,
        "target": target,
        "strict_ok": P.gsm8k_correct(ex.strict, target),
        "flexible_ok": P.gsm8k_correct(ex.flexible, target),
        "strict_ptbr_ok": P.gsm8k_correct_ptbr(ex.strict, target),
        "finished": gen["finished"],
        "n_gen": gen["n_gen"],
        "closed": ex.parsed.closed,
        "n_close": ex.parsed.n_close,
        "answer_nonempty": bool(ex.parsed.answer),
        "trace_nonempty": bool(t),
        "trace_language": trace_language(t) if t else None,
        "trace_last_number_ok": P.gsm8k_correct(ex.trace_last_number(), target) if t else None,
    }


def _rate(xs: list[bool | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(rows: list[dict], batches: list[dict], *, thinking: bool) -> dict:
    """Notas, taxas de erro de formato (A-05), forma da saída (A-08) e custo dos lotes."""
    n = len(rows)
    out = {
        "n": n,
        "thinking": thinking,
        "strict_acc": _rate([r["strict_ok"] for r in rows]),
        "flexible_acc": _rate([r["flexible_ok"] for r in rows]),
        "strict_ptbr_acc": _rate([r["strict_ptbr_ok"] for r in rows]),
        "status_rates": P.rates(r["status"] for r in rows),
        "truncated_rate": _rate([not r["finished"] for r in rows]),
        "answer_nonempty_rate": _rate([r["answer_nonempty"] for r in rows]),
        "n_gen": length_stats([r["n_gen"] for r in rows]),
        "acc_when_truncated": _rate([r["strict_ok"] for r in rows if not r["finished"]]),
        "acc_when_finished": _rate([r["strict_ok"] for r in rows if r["finished"]]),
    }
    if thinking:
        closed = [r for r in rows if r["closed"]]
        out |= {
            "closed_rate": _rate([r["closed"] for r in rows]),
            "multiple_close_rate": _rate([r["n_close"] > 1 for r in rows]),
            "trace_nonempty_rate": _rate([r["trace_nonempty"] for r in rows]),
            "trace_language": dict(Counter(r["trace_language"] for r in rows if r["trace_language"])),
            "trace_last_number_is_answer_rate": _rate([r["trace_last_number_ok"] for r in closed]),
        }
    # Custo: o lote roda até a saída mais longa dele terminar.
    useful = sum(r["n_gen"] for r in rows)
    steps = sum(b["size"] * b["gen_len"] for b in batches)
    sec = sum(b["seconds"] for b in batches)
    out["batches"] = {
        "n": len(batches),
        "gen_len": length_stats([b["gen_len"] for b in batches]),
        "share_with_truncated_item": round(
            sum(any(not it["finished"] for it in b["items"]) for b in batches) / len(batches), 3
        ) if batches else None,
        "useful_token_share": round(useful / steps, 3) if steps else None,
        "seconds_total": round(sec, 1),
        "seconds_per_batch_mean": round(sec / len(batches), 1) if batches else None,
        "useful_tokens_per_s": round(useful / sec) if sec else None,
    }
    return out

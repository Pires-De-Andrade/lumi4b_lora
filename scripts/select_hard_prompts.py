"""Seleciona, de um pool de prompts de RL, os que **têm sinal** para o modelo: os que ele acerta de vez em quando (nem sempre, nem nunca).

Lê as respostas geradas por `scripts/grpo_rollout.py` (um grupo de `--group` respostas por prompt, recompensa 1 só se a resposta terminou, fechou um único `</think>`,
tem traço e resposta, acertou no formato estrito e raciocinou em português) e conta, por prompt, quantas valem 1. Um prompt tem sinal se 1 <= k <= grupo - 1: é exatamente
o grupo que o RL usa (a vantagem de um grupo todo igual é zero e ele não treina nada). Os sempre certos e os sempre errados saem.

  PYTHONPATH=src python scripts/select_hard_prompts.py --rollouts runs/rl_t6/select/rollouts.jsonl --pool data/grpo_prompts_t6.jsonl --out data/grpo_prompts_t6_hard.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path


def pass_counts(rollouts: Path) -> dict[str, list[int]]:
    """prompt_id -> [respostas no grupo, respostas com recompensa 1, respostas truncadas]."""
    out: dict[str, list[int]] = {}
    with rollouts.open() as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            c = out.setdefault(s["prompt_id"], [0, 0, 0])
            c[0] += 1
            c[1] += int(s["reward"] == 1.0)
            c[2] += int(not s["finished"])
    return out


def select(pool: list[dict], counts: dict[str, list[int]], group: int) -> tuple[list[dict], dict]:
    """Os prompts do pool que têm 1 <= k <= grupo - 1 acertos (ordem do pool; só os que foram gerados com o grupo completo) e o resumo por tipo."""
    kept, dist = [], collections.defaultdict(collections.Counter)
    for p in pool:
        c = counts.get(p["id"])
        if not c or c[0] != group:
            continue
        dist[p["kind"]][c[1]] += 1
        if 1 <= c[1] <= group - 1:
            kept.append({**p, "t6_correct_of_group": c[1], "t6_truncated_of_group": c[2]})
    resumo = {}
    for kind, d in dist.items():
        n = sum(d.values())
        resumo[kind] = {"prompts_gerados": n, "com_sinal": sum(v for k, v in d.items() if 1 <= k <= group - 1), "sempre_certos": d.get(group, 0), "sempre_errados": d.get(0, 0),
                        "acertos_por_grupo": {str(k): d[k] for k in sorted(d)}}
    return kept, resumo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--pool", default="data/grpo_prompts_t6.jsonl")
    ap.add_argument("--out", default="data/grpo_prompts_t6_hard.jsonl")
    ap.add_argument("--group", type=int, default=8)
    a = ap.parse_args()
    pool = [json.loads(line) for line in Path(a.pool).read_text().splitlines() if line.strip()]
    counts = pass_counts(Path(a.rollouts))
    kept, resumo = select(pool, counts, a.group)
    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
    rel = {"rollouts": a.rollouts, "pool": a.pool, "grupo": a.group, "prompts_gerados": len(counts), "selecionados": len(kept),
           "por_tipo": dict(collections.Counter(p["kind"] for p in kept)), "resumo": resumo, "saida": a.out}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rel, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

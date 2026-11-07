"""Compara duas avaliações com bootstrap pareado por item (A-07): diferença de nota e IC 95%.

Cada lado é `nome:modo` (por exemplo `viab:thinking`), lido de runs/eval/<nome>/<tarefa>_<modo>_scores.json.
Uso: python scripts/compare_runs.py --a viab:thinking --b base:nothinking --b base:thinking --tasks gsm8k_val,mcq_val
Também imprime a forma da saída do lado A (fecha </think>, idioma do traço, truncamento).
"""
import argparse
import json
import random
from pathlib import Path

N_BOOT = 10_000


def load(spec: str, task: str, root: str) -> dict:
    name, mode = spec.split(":")
    p = Path(root) / name / f"{task}_{mode}_scores.json"
    return {r["id"]: r for r in json.loads(p.read_text())}


def paired_bootstrap(a: list[int], b: list[int], seed: int = 0) -> tuple[float, float, float]:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(d) / n, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="nome:modo do candidato")
    ap.add_argument("--b", action="append", required=True, help="nome:modo de referência (pode repetir)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--root", default="runs/eval")
    ap.add_argument("--key", default="strict_ok", choices=["strict_ok", "flexible_ok"])
    ap.add_argument("--only-ids", default="", help="json com a lista de ids a considerar (ex.: data/val_gsm8k_clean_ids.json); vale só para tarefas gsm8k_val")
    a = ap.parse_args()
    lines = [f"# {a.a} contra {', '.join(a.b)} (bootstrap pareado, {N_BOOT} reamostras, IC 95%, nota = {a.key})\n",
             "| Tarefa | n | " + a.a + " | referência | referência (nota) | diferença | IC 95% | IC acima de 0 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for task in a.tasks.split(","):
        A = load(a.a, task, a.root)
        for spec in a.b:
            B = load(spec, task, a.root)
            ids = sorted(set(A) & set(B))
            assert len(ids) == len(A) == len(B), f"{task}: ids diferentes entre {a.a} e {spec}"
            if a.only_ids and task == "gsm8k_val":
                keep = set(json.loads(Path(a.only_ids).read_text()))
                ids = [i for i in ids if i in keep]
            xa = [int(A[i][a.key]) for i in ids]
            xb = [int(B[i][a.key]) for i in ids]
            diff, lo, hi = paired_bootstrap(xa, xb)
            lines.append(f"| {task} | {len(ids)} | {100 * sum(xa) / len(xa):.1f}% | {spec} | {100 * sum(xb) / len(xb):.1f}% | "
                         f"{100 * diff:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {'sim' if lo > 0 else 'não'} |")
    lines.append("\n## Forma da saída de " + a.a + "\n")
    lines.append("| Tarefa | fecha </think> e responde | traço em português (heurística) | truncadas | mediana de tokens |")
    lines.append("| --- | --- | --- | --- | --- |")
    for task in a.tasks.split(","):
        rows = list(load(a.a, task, a.root).values())
        n = len(rows)
        closed = sum(r["closed"] and r["answer_nonempty"] for r in rows) / n
        langs = [r["trace_language"] for r in rows if r["trace_language"]]
        pt = (langs.count("pt") / len(langs)) if langs else float("nan")
        unk = (langs.count("unk") / len(langs)) if langs else float("nan")
        trunc = sum(not r["finished"] for r in rows) / n
        med = sorted(r["n_gen"] for r in rows)[n // 2]
        lines.append(f"| {task} | {100 * closed:.1f}% | pt {100 * pt:.1f}% (indeterminado {100 * unk:.1f}%) | {100 * trunc:.1f}% | {med} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

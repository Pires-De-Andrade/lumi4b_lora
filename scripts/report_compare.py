"""Tabela de comparação entre vários sistemas nas mesmas provas: nota, diferença contra uma coluna de referência (bootstrap pareado
por item, IC 95%) e a forma da saída (fecha `</think>`, idioma do traço, truncadas, mediana de tokens).

Cada coluna é `Rótulo=nome:modo`, lida de runs/eval/<nome>/<prova>_<modo>_scores.json. Colunas sem resultado aparecem como "(pendente)".
Exemplo (os quatro sistemas do plano):
  python scripts/report_compare.py --ref Lumi \\
    --col "Nemotron 3 Nano 4B=nemotron4b:thinking" --col "Lumi=base:thinking" --col "Nosso Lumi (raciocínio em inglês)=NOME_EN:thinking" \\
    --col "Nosso Lumi (raciocínio em português)=NOME_PT:thinking" --tasks gsm8k_val,mcq_val --only-ids data/val_gsm8k_clean_ids.json
"""
import argparse
import json
import statistics
from pathlib import Path

from compare_runs import load, paired_bootstrap, paired_bootstrap_mean  # noqa: E402  (mesma pasta)

NAMES = {"gsm8k_val": "GSM8K (validação)", "mcq_val": "Múltipla escolha (validação)", "gsm8k_test": "GSM8K-PT (teste)", "mmlu_prox_test": "MMLU-ProX PT",
         "enem": "ENEM", "bluex": "BLUEX", "oab": "OAB",
         "oab_novas": "OAB 2018–2025", "enem_2024": "ENEM 2024", "mmlu_pt_exatas": "MMLU-PT exatas", "mmlu_pt_enade": "MMLU-PT ENADE",
         "mmlu_pt_concursos": "MMLU-PT concursos", "mmlu_prox_diag_pt": "MMLU-ProX PT (1.200)"}


def fmt(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", action="append", required=True, help="Rótulo=nome:modo (pode repetir; a ordem é a das colunas)")
    ap.add_argument("--ref", required=True, help="rótulo da coluna de referência")
    ap.add_argument("--tasks", default="gsm8k_val,mcq_val")
    ap.add_argument("--root", default="runs/eval")
    ap.add_argument("--key", default="strict_ok", choices=["strict_ok", "flexible_ok"])
    ap.add_argument("--only-ids", default="", help="json com ids a considerar nas tarefas gsm8k_val (validação limpa)")
    ap.add_argument("--subset-ids", default="", help="tarefa=arquivo.json,tarefa=arquivo.json: restringe a tarefa aos ids do arquivo (análise de sensibilidade)")
    ap.add_argument("--mean", action="store_true", help="acrescenta a coluna da nota média das provas de --tasks (a Nota de raciocínio, com as cinco da suíte)")
    a = ap.parse_args()
    cols = [tuple(c.split("=", 1)) for c in a.col]
    labels = [c[0] for c in cols]
    assert a.ref in labels, f"--ref {a.ref!r} não está entre as colunas {labels}"
    keep = set(json.loads(Path(a.only_ids).read_text())) if a.only_ids else None
    tasks = a.tasks.split(",")
    subsets = {kv.split("=", 1)[0]: set(json.loads(Path(kv.split("=", 1)[1]).read_text())) for kv in a.subset_ids.split(",") if kv}
    data: dict[tuple[str, str], dict | None] = {}
    for lab, spec in cols:
        for t in tasks:
            try:
                d = load(spec, t, a.root)
            except FileNotFoundError:
                d = None
            if d is not None and keep is not None and t == "gsm8k_val":
                d = {k: v for k, v in d.items() if k in keep}
            if d is not None and t in subsets:
                d = {k: v for k, v in d.items() if k in subsets[t]}
            data[(lab, t)] = d
    head = [NAMES.get(t, t) for t in tasks] + (["Nota média"] if a.mean else [])
    out = [f"# Comparação (nota = {a.key}; diferença contra **{a.ref}**, bootstrap pareado, IC 95%)\n",
           "| Sistema | " + " | ".join(head) + " |", "| --- | " + " | ".join("---" for _ in head) + " |"]
    for lab in labels:
        cells = []
        for t in tasks:
            d, r = data[(lab, t)], data[(a.ref, t)]
            if d is None:
                cells.append("(pendente)")
                continue
            ids = sorted(d)
            acc = sum(int(d[i][a.key]) for i in ids) / len(ids)
            if lab == a.ref or r is None or set(r) != set(d):
                cells.append(f"{fmt(acc)} (n={len(ids)})")
                continue
            diff, lo, hi = paired_bootstrap([int(d[i][a.key]) for i in ids], [int(r[i][a.key]) for i in ids])
            sig = " ▲" if lo > 0 else (" ▼" if hi < 0 else "")
            cells.append(f"{fmt(acc)} ({100 * diff:+.1f}; {100 * lo:+.1f} a {100 * hi:+.1f}){sig}".replace(".", ","))
        if a.mean:
            ds = [data[(lab, t)] for t in tasks]
            rs = [data[(a.ref, t)] for t in tasks]
            if any(d is None for d in ds):
                cells.append("(pendente)")
            else:
                acc = sum(sum(int(d[i][a.key]) for i in d) / len(d) for d in ds) / len(ds)
                if lab == a.ref or any(r is None or set(r) != set(d) for r, d in zip(rs, ds)):
                    cells.append(f"{fmt(acc)}")
                else:
                    ids = [sorted(d) for d in ds]
                    diff, lo, hi = paired_bootstrap_mean([([int(d[i][a.key]) for i in ii], [int(r[i][a.key]) for i in ii]) for d, r, ii in zip(ds, rs, ids)])
                    sig = " ▲" if lo > 0 else (" ▼" if hi < 0 else "")
                    cells.append(f"{fmt(acc)} ({100 * diff:+.1f}; {100 * lo:+.1f} a {100 * hi:+.1f}){sig}".replace(".", ","))
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("\n▲ ganho com IC acima de 0, ▼ perda com IC abaixo de 0, sem marca = empate estatístico. Entre parênteses: diferença em pontos; IC 95%.\n")
    out.append("## Forma da saída\n")
    out.append("| Sistema | Prova | fecha `</think>` e responde | traço em português | traço em inglês | truncadas | mediana de tokens |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for lab, spec in cols:
        for t in tasks:
            d = data[(lab, t)]
            if d is None:
                continue
            rows = list(d.values())
            n = len(rows)
            langs = [r["trace_language"] for r in rows if r["trace_language"]]
            closed = fmt(sum(r["closed"] and r["answer_nonempty"] for r in rows) / n) if spec.endswith(":thinking") else "–"
            pt = fmt(langs.count("pt") / len(langs)) if langs else "–"
            en = fmt(langs.count("en") / len(langs)) if langs else "–"
            out.append(f"| {lab} | {NAMES.get(t, t)} | {closed} | {pt} | {en} | {fmt(sum(not r['finished'] for r in rows) / n)} | {sorted(r['n_gen'] for r in rows)[n // 2]} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()

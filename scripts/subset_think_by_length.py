"""Filtra um parquet de SFT pelo tamanho do raciocínio: mantém os exemplos com thinking cujo raciocínio (tokens entre
`<think>` e `</think>`) tem no máximo `--max-think-tokens`, e mantém todos os exemplos sem thinking como estão.

Serve para testar se a cauda longa dos traços do professor (ex.: Nemotron 30B, P90 de ~1.000 tokens contra ~490 do Qwen)
é o que ensina o modelo a não fechar o raciocínio (laços). Não altera nenhum exemplo; só descarta.

Uso: PYTHONPATH=src python scripts/subset_think_by_length.py --src data/thor_pt.parquet --out data/thor_pt_short.parquet --max-think-tokens 600
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value

THINK_OPEN, THINK_CLOSE = 12, 13  # ids de <think> e </think> no tokenizador do Lumi

FEATS = Features({"id": Value("string"), "mode": Value("string"), "group": Value("string"), "task_type": Value("string"),
                  "bucket": Value("int32"), "viab": Value("bool"), "n_tokens": Value("int32"), "n_prompt": Value("int32"),
                  "loss_end": Value("int32"), "input_ids": Sequence(Value("int32"))})


def think_tokens(ids) -> int | None:
    """Tokens entre o primeiro <think> e o </think> seguinte; None se algum dos dois não aparece."""
    ids = list(ids)
    try:
        a = ids.index(THINK_OPEN)
        b = ids.index(THINK_CLOSE, a)
    except ValueError:
        return None
    return b - a - 1


def subset_by_think_len(rows: list[dict], max_think_tokens: int) -> tuple[list[dict], dict]:
    """Devolve (linhas mantidas, contagens). Exemplos `think` sem os dois marcadores são descartados e contados."""
    kept, c = [], Counter()
    for r in rows:
        if r["mode"] != "think":
            kept.append(r)
            c["nothink_mantidos"] += 1
            continue
        n = think_tokens(r["input_ids"])
        if n is None:
            c["think_sem_marcadores"] += 1
        elif n > max_think_tokens:
            c["think_removidos_longos"] += 1
        else:
            kept.append(r)
            c["think_mantidos"] += 1
    return kept, dict(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-think-tokens", type=int, required=True)
    a = ap.parse_args()
    rows = pq.read_table(a.src).to_pylist()
    kept, counts = subset_by_think_len(rows, a.max_think_tokens)
    Dataset.from_dict({c: [r[c] for r in kept] for c in FEATS}, features=FEATS).to_parquet(a.out)
    g = Counter((r["mode"], r["group"]) for r in kept)
    toks = {f"{m}/{grp}": sum(r["n_tokens"] for r in kept if (r["mode"], r["group"]) == (m, grp)) for (m, grp) in g}
    manifest = {"counts": counts, "rows": {f"{m}/{grp}": n for (m, grp), n in g.items()}, "tokens": toks,
                "total_tokens": sum(toks.values()), "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                "args": vars(a)}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

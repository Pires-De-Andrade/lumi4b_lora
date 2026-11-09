"""Junta parquets de SFT com as mesmas colunas (saída de build_teacher_dataset.py) em um só e grava o manifest ao lado.

Confere que o esquema é igual e que os `id` não se repetem. Não embaralha: o SFT sorteia a ordem a cada época.
Uso: PYTHONPATH=src python scripts/concat_parquets.py --inputs data/train_adalberto.parquet data/thor4/prep/mcq_only.parquet --out data/qwen_mix.parquet
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def concat(tables: list[pa.Table]) -> pa.Table:
    """Concatena tabelas de esquema igual (ignorando metadados) e com `id` únicos."""
    tables = [t.replace_schema_metadata(None) for t in tables]
    for t in tables[1:]:
        if not t.schema.equals(tables[0].schema):
            raise ValueError(f"esquemas diferentes:\n{tables[0].schema}\n{t.schema}")
    out = pa.concat_tables(tables)
    ids = out.column("id").to_pylist()
    if len(set(ids)) != len(ids):
        dup = [k for k, n in Counter(ids).items() if n > 1][:3]
        raise ValueError(f"ids repetidos entre as entradas (ex.: {dup})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = concat([pq.read_table(p) for p in a.inputs])
    pq.write_table(out, a.out)
    small = out.select(["mode", "group", "n_tokens"]).to_pylist()
    c = Counter((r["mode"], r["group"]) for r in small)
    toks = {f"{m}/{g}": sum(r["n_tokens"] for r in small if (r["mode"], r["group"]) == (m, g)) for (m, g) in c}
    manifest = {"inputs": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a.inputs},
                "rows": {f"{m}/{g}": n for (m, g), n in c.items()}, "tokens": toks, "total_tokens": sum(toks.values()),
                "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"linhas": out.num_rows, **{k: manifest[k] for k in ("rows", "total_tokens")}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

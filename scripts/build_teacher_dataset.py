"""Monta um parquet de treino (mesmo esquema do train.parquet) a partir de registros de um professor.

Os problemas são divididos por hash em 70% para exemplos com raciocínio e 30% para exemplos sem raciocínio (só a
resposta), para o mesmo problema não aparecer nos dois. Às linhas do professor soma as linhas sem raciocínio do
GigaVerbo do subconjunto de viabilidade (`--base`), como no treino de viabilidade. Nunca trunca: acima de `--max-len` tokens (padrão 1.024), descarta.

Uso: PYTHONPATH=src python scripts/build_teacher_dataset.py --think data/adalberto_think.jsonl \
        --nothink data/adalberto_nothink.jsonl --out data/train_adalberto.parquet --max-think 10000 --max-nothink 3000
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value

from lumi_ft.data import gigaverbo as GV
from lumi_ft.data import teacher as T
from lumi_ft.modeling.load import load_tokenizer

FEATS = Features({"id": Value("string"), "mode": Value("string"), "group": Value("string"), "task_type": Value("string"),
                  "bucket": Value("int32"), "viab": Value("bool"), "n_tokens": Value("int32"), "n_prompt": Value("int32"),
                  "loss_end": Value("int32"), "input_ids": Sequence(Value("int32"))})


def read(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()] if p else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", required=True)
    ap.add_argument("--nothink", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-think", type=int, default=10_000)
    ap.add_argument("--max-nothink", type=int, default=3_000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=0, help="descarta exemplos com mais tokens que isso (0 = 1.024, o padrão dos dados)")
    ap.add_argument("--base", default="data/train.parquet", help="de onde vêm as linhas sem raciocínio do GigaVerbo (viab)")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disjoint-inputs", action="store_true",
                    help="os problemas de --think e --nothink já são conjuntos separados: não divide por hash (70/30)")
    a = ap.parse_args()
    tok = load_tokenizer()
    think, nothink = read(a.think), read(a.nothink)
    if not a.disjoint_inputs:
        think = [r for r in think if GV.bucket(r["problem_id"]) < 7000]  # 70% dos problemas
        nothink = [r for r in nothink if GV.bucket(r["problem_id"]) >= 7000]  # os outros 30%
    else:
        assert not ({r["problem_id"] for r in think} & {r["problem_id"] for r in nothink}), "problemas repetidos entre think e nothink"
    rows, stats = T.build_rows(tok, think, nothink, max_think=a.max_think, max_nothink=a.max_nothink, per_problem=a.per_problem, seed=a.seed, max_len=a.max_len)
    if not a.no_base:
        base = [r for r in pq.read_table(a.base).to_pylist() if r["viab"] and r["mode"] == "nothink"]
        rows += base
        stats["base_nothink_rows"] = len(base)
    Dataset.from_dict({c: [r[c] for r in rows] for c in FEATS}, features=FEATS).to_parquet(a.out)
    c = Counter((r["mode"], r["group"]) for r in rows)
    toks = {f"{m}/{g}": sum(r["n_tokens"] for r in rows if (r["mode"], r["group"]) == (m, g)) for (m, g) in c}
    manifest = {"stats": stats, "rows": {f"{m}/{g}": n for (m, g), n in c.items()}, "tokens": toks, "total_tokens": sum(toks.values()),
                "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), "args": vars(a)}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

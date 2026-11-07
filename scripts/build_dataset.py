"""Fase 1: monta data/train.parquet, data/valid.parquet e reports/data_manifest.json (D-01 a D-11).

Raciocínio: todo o subset `reasoning` depois dos filtros. Sem raciocínio: amostra de subsets do GigaVerbo-v2 até
25% dos tokens (general 12%, structured/system_prompts/rewriting/summarization 2% cada, math/retrieval 2,5% cada).
Cada exemplo guarda só `input_ids` e as posições da loss (`n_prompt`, `loss_end`); os labels saem daí.

Uso: scripts/detach.sh build-data PYTHONPATH=src python scripts/build_dataset.py
     (smoke: ... --limit 3000 --out runs/data/smoke)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, load_dataset

from lumi_ft.data import gigaverbo as GV
from lumi_ft.data.decontaminate import NgramIndex, build_index
from lumi_ft.modeling.load import REVISION, load_tokenizer

GV_ID = "Polygl0t/gigaverbo-v2-sft"
NONREASONING_SHARE = {"general": 0.12, "structured": 0.02, "system_prompts": 0.02, "rewriting": 0.02,
                      "summarization": 0.02, "math": 0.025, "retrieval": 0.025}  # fração do total de tokens
NONREASONING_FILES = {"general": ["general/train-00000-of-00003.parquet"], "structured": ["structured/train-00000-of-00001.parquet"],
                      "system_prompts": ["system_prompts/train-00000-of-00001.parquet"], "rewriting": ["rewriting/train-00000-of-00001.parquet"],
                      "summarization": ["summarization/train-00000-of-00001.parquet"], "math": ["math/train-00000-of-00001.parquet"],
                      "retrieval": ["retrieval/train-00000-of-00004.parquet"]}
# Provas de teste e de validação para descontaminar (o `train` do GSM8K-PT fica de fora: ele é a fonte da validação A-10).
DECONTAM = [("Polygl0t/gsm8k-pt", None, "test"), ("li-lab/MMLU-ProX", "pt", "test"), ("li-lab/MMLU-ProX", "pt", "validation"),
            ("eduagarcia/enem_challenge", None, None), ("nicholasKluge/BLUEX_without_images", None, None),
            ("eduagarcia/oab_exams", None, None), ("CohereLabs/Global-MMLU", "pt", "test"), ("CohereLabs/Global-MMLU", "pt", "dev"),
            ("openai/MMMLU", "PT_BR", "test"), ("CohereLabs/include-base-44", "Portuguese", "test"),
            ("CohereLabs/include-base-44", "Portuguese", "validation")]
STAGES = ["total", "single_turn", "trace_ok", "dedup", "no_residue", "last_number", "language_pt", "decontam", "length_ok"]


def snapshot_sha(repo_dir: str) -> str:
    snaps = sorted(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", repo_dir, "snapshots", "*")))
    return os.path.basename(snaps[-1]) if snaps else "desconhecido"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_decontam_index(n: int):
    idx, per_source = NgramIndex(n), {}
    for name, cfg, split in DECONTAM:
        try:
            ds = load_dataset(name, cfg, split=split) if split else load_dataset(name, cfg)
            parts = [ds] if split else list(ds.values())
            before = idx.n_texts
            for part in parts:
                sub = build_index(part, n)
                idx.hashes |= sub.hashes
                idx.n_texts += sub.n_texts
            per_source[f"{name}|{cfg}|{split}"] = {"items": idx.n_texts - before}
        except Exception as e:  # noqa: BLE001
            per_source[f"{name}|{cfg}|{split}"] = {"error": repr(e)[:200]}
            print("AVISO: fonte de descontaminação indisponível:", name, cfg, split, repr(e)[:120], flush=True)
    return idx, per_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=0, help="só as primeiras N linhas do `reasoning` (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument("--viab-reasoning", type=int, default=10_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path("runs/data").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = load_tokenizer()
    idx, decontam_sources = build_decontam_index(a.ngram)
    print(f"índice de descontaminação: {len(idx):,} {a.ngram}-gramas de {idx.n_texts:,} itens ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------------ raciocínio
    ds = load_dataset(GV_ID, "reasoning", split="train")
    if a.limit:
        ds = ds.select(range(a.limit))
    funnel: dict[str, Counter] = {s: Counter() for s in STAGES}
    removed_samples: dict[str, list] = defaultdict(list)

    def note_removed(stage: str, ex: dict) -> None:
        if len(removed_samples[stage]) < 15:
            removed_samples[stage].append({"id": ex["id"], "task_type": ex["task_type"], "user": ex["user"][:200],
                                           "thought": ex["thought"][:500]})

    cands = []
    for i, row in enumerate(ds):
        tt = row["task_type"]
        funnel["total"][tt] += 1
        m = GV.parse_messages(row["messages"])
        if m is None:
            continue
        funnel["single_turn"][tt] += 1
        sp = GV.split_trace(m["assistant"])
        if sp is None:
            continue
        funnel["trace_ok"][tt] += 1
        cands.append({"id": f"reasoning:{i}", "task_type": tt, "score": row["instruct_score"], "system": m["system"],
                      "user": m["user"], "thought": sp[0], "answer": sp[1], "raw": m["assistant"]})
    # D-06: prompts duplicados -> fica o de maior instruct_score (empate: o primeiro)
    best: dict[str, dict] = {}
    for ex in cands:
        k = GV.norm_prompt(ex["user"])
        if k not in best or ex["score"] > best[k]["score"]:
            best[k] = ex
    keep_ids = {ex["id"] for ex in best.values()}
    stage_rows = []
    for ex in cands:
        if ex["id"] in keep_ids:
            funnel["dedup"][ex["task_type"]] += 1
            stage_rows.append(ex)
    rows = []
    for ex in stage_rows:
        tt = ex["task_type"]
        if GV.has_residue(ex["thought"], ex["answer"]):
            note_removed("residue", ex)
            continue
        funnel["no_residue"][tt] += 1
        if tt == "math_reasoning" and not GV.last_trace_number_in_answer(ex["thought"], ex["answer"]):
            note_removed("last_number", ex)
            continue
        funnel["last_number"][tt] += 1
        if GV.trace_dropped_by_language(ex["thought"]):
            note_removed("language", ex)
            continue
        funnel["language_pt"][tt] += 1
        if idx.hits("\n".join([ex["system"] or "", ex["user"], ex["raw"]])):
            note_removed("decontam", ex)
            continue
        funnel["decontam"][tt] += 1
        rows.append(ex)
    print(f"raciocínio: {len(rows):,} de {sum(funnel['total'].values()):,} depois dos filtros de texto ({time.time() - t0:.0f} s)", flush=True)

    examples, render_fail = [], Counter()
    for ex in rows:
        try:
            r = GV.render_example(tok, ex["system"], ex["user"], ex["answer"], thought=ex["thought"])
        except AssertionError as e:
            render_fail[str(e)[:60]] += 1
            continue
        if r["n_tokens"] > GV.MAX_LEN:  # D-08: nunca truncar
            note_removed("length", ex)
            continue
        funnel["length_ok"][ex["task_type"]] += 1
        examples.append({"id": ex["id"], "mode": "think", "group": "reasoning", "task_type": ex["task_type"], **r})
    r_tokens = sum(e["n_tokens"] for e in examples)
    print(f"raciocínio renderizado: {len(examples):,} exemplos, {r_tokens / 1e6:.1f} M de tokens ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ sem raciocínio
    total_budget = r_tokens / 0.75
    nr_report = {}
    for cfg, share in NONREASONING_SHARE.items():
        budget = share * total_budget
        tables = [pq.read_table(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "datasets--Polygl0t--gigaverbo-v2-sft", "snapshots", "*", f))[0])
                  for f in NONREASONING_FILES[cfg]]
        table = pa.concat_tables(tables)
        score = np.asarray(table["instruct_score"].to_numpy(zero_copy_only=False))
        ntok = np.asarray(table["token_count"].to_numpy(zero_copy_only=False))
        cand_idx = [int(i) for i in np.flatnonzero((score >= GV.MIN_SCORE_NONREASONING) & (ntok <= GV.MAX_LEN - 60))]
        random.Random(a.seed + int(hashlib.sha256(cfg.encode()).hexdigest()[:6], 16)).shuffle(cand_idx)  # não usar hash(): varia por processo
        got, seen_prompts, stats = 0, set(), Counter()
        for i in cand_idx:
            if got >= budget:
                break
            row = table.slice(int(i), 1).to_pylist()[0]
            m = GV.parse_messages(row["messages"])
            if m is None:
                stats["multi_turn"] += 1
                continue
            if any(GV.has_think_tags(x) for x in (m["user"], m["assistant"], m["system"] or "")):  # D-04
                stats["think_tags"] += 1
                continue
            k = GV.norm_prompt(m["user"])
            if k in seen_prompts:
                stats["dup"] += 1
                continue
            seen_prompts.add(k)
            if idx.hits("\n".join([m["system"] or "", m["user"], m["assistant"]])):
                stats["decontam"] += 1
                continue
            try:
                r = GV.render_example(tok, m["system"], m["user"], m["assistant"])
            except AssertionError:
                stats["render"] += 1
                continue
            if r["n_tokens"] > GV.MAX_LEN:
                stats["length"] += 1
                continue
            examples.append({"id": f"{cfg}:{int(i)}", "mode": "nothink", "group": cfg, "task_type": row["task_type"], **r})
            got += r["n_tokens"]
        nr_report[cfg] = {"budget_tokens": int(budget), "got_tokens": int(got), "shortfall": got < budget,
                          "candidates": len(cand_idx), "rows_in_shard": len(table), **dict(stats)}
        print(f"sem raciocínio {cfg}: {got / 1e6:.2f} M de {budget / 1e6:.2f} M tokens", nr_report[cfg], flush=True)

    # --------------------------------------------- validação 2% (D-11) e flags de subconjunto
    for e in examples:
        e["bucket"] = GV.bucket(e["id"])
        e["viab"] = False
    strata = defaultdict(list)
    for e in examples:
        strata[(e["mode"], e["group"], e["task_type"])].append(e)
    valid_ids = set()
    for key, es in strata.items():
        es.sort(key=lambda e: GV.bucket(e["id"] + "|val"))
        valid_ids.update(e["id"] for e in es[: max(1, math.ceil(a.valid_frac * len(es)))])
    train = [e for e in examples if e["id"] not in valid_ids]
    valid = [e for e in examples if e["id"] in valid_ids]
    # viabilidade: 10.000 de raciocínio estratificados por task_type + sem raciocínio até 25% dos tokens
    by_tt = defaultdict(list)
    for e in train:
        if e["mode"] == "think":
            by_tt[e["task_type"]].append(e)
    n_think = sum(len(v) for v in by_tt.values())
    viab_tokens = 0
    for tt, es in by_tt.items():
        es.sort(key=lambda e: e["bucket"])
        for e in es[: round(a.viab_reasoning * len(es) / n_think)]:
            e["viab"] = True
            viab_tokens += e["n_tokens"]
    nonthink = sorted((e for e in train if e["mode"] == "nothink"), key=lambda e: e["bucket"])
    got = 0
    for e in nonthink:
        if got >= viab_tokens / 3:
            break
        e["viab"] = True
        got += e["n_tokens"]

    # ------------------------------------------------------------------ salvar
    feats = Features({"id": Value("string"), "mode": Value("string"), "group": Value("string"), "task_type": Value("string"),
                      "bucket": Value("int32"), "viab": Value("bool"), "n_tokens": Value("int32"), "n_prompt": Value("int32"),
                      "loss_end": Value("int32"), "input_ids": Sequence(Value("int32"))})
    cols = list(feats)
    paths = {}
    for name, rows_ in (("train", train), ("valid", valid)):
        Dataset.from_dict({c: [e[c] for e in rows_] for c in cols}, features=feats).to_parquet(out / f"{name}.parquet")
        paths[name] = out / f"{name}.parquet"

    def summ(rows_):
        c = Counter((e["mode"], e["group"]) for e in rows_)
        return {f"{m}/{g}": {"examples": n, "tokens": sum(e["n_tokens"] for e in rows_ if (e["mode"], e["group"]) == (m, g))}
                for (m, g), n in sorted(c.items())}

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "desconhecido"
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0), "git_commit": commit,
        "args": vars(a), "tokenizer_revision": REVISION, "gigaverbo_snapshot": snapshot_sha("datasets--Polygl0t--gigaverbo-v2-sft"),
        "filters": {"max_len": GV.MAX_LEN, "min_score_nonreasoning": GV.MIN_SCORE_NONREASONING, "residue_regex": GV.RESIDUE.pattern,
                    "language": f"shape.trace_language, descarta {sorted(GV.DROP_LANGUAGES)}", "ngram": a.ngram,
                    "dedupe": "prompt normalizado; fica o de maior instruct_score"},
        "decontam_sources": decontam_sources, "decontam_index": {"ngrams": len(idx), "items": idx.n_texts},
        "reasoning_funnel": {s: {"total": sum(c.values()), **dict(c)} for s, c in funnel.items()},
        "render_failures": dict(render_fail), "nonreasoning": nr_report,
        "train": summ(train), "valid": summ(valid),
        "viab": {"examples": sum(e["viab"] for e in train), "tokens": sum(e["n_tokens"] for e in train if e["viab"])},
        "sha256": {k: sha256_file(p) for k, p in paths.items()},
    }
    Path("reports").mkdir(exist_ok=True)
    mpath = Path("reports/data_manifest.json") if a.out == "data" else out / "data_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    Path("runs/data/removed_samples.json").write_text(json.dumps(removed_samples, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ("train", "valid", "viab", "reasoning_funnel")}, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())

"""Dados de um professor (outro modelo, ou o próprio Lumi) para o SFT: assinatura de problema, conversão de formato e
montagem das linhas de treino no mesmo esquema do `train.parquet`.

Registro de entrada: {id, problem_id, question, thought (só com raciocínio), answer, source, prompt (opcional)}. O prompt de
treino é o da avaliação, sem system, para o modelo aprender a seguir a instrução de formato: `gsm8k.build_prompt(question)`,
ou o campo `prompt` quando o registro o traz (múltipla escolha).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from decimal import Decimal

from lumi_ft.data import gigaverbo as GV
from lumi_ft.eval import gsm8k as K
from lumi_ft.eval import parse as P

FINAL_LINE = re.compile(r"^[ \t>*]*\**Resposta final\**[ \t]*:\**[ \t]*(.*?)[ \t*]*$", re.IGNORECASE | re.MULTILINE)


def number_signature(text: str) -> tuple[str, ...]:
    """Números do enunciado, normalizados (`1.000` = `1000`, `0,5` = `0.5`) e ordenados. Duas traduções do mesmo
    problema têm a mesma assinatura; serve para achar o mesmo problema com outras palavras."""
    out = []
    for n in re.findall(r"\d+(?:[.,]\d+)*", text):
        d = P.to_decimal_ptbr(n)
        out.append(format(d.normalize(), "f") if d is not None else n)
    return tuple(sorted(out))


def plain_number(s: str) -> str | None:
    """Número como texto simples (`70.000` -> `70000`, `3,5` -> `3.5`); None se não der para ler."""
    ln = P.last_number(s)
    d = P.to_decimal_ptbr(ln) if ln else None
    return format(d.normalize(), "f") if isinstance(d, Decimal) else None


def to_hash_format(answer: str, ref: str) -> str | None:
    """Troca a última linha `Resposta final: X` por `#### <ref>`. None se a linha não for a última ou não existir."""
    ms = list(FINAL_LINE.finditer(answer))
    if not ms:
        return None
    last = ms[-1]
    if answer[last.end():].strip():  # há texto depois da linha final: descarta
        return None
    return answer[: last.start()].rstrip() + f"\n\n#### {ref}"


def build_rows(tok, think_records: list[dict], nothink_records: list[dict], *, max_think: int = 0, max_nothink: int = 0,
               per_problem: int = 2, seed: int = 0, max_len: int = 0) -> tuple[list[dict], dict]:
    """Renderiza os registros no esquema do `train.parquet`. Devolve (linhas, contagens). Nunca trunca: o que passa de
    `max_len` tokens (0 = `GV.MAX_LEN`) é descartado e contado."""
    limit = max_len or GV.MAX_LEN
    stats = {"think_in": len(think_records), "nothink_in": len(nothink_records), "too_long": 0, "render_fail": 0, "max_len": limit}
    rng = random.Random(seed)
    rows: list[dict] = []

    def one(rec: dict, mode: str):
        user = rec.get("prompt") or K.build_prompt(rec["question"])  # registros de múltipla escolha trazem o prompt pronto
        try:
            r = (GV.render_example(tok, None, user, rec["answer"], thought=rec["thought"]) if mode == "think"
                 else GV.render_example(tok, None, user, rec["answer"]))
        except AssertionError:
            stats["render_fail"] += 1
            return None
        if r["n_tokens"] > limit:
            stats["too_long"] += 1
            return None
        return {"id": rec["id"], "mode": mode, "group": rec["source"], "task_type": "teacher", "bucket": GV.bucket(rec["id"]),
                "viab": True, "n_tokens": r["n_tokens"], "n_prompt": r["n_prompt"], "loss_end": r["loss_end"], "input_ids": r["input_ids"]}

    for recs, mode, cap in ((think_records, "think", max_think), (nothink_records, "nothink", max_nothink)):
        recs = list(recs)
        rng.shuffle(recs)
        per = defaultdict(int)
        n = 0
        for rec in recs:
            if per[rec["problem_id"]] >= per_problem:
                continue
            row = one(rec, mode)
            if row is None:
                continue
            per[rec["problem_id"]] += 1
            rows.append(row)
            n += 1
            if cap and n >= cap:
                break
        stats[f"{mode}_out"] = n
    return rows, stats


# ------------------------------------------------------------------ duplicatas da validação
def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-zà-ÿ]{4,}", t.lower()))


def load_val_signatures(path: str = "data/val_gsm8k.jsonl") -> dict[tuple, list[set]]:
    import json
    from pathlib import Path

    out: dict[tuple, list[set]] = {}
    for line in Path(path).read_text().splitlines():
        q = json.loads(line)["question"]
        s = number_signature(q)
        if len(s) >= 2:
            out.setdefault(s, []).append(_words(q))
    return out


def is_val_duplicate(question: str, val_by_sig: dict[tuple, list[set]], j_min: float = 0.3) -> bool:
    """Mesma assinatura de números e sobreposição de palavras >= j_min com algum item da validação (outra tradução)."""
    s = number_signature(question)
    if len(s) < 2 or s not in val_by_sig:
        return False
    w = _words(question)
    return any(len(w & c) / max(1, len(w | c)) >= j_min for c in val_by_sig[s])

"""Monta os prompts do pacote 4 da Thor (traços de múltipla escolha em português com o Qwen3-30B-A3B-Thinking, gabarito do dataset).

Parte das questões do RL com gabarito (`data/grpo_prompts_gold.jsonl`: já traduzidas, filtradas por 8-grama contra as provas de teste, a
validação e o MMLU-ProX, e por qualidade da tradução) e tira as 1.024 que o RL do `gspo_gold` usa em 8 ciclos (`data/audit_gold/used_ids.json`),
para o SFT e o RL não se sobreporem. Embaralha com semente fixa dentro de cada fonte e intercala as fontes, para que qualquer entrega parcial
(a geração é retomável e vem em blocos) seja equilibrada entre MedMCQA e WebInstruct.
Saída: handoff/thor4/data/prompts.jsonl ({id, source, kind: "mcq", category, prompt, ref}) + prompts.sha256 + prompts.report.json.
Uso: PYTHONPATH=src python scripts/make_thor4_prompts.py
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


def build(rows: list[dict], used: set[str], seed: int = 0) -> list[dict]:
    """Questões de múltipla escolha que não estão em `used`, embaralhadas por fonte e intercaladas (uma de cada fonte por vez)."""
    mcq = [r for r in rows if r["kind"] == "mcq" and r["id"] not in used]
    by_src: dict[str, list[dict]] = {}
    for r in sorted(mcq, key=lambda r: r["id"]):
        by_src.setdefault(str(r.get("source")), []).append(r)
    rnd = random.Random(seed)
    for lst in by_src.values():
        rnd.shuffle(lst)
    out, i = [], 0
    while any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src]:
                r = by_src[src].pop()
                out.append({"id": r["id"].removeprefix("mcq:"), "source": r["source"], "kind": "mcq", "category": r.get("category"),
                            "prompt": r["prompt"], "ref": r["gold"]})
        i += 1
    return out


def main() -> int:
    rows = [json.loads(line) for line in Path("data/grpo_prompts_gold.jsonl").read_text().splitlines() if line.strip()]
    used = set(json.loads(Path("data/audit_gold/used_ids.json").read_text())["mcq_ids"])
    out = build(rows, used)
    d = Path("handoff/thor4/data")
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    (d / "prompts.jsonl").write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    (d / "prompts.sha256").write_text(f"{sha}  prompts.jsonl\n")
    rep = {"mcq_no_gold": sum(r["kind"] == "mcq" for r in rows), "usadas_pelo_rl_e_tiradas": len(used & {r["id"] for r in rows}), "prompts": len(out),
           "por_fonte": dict(Counter(r["source"] for r in out)), "gabarito": dict(sorted(Counter(r["ref"] for r in out).items())), "sha256": sha}
    (d / "prompts.report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

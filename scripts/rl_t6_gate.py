"""Portão do RL a partir do T6-Lumi (H75): a validação do modelo final passa para a suíte de teste?

Regra (escrita em estudos/07, H75, antes de qualquer resultado): múltipla escolha com thinking >= 75,2% (a linha de troca de candidato da H65: o T6-Lumi, 73,25%, mais 2,0
pontos), fecha o `</think>` em >= 90% e traço em português em >= 90%. Lê runs/eval/<nome>/summary.json (chaves `mcq_val|thinking`). Sai com 0 se passa e 1 se não.

  python scripts/rl_t6_gate.py [--name rl_t6]
"""
import argparse
import json
import sys
from pathlib import Path

MIN_MCQ, MIN_CLOSED, MIN_PT = 0.752, 0.90, 0.90


def gate(summary: dict) -> dict:
    v = summary["mcq_val|thinking"]
    langs = v.get("trace_language", {})
    pt = langs.get("pt", 0) / max(1, sum(langs.values()))
    checks = {"mcq": (v["strict_acc"], v["strict_acc"] >= MIN_MCQ), "fecha": (v["closed_rate"], v["closed_rate"] >= MIN_CLOSED), "portugues": (pt, pt >= MIN_PT)}
    return {"passa": all(ok for _, ok in checks.values()), "n": v["n"], **{k: {"valor": round(x, 4), "ok": ok} for k, (x, ok) in checks.items()},
            "limiares": {"mcq": MIN_MCQ, "fecha": MIN_CLOSED, "portugues": MIN_PT}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rl_t6")
    a = ap.parse_args()
    try:
        res = gate(json.loads(Path(f"runs/eval/{a.name}/summary.json").read_text()))
    except (FileNotFoundError, KeyError) as e:
        print(f"sem validação de {a.name}: {e!r}")
        return 1
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["passa"] else 1


if __name__ == "__main__":
    sys.exit(main())

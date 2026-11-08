"""Resumo por ciclo de uma rodada de GRPO: recompensa, truncadas, respostas sem </think>, entropia, KL, corte e avaliações salvas.
Uso: PYTHONPATH=src python scripts/grpo_status.py grpo2 [grpo3 ...]
"""
import json
import sys
from pathlib import Path


def last_end(path: Path):
    if not path.exists():
        return None
    ev = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return next((e for e in reversed(ev) if e.get("event") == "end"), None)


def main() -> int:
    for name in sys.argv[1:]:
        root = Path("runs/grpo") / name
        print(f"== {name}" + ("  [PARADA: " + (root / "STOPPED").read_text().strip() + "]" if (root / "STOPPED").exists() else ""))
        print("ciclo | recompensa mcq/math | trunc mcq | sem </think> mcq/math | entropia todas/mcq/math | KL | corte | usadas")
        for d in sorted(root.glob("it[0-9][0-9][0-9]")):
            sp = d / "rollouts.stats.json"
            if not sp.exists():
                print(f"{d.name} | (gerando)")
                continue
            s = json.loads(sp.read_text()); w = s["whys"]
            n = {k: sum(v for kk, v in w.items() if kk.startswith(k + ":")) for k in ("mcq", "math")}
            tr = w.get("mcq:truncado", 0) / max(1, n["mcq"])
            nc = {k: w.get(f"{k}:sem_um_unico_close", 0) / max(1, n[k]) for k in n}
            e = last_end(d / "train_log.jsonl")
            ent = e["entropy"] if e and "entropy" in e else {}
            print(f"{d.name} | {s['reward_mean'].get('mcq')}/{s['reward_mean'].get('math')} | {tr:.2f} | {nc['mcq']:.3f}/{nc['math']:.3f} | "
                  f"{ent.get('all', '-')}/{ent.get('mcq', '-')}/{ent.get('math', '-')} | {e.get('kl_mean', '-') if e else '-'} | "
                  f"{e.get('clip_fraction', '-') if e else '-'} | {e['used'] if e else '(treinando)'}")
        for ev in sorted(Path("runs/eval").glob(f"{name}*")):
            sm = ev / "summary.json"
            if sm.exists():
                j = json.loads(sm.read_text())
                print(f"  avaliação {ev.name}:", {k: (round(v["strict_acc"], 3), "trunc", round(v["truncated_rate"], 3)) for k, v in j.items() if not k.startswith("_")})
    return 0


if __name__ == "__main__":
    sys.exit(main())

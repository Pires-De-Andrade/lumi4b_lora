"""Guarda do GRPO: sai com código 3 se, em algum tipo (math/mcq), a fração de respostas que terminaram sozinhas sem exatamente um
`</think>` passar de --max-noclose. Lê o `rollouts.stats.json` de scripts/grpo_rollout.py.
Uso: PYTHONPATH=src python scripts/grpo_guard.py runs/grpo/x/it003/rollouts.stats.json --max-noclose 0.05
"""
import argparse
import json
import sys
from pathlib import Path

from lumi_ft.train import grpo as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--max-noclose", type=float, required=True)
    a = ap.parse_args()
    share = R.noclose_share(json.loads(Path(a.stats).read_text()))
    print("respostas sem um único </think>:", {k: round(v, 4) for k, v in share.items()}, "limite", a.max_noclose, flush=True)
    return 3 if any(v > a.max_noclose for v in share.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

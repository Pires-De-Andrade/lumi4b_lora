"""Decomposição pareada da diferença de acerto entre um modelo e a referência, por quadrante de fechamento do `</think>`.

Para cada item (mesmo id nas duas avaliações) classifica em: os dois fecham, só a referência fecha, só o modelo fecha, nenhum fecha, e mostra quanto
cada quadrante contribui à diferença total (em pontos). Mostra onde o modelo ganha (itens que a referência não fecha) e onde perde (itens em
que os dois fecham: custo de raciocinar em português; itens que só a referência fecha: laços e truncamento do modelo).
  python scripts/decompose_paired.py --task mcq_val --ref base --names gspo1 gspo1_c4
  python scripts/decompose_paired.py --task gsm8k_val --ref base --names gspo1 --only-ids data/val_gsm8k_clean_ids.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(root: str, name: str, task: str, mode: str) -> dict[str, dict]:
    rows = json.loads((Path(root) / name / f"{task}_{mode}_scores.json").read_text())
    return {r["id"]: r for r in rows}


def decompose(ref: dict[str, dict], cur: dict[str, dict], ids: list[str], key: str = "strict_ok") -> dict:
    """Por quadrante: n, acerto de cada lado e contribuição (em pontos) à diferença total `cur - ref`."""
    n = len(ids)
    cnt, hit = Counter(), defaultdict(lambda: [0, 0])
    for i in ids:
        a, b = ref[i], cur[i]
        q = ("ref fecha" if a["closed"] else "ref não fecha", "modelo fecha" if b["closed"] else "modelo não fecha")
        cnt[q] += 1
        hit[q][0] += bool(a[key])
        hit[q][1] += bool(b[key])
    quads = {}
    for q, c in cnt.items():
        quads[q] = {"n": c, "share": c / n, "acc_ref": hit[q][0] / c, "acc_modelo": hit[q][1] / c, "contrib_pts": 100 * (hit[q][1] - hit[q][0]) / n}
    tot_ref, tot_cur = sum(bool(ref[i][key]) for i in ids), sum(bool(cur[i][key]) for i in ids)
    return {"n": n, "acc_ref": tot_ref / n, "acc_modelo": tot_cur / n, "diff_pts": 100 * (tot_cur - tot_ref) / n, "quadrantes": quads}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, help="mcq_val ou gsm8k_val")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--mode", default="thinking")
    ap.add_argument("--root", default="runs/eval")
    ap.add_argument("--only-ids", default=None, help="json com os ids a considerar (validação limpa do GSM8K)")
    a = ap.parse_args()
    ref = load(a.root, a.ref, a.task, a.mode)
    keep = set(json.loads(Path(a.only_ids).read_text())) if a.only_ids else None
    for name in a.names:
        cur = load(a.root, name, a.task, a.mode)
        ids = [i for i in ref if i in cur and (keep is None or i in keep)]
        d = decompose(ref, cur, ids)
        print(f"-- {name}: n={d['n']}  {a.ref} {d['acc_ref']:.1%}  {name} {d['acc_modelo']:.1%}  dif {d['diff_pts']:+.1f} pts")
        for q, v in sorted(d["quadrantes"].items()):
            print(f"   {q[0]} / {q[1]:<13s} n={v['n']:4d} ({v['share']:5.1%})  acerto {a.ref} {v['acc_ref']:5.1%}  acerto {name} {v['acc_modelo']:5.1%}  "
                  f"contribuição {v['contrib_pts']:+.1f} pts")


if __name__ == "__main__":
    main()

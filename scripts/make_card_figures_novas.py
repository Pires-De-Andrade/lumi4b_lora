"""Figures for the "new exams" section of the T6-Lumi model card (hf/t6-lumi/fig6..fig9_*.svg), in the same style as scripts/make_card_figures.py.

Unlike that script, the numbers are not typed in: they are computed here from the per-item scores in runs/eval, with the same functions as the study
reports (`compare_runs.paired_bootstrap`, `paired_bootstrap_mean`, `decompose_paired.decompose`), so the figures and `report_compare.py` agree by construction.
The numbers also go to a JSON file (--numbers) for the card tables.

  T6-Lumi and Lumi-4B-v0.2: same machine (the RTX 4090 runs, `thor_pt_antidoom_of` and `base`).
  Nemotron 3 Nano 4B: ran on a Jetson Thor (`nemotron4b_thor`), same protocol; it is drawn hatched and never compared with a paired interval.

Usage:  python scripts/make_card_figures_novas.py [--out hf/t6-lumi] [--lumi base] [--png-dir <dir>] [--numbers runs/jobs/card_novas_numeros.json]
Needs matplotlib and numpy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare_runs import N_BOOT, load, paired_bootstrap, paired_bootstrap_mean  # noqa: E402
from decompose_paired import decompose  # noqa: E402

_spec = importlib.util.spec_from_file_location("card_figures", HERE / "make_card_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

TASKS = {"oab_novas": "OAB 2018–2025", "enem_2024": "ENEM 2024", "mmlu_pt_exatas": "MMLU-PT exatas", "mmlu_pt_enade": "MMLU-PT ENADE",
         "mmlu_pt_concursos": "MMLU-PT concursos"}
MEAN = "Mean of the five"
T6, LT, LN, NEMO = "T6-Lumi (thinking)", "Lumi-4B-v0.2 (thinking)", "Lumi-4B-v0.2 (no thinking)", "Nemotron 3 Nano 4B (thinking)"
F.COLOR[NEMO] = "#eda100"  # slot 4 of the reference palette (yellow); drawn with a hatch because it comes from another machine


def series_spec(lumi: str, t6: str, nemo: str) -> dict[str, tuple[str, str]]:
    """Card name -> (folder in runs/eval, mode)."""
    return {T6: (t6, "thinking"), LT: (lumi, "thinking"), LN: (lumi, "nothinking"), NEMO: (nemo, "thinking")}


# ---------------------------------------------------------------------------------------------------------------------
# Computation (pure functions over {id: row}; tested in tests/test_card_figures_novas.py)
# ---------------------------------------------------------------------------------------------------------------------
def accuracy(rows: dict[str, dict]) -> float:
    return 100 * sum(bool(r["strict_ok"]) for r in rows.values()) / len(rows)


def closure_share(rows: dict[str, dict]) -> float:
    return 100 * sum(bool(r["closed"]) for r in rows.values()) / len(rows)


def quadrants(ref: dict[str, dict], cur: dict[str, dict]) -> tuple[float, float, float]:
    """Contribution to (cur - ref), in points, of the items (both close, only ref closes, only cur closes); the fourth quadrant adds 0."""
    q = decompose(ref, cur, sorted(ref))["quadrantes"]
    get = lambda a, b: q.get((a, b), {}).get("contrib_pts", 0.0)  # noqa: E731
    return get("ref fecha", "modelo fecha"), get("ref fecha", "modelo não fecha"), get("ref não fecha", "modelo fecha")


def compute(root: str, spec: dict[str, tuple[str, str]]) -> dict:
    rows = {m: {t: load(f"{name}:{mode}", t, root) for t in TASKS} for m, (name, mode) in spec.items()}
    for t in TASKS:  # same items in every evaluation, otherwise a pairing would be wrong
        ids = set(rows[T6][t])
        for m in spec:
            if set(rows[m][t]) != ids:
                raise SystemExit(f"{t}: {m} does not have the same items as {T6}")
    out: dict = {"n": {t: len(rows[T6][t]) for t in TASKS}, "acc": {}, "closure": {}, "diff": {}, "quadrants": {}}
    for m in spec:
        out["acc"][m] = {t: accuracy(rows[m][t]) for t in TASKS}
        out["acc"][m][MEAN] = sum(out["acc"][m][t] for t in TASKS) / len(TASKS)
        out["closure"][m] = {t: closure_share(rows[m][t]) for t in TASKS}
    for ref in (LT, LN):  # T6-Lumi minus Lumi, paired by item
        d, pairs = {}, []
        for t in TASKS:
            ids = sorted(rows[T6][t])
            a, b = [int(bool(rows[T6][t][i]["strict_ok"])) for i in ids], [int(bool(rows[ref][t][i]["strict_ok"])) for i in ids]
            e, lo, hi = paired_bootstrap(a, b)
            d[t] = (100 * e, 100 * lo, 100 * hi)
            pairs.append((a, b))
        e, lo, hi = paired_bootstrap_mean(pairs)
        d[MEAN] = (100 * e, 100 * lo, 100 * hi)
        out["diff"][ref] = d
    out["quadrants"] = {t: quadrants(rows[LT][t], rows[T6][t]) for t in TASKS}
    out["n_boot"] = N_BOOT
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------------
def _xtick_labels(n: dict[str, int], with_mean: bool = True) -> list[str]:
    labels = [f"{name}\n(n={n[t]:,})" for t, name in TASKS.items()]
    return labels + ([f"{MEAN}"] if with_mean else [])


def fig_benchmarks(path: Path, D: dict):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    F._style()
    models = [NEMO, LN, LT, T6]
    groups = list(TASKS) + [MEAN]
    values = {m: {**{t: D["acc"][m][t] for t in TASKS}, MEAN: D["acc"][m][MEAN]} for m in models}
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.2)
    xs = F._group_bars(ax, groups, models, values, width=0.2, label_size=7.6, gap_after=len(TASKS) - 1)
    for b in ax.containers[0]:  # Nemotron: another machine, so also a texture (not colour alone)
        b.set_hatch("////")
    ax.set_xticklabels(_xtick_labels(D["n"]), fontsize=9.3)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Accuracy (strict)")
    F._frame(ax)
    ax.axvline((xs[-2] + xs[-1]) / 2, color=F.AXIS, lw=0.8, ls=(0, (3, 3)))
    handles = [Patch(facecolor=F.COLOR[m], edgecolor=F.SURFACE, hatch="////" if m == NEMO else None) for m in models[::-1]]
    labels = [m + (" – Jetson Thor" if m == NEMO else "") for m in models[::-1]]
    F._legend(fig, handles, labels, ncol=4)
    F._foot(fig, "New exams. Zero-shot, greedy, at most 4,096 new tokens, strict accuracy. T6-Lumi and Lumi: same RTX 4090 (vLLM). "
                 "Nemotron (hatched): Jetson Thor, same protocol; a different machine.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_paired_difference(path: Path, D: dict):
    import matplotlib.pyplot as plt

    F._style()
    rows = list(TASKS) + [MEAN]
    names = {**TASKS, MEAN: MEAN}
    panels = [("vs Lumi-4B-v0.2 (thinking)", D["diff"][LT]), ("vs Lumi-4B-v0.2 (no thinking)", D["diff"][LN])]
    lo_all = min(v[1] for _, d in panels for v in d.values())
    hi_all = max(v[2] for _, d in panels for v in d.values())
    xmin, xmax = 5 * math.floor((lo_all - 1) / 5), 5 * math.ceil((hi_all + 1) / 5)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.8, top=0.84, bottom=0.2, wspace=0.8)
    y = list(range(len(rows)))[::-1]
    blue = F.COLOR[T6]
    for ax, (title, data) in zip(axes, panels):
        F._frame(ax, "x")
        ax.axvline(0, color=F.INK2, lw=1.0, zorder=2)
        for yi, r in zip(y, rows):
            est, lo, hi = data[r]
            summary = r == MEAN
            solid = lo > 0
            ax.plot([lo, hi], [yi, yi], color=blue, lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot([est], [yi], marker="D" if summary else "o", ms=8.5 if summary else 7.5, mfc=blue if solid else F.SURFACE, mec=blue, mew=1.8, zorder=4)
            ax.text(1.04, yi, f"{F.fmt(est, True)}  [{F.fmt(lo, True)}, {F.fmt(hi, True)}]", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=9.3, color=F.INK, fontweight="bold" if summary else "normal")
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(range(xmin, xmax + 1, 5))
        ax.set_xlabel("Difference in accuracy, percentage points")
        ax.set_title(title, fontsize=11, color=F.INK, pad=12, loc="left")
        ax.axhline(0.5, color=F.AXIS, lw=0.8, ls=(0, (3, 3)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[r] for r in rows])
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    F._foot(fig, f"New exams. T6-Lumi minus Lumi-4B-v0.2 (same RTX 4090), point estimate and 95% paired-bootstrap CI ({D['n_boot']:,} resamples of items). "
                 "Filled marker: CI above 0; hollow: CI includes 0.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_closure(path: Path, D: dict):
    import matplotlib.pyplot as plt

    F._style()
    models = [LT, T6]
    values = {m: dict(D["closure"][m]) for m in models}
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.17)
    F._group_bars(ax, list(TASKS), models, values, width=0.3, label_size=9)
    ax.set_xticklabels(list(TASKS.values()), fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Generations that close </think>")
    F._frame(ax)
    F._legend(fig, *F._swatches(models), ncol=2)
    F._foot(fig, "New exams. Share of generations that close the reasoning (</think>) inside the 4,096-token budget. Greedy decoding, same RTX 4090.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_decomposition(path: Path, D: dict):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    F._style()
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.84, bottom=0.2)
    rows = list(TASKS)
    y = list(range(len(rows)))[::-1]
    both_c, lumi_c, t6_c = F.COLOR["Both close"], F.COLOR[LT], F.COLOR[T6]
    for yi, r in zip(y, rows):
        both, only_lumi, only_t6 = D["quadrants"][r]
        # diverging stack: positive parts go right from 0 and negative parts go left from 0, "both close" first so it sits next to the axis
        # (on these exams it is positive on some and negative on others, unlike on the suite)
        right = left = 0.0
        for v, c in ((both, both_c), (only_lumi, lumi_c), (only_t6, t6_c)):
            if v >= 0:
                ax.barh(yi, v, left=right, height=0.46, color=c, edgecolor=F.SURFACE, linewidth=1.2, zorder=3)
                right += v
            else:
                ax.barh(yi, v, left=left, height=0.46, color=c, edgecolor=F.SURFACE, linewidth=1.2, zorder=3)
                left += v
        net = D["diff"][LT][r][0]
        ax.plot([net], [yi], marker="D", ms=7, mfc=F.INK, mec=F.SURFACE, mew=1.2, zorder=5)
        ax.text(right + 0.45, yi + 0.34, f"net {F.fmt(net, True)}", ha="left", va="center", fontsize=9, color=F.INK)
    ax.axvline(0, color=F.INK2, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([TASKS[r] for r in rows])
    top = max(sum(v for v in q if v > 0) for q in D["quadrants"].values())
    low = min(sum(v for v in q if v < 0) for q in D["quadrants"].values())
    ax.set_xlim(5 * math.floor((low - 1) / 5), 5 * math.ceil((top + 4) / 5))
    ax.set_xlabel("Contribution to the accuracy difference vs Lumi-4B-v0.2 (thinking), percentage points")
    F._frame(ax, "x")
    handles = [Patch(facecolor=t6_c, edgecolor=F.SURFACE), Patch(facecolor=both_c, edgecolor=F.SURFACE), Patch(facecolor=lumi_c, edgecolor=F.SURFACE),
               Line2D([], [], marker="D", ls="", mfc=F.INK, mec=F.SURFACE, ms=7)]
    F._legend(fig, handles, ["Only T6-Lumi closes", "Both close", "Only Lumi closes", "Net difference"], ncol=4)
    F._foot(fig, "New exams. Paired by item. A generation counts as closed if it emits </think> inside 4,096 tokens. Greedy decoding, same RTX 4090.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


FIGURES = {
    "fig6_new_exams.svg": fig_benchmarks,
    "fig7_new_exams_paired_difference.svg": fig_paired_difference,
    "fig8_new_exams_closure.svg": fig_closure,
    "fig9_new_exams_decomposition.svg": fig_decomposition,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="hf/t6-lumi")
    ap.add_argument("--root", default="runs/eval")
    ap.add_argument("--lumi", default="base", help="folder of the Lumi evaluation on the same machine as the T6-Lumi (both modes)")
    ap.add_argument("--t6", default="thor_pt_antidoom_of")
    ap.add_argument("--nemo", default="nemotron4b_thor")
    ap.add_argument("--numbers", default="runs/jobs/card_novas_numeros.json")
    ap.add_argument("--png-dir", default=None, help="also write PNG copies here (for a quick look; not part of the release)")
    a = ap.parse_args()
    D = compute(a.root, series_spec(a.lumi, a.t6, a.nemo))
    D["origem"] = {"lumi": a.lumi, "t6": a.t6, "nemotron": a.nemo}
    Path(a.numbers).parent.mkdir(parents=True, exist_ok=True)
    Path(a.numbers).write_text(json.dumps(D, ensure_ascii=False, indent=1))
    print("numbers ->", a.numbers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fn(out / name, D)
        print("wrote", out / name)
        if a.png_dir:
            Path(a.png_dir).mkdir(parents=True, exist_ok=True)
            fn(Path(a.png_dir) / (Path(name).stem + ".png"), D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

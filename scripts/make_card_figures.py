"""Figures for the T6-Lumi model card (hf/t6-lumi/*.svg).

Nothing is measured here: every number is copied from the study records and each block names its source
(discovery.md section 20.N, estudos/07 hypothesis H-N). tests/test_card_figures.py checks the internal consistency
(means, paired-difference sums, ordering), so a typo in a number is caught before the figures are regenerated.

Usage:  python scripts/make_card_figures.py [--out hf/t6-lumi] [--png-dir <dir>]
Needs only matplotlib. Text is converted to outlines (svg.fonttype = "path"), so the SVG looks the same on any machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------------
# Data. T6-Lumi = `thor_pt_antidoom_of`; Lumi = Lumi-4B v0.2 (`base`); both run on the same machine (RTX 4090), vLLM,
# greedy decoding, at most 4,096 new tokens, strict accuracy (discovery.md 20.66 and 20.68).
# ---------------------------------------------------------------------------------------------------------------------
EXAMS = ["GSM8K-PT", "MMLU-ProX PT", "ENEM", "BLUEX", "OAB Exams"]
N_ITEMS = {"GSM8K-PT": 1295, "MMLU-ProX PT": 11759, "ENEM": 1431, "BLUEX": 722, "OAB Exams": 2210}

# Accuracy (%). T6-Lumi GSM8K-PT is 1,170 of 1,295 (the record rounds it to 90.35; 90.3475 prints as 90.3).
ACC = {
    "T6-Lumi (thinking)": {"GSM8K-PT": 100 * 1170 / 1295, "MMLU-ProX PT": 54.49, "ENEM": 83.9, "BLUEX": 76.9, "OAB Exams": 44.0},
    "Lumi-4B-v0.2 (thinking)": {"GSM8K-PT": 88.73, "MMLU-ProX PT": 53.57, "ENEM": 80.22, "BLUEX": 72.02, "OAB Exams": 21.22},
    "Lumi-4B-v0.2 (no thinking)": {"GSM8K-PT": 85.33, "MMLU-ProX PT": 48.84, "ENEM": 78.83, "BLUEX": 71.19, "OAB Exams": 43.53},
}
REASONING_SCORE_RECORD = {"T6-Lumi (thinking)": 69.9, "Lumi-4B-v0.2 (thinking)": 63.2, "Lumi-4B-v0.2 (no thinking)": 65.5}  # 20.68

# Paired bootstrap (10,000 resamples of items), T6-Lumi minus Lumi, percentage points: (estimate, CI low, CI high). 20.68.
DIFF_VS_THINKING = {
    "GSM8K-PT": (1.6, 0.0, 3.2), "MMLU-ProX PT": (0.9, 0.1, 1.7), "ENEM": (3.7, 1.8, 5.7), "BLUEX": (4.8, 1.8, 7.8),
    "OAB Exams": (22.8, 20.6, 24.9), "Reasoning score": (6.8, 5.9, 7.6),
}
DIFF_VS_NO_THINKING = {
    "GSM8K-PT": (5.0, 3.3, 6.7), "MMLU-ProX PT": (5.6, 4.8, 6.5), "ENEM": (5.1, 3.0, 7.2), "BLUEX": (5.7, 2.2, 9.0),
    "OAB Exams": (0.5, -1.8, 2.6), "Reasoning score": (4.4, 3.4, 5.4),
}

# Share of generations that close `</think>` within the budget (20.66, 20.68; Lumi on the 4090).
CLOSURE = {
    "T6-Lumi (thinking)": {"GSM8K-PT": 99.8, "MMLU-ProX PT": 82.6, "ENEM": 97.1, "BLUEX": 95.7, "OAB Exams": 89.4},
    "Lumi-4B-v0.2 (thinking)": {"GSM8K-PT": 96.8, "MMLU-ProX PT": 73.4, "ENEM": 91.3, "BLUEX": 83.5, "OAB Exams": 39.2},
}

# Paired decomposition against Lumi (thinking): contribution to the accuracy difference, in points, by quadrant (20.68).
# (both close, only Lumi closes, only T6-Lumi closes); their sum is the difference in DIFF_VS_THINKING up to rounding.
QUADRANTS = {
    "GSM8K-PT": (-0.5, 0.0, 2.1), "MMLU-ProX PT": (-1.8, -3.4, 6.1), "ENEM": (-0.3, -0.6, 4.6),
    "BLUEX": (-1.1, -1.1, 7.1), "OAB Exams": (-0.3, -0.7, 23.8),
}

# Recipe stages on the clean validation sets (385 multiple-choice items, 676 GSM8K-PT items), thinking mode (20.64, estudos/02).
STAGES = ["Lumi-4B-v0.2", "+ SFT (thor_pt)", "+ Antidoom (T6-Lumi)"]
VALIDATION = {  # metric -> value per stage
    "Multiple-choice accuracy": (69.6, 54.3, 73.2),
    "GSM8K-PT accuracy": (81.7, 77.8, 85.8),
    "Closes </think>": (86.2, 66.0, 95.6),
    "Reasoning in Portuguese": (0.0, 99.6, 99.5),
}
# Held-out loop diagnostic: 500 multiple-choice prompts, greedy, 3,072 tokens (20.64). SFT model vs T6-Lumi.
LOOPS = {"Repetition loop": (48.6, 0.4), "Not finished": (53.0, 15.2)}

# Colours: dataviz reference palette, fixed order (blue, orange, aqua), validated; the colour follows the entity in every figure.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {
    "T6-Lumi (thinking)": "#2a78d6",
    "Lumi-4B-v0.2 (thinking)": "#eb6834",
    "Lumi-4B-v0.2 (no thinking)": "#1baf7a",
    "SFT only": "#898781",
    "Both close": "#898781",
}


def mean_score(model: str) -> float:
    """Reasoning score: unweighted mean of the five exam accuracies."""
    return sum(ACC[model][e] for e in EXAMS) / len(EXAMS)


def fmt(x: float, sign: bool = False) -> str:
    s = f"{x:+.1f}" if sign and round(x, 1) != 0 else f"{x:.1f}"
    return s.replace("-", "−")


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "svg.hashsalt": "t6-lumi",
    })
    return mpl


def _frame(ax, grid_axis: str = "y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _legend(fig, handles, labels, ncol, y=0.985):
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.1,
                     handleheight=1.1, columnspacing=1.8, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _foot(fig, text: str):
    fig.text(0.012, 0.012, text, fontsize=8, color=MUTED, ha="left", va="bottom")


def _swatches(models):
    from matplotlib.patches import Patch

    return [Patch(facecolor=COLOR[m], edgecolor=SURFACE) for m in models], list(models)


def _group_bars(ax, groups, series, values, width=0.26, label_size=8.3, gap_after=None):
    """Grouped columns. `values[series][group]`; `gap_after` inserts extra space after that group index."""
    xs = []
    x = 0.0
    for i, _ in enumerate(groups):
        xs.append(x)
        x += 1.0 + (0.55 if gap_after is not None and i == gap_after else 0.0)
    k = len(series)
    for j, s in enumerate(series):
        off = (j - (k - 1) / 2) * (width + 0.02)
        vals = [values[s][g] for g in groups]
        ax.bar([xg + off for xg in xs], vals, width=width, color=COLOR[s], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for xg, v in zip(xs, vals):
            ax.text(xg + off, v + 1.0, fmt(v), ha="center", va="bottom", fontsize=label_size, color=INK, zorder=4)
    ax.set_xticks(xs)
    return xs


def fig_benchmarks(path: Path):
    import matplotlib.pyplot as plt

    _style()
    models = ["Lumi-4B-v0.2 (no thinking)", "Lumi-4B-v0.2 (thinking)", "T6-Lumi (thinking)"]
    groups = EXAMS + ["Reasoning score"]
    values = {m: {**ACC[m], "Reasoning score": mean_score(m)} for m in models}
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2)
    xs = _group_bars(ax, groups, models, values, gap_after=len(EXAMS) - 1)
    ax.set_xticklabels(["GSM8K-PT\n(n=1,295)", "MMLU-ProX PT\n(n=11,759)", "ENEM\n(n=1,431)", "BLUEX\n(n=722)", "OAB Exams\n(n=2,210)",
                        "Reasoning score\n(mean of the five)"], fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Accuracy (strict)")
    _frame(ax)
    ax.axvline((xs[-2] + xs[-1]) / 2, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    _legend(fig, *_swatches(models[::-1]), ncol=3)
    _foot(fig, "Zero-shot, greedy decoding, at most 4,096 new tokens, strict accuracy (answer read only after </think>). Test sets; both models on the same RTX 4090 with vLLM.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_paired_difference(path: Path):
    import matplotlib.pyplot as plt

    _style()
    rows = EXAMS + ["Reasoning score"]
    panels = [("vs Lumi-4B-v0.2 (thinking)", DIFF_VS_THINKING), ("vs Lumi-4B-v0.2 (no thinking)", DIFF_VS_NO_THINKING)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    fig.subplots_adjust(left=0.145, right=0.815, top=0.84, bottom=0.2, wspace=0.78)
    y = list(range(len(rows)))[::-1]
    for ax, (title, data) in zip(axes, panels):
        _frame(ax, "x")
        ax.axvline(0, color=INK2, lw=1.0, zorder=2)
        for yi, r in zip(y, rows):
            est, lo, hi = data[r]
            summary = r == "Reasoning score"
            solid = lo > 0
            ax.plot([lo, hi], [yi, yi], color=COLOR["T6-Lumi (thinking)"], lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot([est], [yi], marker="D" if summary else "o", ms=8.5 if summary else 7.5, mfc=COLOR["T6-Lumi (thinking)"] if solid else SURFACE,
                    mec=COLOR["T6-Lumi (thinking)"], mew=1.8, zorder=4)
            ax.text(1.04, yi, f"{fmt(est, True)}  [{fmt(lo, True)}, {fmt(hi, True)}]", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=9.3, color=INK, fontweight="bold" if summary else "normal")
        ax.set_xlim(-4, 27)
        ax.set_xticks(range(-0, 26, 5))
        ax.set_xlabel("Difference in accuracy, percentage points")
        ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
        ax.axhline(0.5, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    _foot(fig, "T6-Lumi minus Lumi-4B-v0.2, point estimate and 95% paired-bootstrap CI (10,000 resamples of items). Filled marker: CI above 0; hollow: CI includes 0.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_closure(path: Path):
    import matplotlib.pyplot as plt

    _style()
    models = ["Lumi-4B-v0.2 (thinking)", "T6-Lumi (thinking)"]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.14)
    _group_bars(ax, EXAMS, models, CLOSURE, width=0.3, label_size=9)
    ax.set_xticklabels(EXAMS, fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("Generations that close </think>")
    _frame(ax)
    _legend(fig, *_swatches(models), ncol=2)
    _foot(fig, "Share of generations that close the reasoning (</think>) inside the 4,096-token budget. Test sets, greedy decoding, same RTX 4090.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_decomposition(path: Path):
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.2)
    rows = EXAMS
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        both, only_lumi, only_t6 = QUADRANTS[r]
        ax.barh(yi, only_t6, left=0, height=0.46, color=COLOR["T6-Lumi (thinking)"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.barh(yi, both, left=0, height=0.46, color=COLOR["Both close"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.barh(yi, only_lumi, left=both, height=0.46, color=COLOR["Lumi-4B-v0.2 (thinking)"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        net = DIFF_VS_THINKING[r][0]  # the paired-bootstrap estimate; the three rounded parts can add to 0.1 more or less
        ax.plot([net], [yi], marker="D", ms=7, mfc=INK, mec=SURFACE, mew=1.2, zorder=5)
        ax.text(max(only_t6, 0) + 0.45, yi + 0.34, f"net {fmt(net, True)}", ha="left", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlim(-6, 27)
    ax.set_xlabel("Contribution to the accuracy difference vs Lumi-4B-v0.2 (thinking), percentage points")
    _frame(ax, "x")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=COLOR["T6-Lumi (thinking)"], edgecolor=SURFACE), Patch(facecolor=COLOR["Both close"], edgecolor=SURFACE),
               Patch(facecolor=COLOR["Lumi-4B-v0.2 (thinking)"], edgecolor=SURFACE),
               Line2D([], [], marker="D", ls="", mfc=INK, mec=SURFACE, ms=7)]
    labels = ["Only T6-Lumi closes", "Both close", "Only Lumi closes", "Net difference"]
    _legend(fig, handles, labels, ncol=4)
    _foot(fig, "Paired by item. A generation counts as closed if it emits </think> inside 4,096 tokens. Test sets, greedy decoding, same RTX 4090.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


def fig_recipe(path: Path):
    import matplotlib.pyplot as plt

    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [4, 2.3], "wspace": 0.16})
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.17)
    stage_color = {STAGES[0]: COLOR["Lumi-4B-v0.2 (thinking)"], STAGES[1]: COLOR["SFT only"], STAGES[2]: COLOR["T6-Lumi (thinking)"]}

    def draw(ax, metrics, per_stage_idx, width):
        xs = list(range(len(metrics)))
        for j, st in enumerate(per_stage_idx):
            off = (j - (len(per_stage_idx) - 1) / 2) * (width + 0.02)
            vals = [metrics[m][j] for m in metrics]
            ax.bar([x + off for x in xs], vals, width=width, color=stage_color[st], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            for x, v in zip(xs, vals):
                ax.text(x + off, v + 1.0, fmt(v), ha="center", va="bottom", fontsize=8.3, color=INK, zorder=4)
        ax.set_xticks(xs)
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        _frame(ax)

    draw(a1, VALIDATION, STAGES, 0.26)
    a1.set_xticklabels(["Multiple-choice\naccuracy (n=385)", "GSM8K-PT\naccuracy (n=676)", "Closes\n</think>", "Reasoning in\nPortuguese"], fontsize=9.3)
    a1.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    a1.set_title("Validation, thinking mode (higher is better)", fontsize=10.5, loc="left", pad=10, color=INK)
    loops = {k: v for k, v in LOOPS.items()}
    draw(a2, loops, STAGES[1:], 0.3)
    a2.set_xticklabels(["Repetition\nloop", "Not finished in\n3,072 tokens"], fontsize=9.3)
    a2.set_yticklabels([])
    a2.set_title("Held-out prompts (lower is better)", fontsize=10.5, loc="left", pad=10, color=INK)
    from matplotlib.patches import Patch

    _legend(fig, [Patch(facecolor=stage_color[s], edgecolor=SURFACE) for s in STAGES], STAGES, ncol=3)
    _foot(fig, "Clean validation sets (385 multiple-choice, 676 GSM8K-PT items), greedy, 4,096 tokens. Right: 500 held-out multiple-choice prompts, greedy, 3,072 tokens, official loop detector.")
    fig.savefig(path, format=path.suffix[1:])
    plt.close(fig)


FIGURES = {
    "fig1_benchmarks.svg": fig_benchmarks,
    "fig2_paired_difference.svg": fig_paired_difference,
    "fig3_closure.svg": fig_closure,
    "fig4_decomposition.svg": fig_decomposition,
    "fig5_recipe.svg": fig_recipe,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="hf/t6-lumi")
    ap.add_argument("--png-dir", default=None, help="also write PNG copies here (for a quick look; not part of the release)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fn(out / name)
        print("wrote", out / name)
        if a.png_dir:
            Path(a.png_dir).mkdir(parents=True, exist_ok=True)
            fn(Path(a.png_dir) / (Path(name).stem + ".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

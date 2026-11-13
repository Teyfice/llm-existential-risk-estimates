"""
Five-panel figure from the mixed beta regressions in xrisk_models.R.

    A  Stated probability by developer region, next to AI researchers (2023 ESPAI)
    B  Stated probability vs. release date
    C  Stated probability vs. input price
    D  Within-model effect of requested reasoning level
    E  Primary cause by region

Each panel states its main result in one line under the title.

A: dots are run-averaged answers, one per model at its default setting, and the
answers of AI researchers to the same question (2023 ESPAI, general framing).
Boxes show the median and the middle 50%, whiskers the middle 90% (5th to 95th
percentile). Diamonds beside the boxes are simple means with 95% t-intervals.
Values at or below 0.1%, including answers of exactly 0%, are drawn at the
bottom of the axis.
B, C: black lines are model-implied values for a typical model (random effects at
0) from M1, other covariates at their means over cells, 95% CIs from the
fixed-effect covariance; dashed grey lines are the same for the upper and lower
bound of the 90% interval (M2, M3).
D: average marginal differences between reasoning levels (categorical term of
M1-M3) within models answered at two or more reasoning levels.

Usage (from the project folder, after Rscript modeling/xrisk_models.R):
    python modeling/figure.py [--out modeling/output/figure1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

OUT = Path(__file__).resolve().parent / "output"
ESPAI = (Path(__file__).resolve().parent.parent
         / "2023 ESPAI - Cleaned and Anonymized Responses - espai2023 2024-01-08 redacted and clean data.csv")
ESPAI_COLUMN = "extinction_basic_clean"   # same question as ours, answers in percent
P_FLOOR = 1e-3                    # values at or below are drawn at the floor ("<=0.1%")
LEVELS = ["none/off", "minimal", "low", "medium/on", "high", "xhigh", "max"]
REF_LEVEL = "medium/on"

# ---------------------------------------------------------------------------
# Style: full page width 17.8 cm; text 6-7 pt at final size
# ---------------------------------------------------------------------------
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
REGION_COLOR = {"US": "#2a78d6", "China": "#eb6834", "Other": "#1baf7a"}
HUMAN_COLOR = "#8a8984"
REGIONS = ["US", "China", "Other"]
CAUSES = ["rogue_ai", "ai_race", "organizational_risks", "malicious_use"]
CAUSE_LABEL = {"rogue_ai": "Rogue AI", "ai_race": "AI race",
               "organizational_risks": "Organizational", "malicious_use": "Malicious use"}
CAUSE_SHADE = {"rogue_ai": "#3b3a37", "ai_race": "#8a8984",
               "organizational_risks": "#c3c2bb", "malicious_use": "#e6e5e1"}
UB_COLOR = "#8a8984"
FINDING_FS = 6.0                  # one-line result under each title
TITLE_PAD = 13                    # points between axes top and title (room for the finding)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.edgecolor": INK3, "axes.linewidth": 0.5, "axes.labelcolor": INK2,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
PROB_TICKS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DOT_S, DOT_A = 5.5, 0.6
HUMANS_FIRST = True               # order of panel A: researchers first, then the three LLM regions


def load_researchers():
    x = pd.to_numeric(pd.read_csv(ESPAI, usecols=[ESPAI_COLUMN])[ESPAI_COLUMN], errors="coerce").dropna()
    return (x / 100).values


def pct(x, _=None):
    return "≤0.1%" if x <= P_FLOOR else f"{x*100:g}%"


def prob_axis(ax, label=True):
    ax.set_yscale("log")
    ax.set_ylim(P_FLOOR * 0.75, 1.15)
    ax.yaxis.set_major_locator(FixedLocator(PROB_TICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    if label:
        ax.set_ylabel("Stated probability")


def reference_line(ax, ref):
    ax.axhline(ref, color=INK3, lw=0.6, ls=(0, (1, 1.5)), zorder=1)


def heading(ax, letter, title, finding=None):
    """Panel letter and title, with the main result in one line directly below the title."""
    ax.set_title(title, loc="left", fontsize=7, color=INK, pad=TITLE_PAD)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-4, TITLE_PAD),
                textcoords="offset points", fontsize=9, fontweight="bold", ha="right", va="bottom", color=INK)
    if finding:
        ax.annotate(finding, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", fontsize=FINDING_FS, ha="left", va="bottom", color=INK2)


def mean_ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def strip(ax, x0, values, color, width, s, alpha, rng):
    """Dots spread sideways in proportion to their density (on the log scale)."""
    p = np.clip(values, P_FLOOR, None)
    lp = np.log10(p)
    dens = stats.gaussian_kde(lp, bw_method=0.2)(lp)
    jit = rng.uniform(-1, 1, len(p)) * width * dens / dens.max()
    ax.scatter(x0 + jit, p, s=s, color=color, alpha=alpha, linewidths=0.3 if s > 3 else 0,
               edgecolors="white", zorder=3)


def box_over_dots(ax, x0, values, width):
    """Quartile box, median line and whiskers at the 5th and 95th percentile, drawn over the dots."""
    p5, q1, med, q3, p95 = np.percentile(np.clip(values, P_FLOOR, None), [5, 25, 50, 75, 95])
    halo = [pe.withStroke(linewidth=1.7, foreground="white", alpha=0.85)]
    for a, b in [(p5, q1), (q3, p95)]:
        ax.plot([x0, x0], [a, b], color=INK, lw=0.7, zorder=6, path_effects=halo, solid_capstyle="butt")
    ax.add_patch(Rectangle((x0 - width / 2, q1), width, q3 - q1, fc="none", ec=INK, lw=0.7, zorder=6, path_effects=halo))
    ax.plot([x0 - width / 2, x0 + width / 2], [med, med], color=INK, lw=1.2, zorder=7, solid_capstyle="butt")


def scatter_by_region(ax, x, ms):
    for r in REGIONS:
        s = ms["region"] == r
        ax.scatter(x[s], ms.loc[s, "probability"].clip(lower=P_FLOOR), s=DOT_S,
                   color=REGION_COLOR[r], alpha=DOT_A, linewidths=0.3,
                   edgecolors="white", zorder=3)


def fit_lines(ax, xs, pred, varied):
    """Model-implied estimate (solid + CI band) and both bounds (dashed + band)."""
    out = {}
    for key, color, ls, lw in [("upper_bound", UB_COLOR, (0, (3, 1.5)), 1.0),
                               ("lower_bound", UB_COLOR, (0, (3, 1.5)), 1.0),
                               ("probability", INK, "-", 1.3)]:
        p = pred[(pred["outcome"] == key) & (pred["varied"] == varied)]
        ax.fill_between(xs, p["ci_low"], p["ci_high"], color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(xs, p["predicted"], color=color, lw=lw, ls=ls, zorder=4)
        out[key] = p["predicted"].values
    for key, txt, c, f, va in [("probability", "Estimate", INK, 1.15, "bottom"),
                               ("upper_bound", "Upper bound", INK2, 1.15, "bottom"),
                               ("lower_bound", "Lower bound", INK2, 0.85, "top")]:
        ax.text(xs[-1], out[key][-1] * f, txt, ha="right", va=va, fontsize=5.5, color=c, zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_models_vs_researchers(ax, ms, humans, humans_first=HUMANS_FIRST):
    rng = np.random.default_rng(7)
    groups = [(r, ms.loc[ms["region"] == r, "probability"].values, REGION_COLOR[r], 0.30, 4.5, 0.55)
              for r in REGIONS]
    human = ("AI researchers", humans, HUMAN_COLOR, 0.34, 1.6, 0.28)
    groups = [human] + groups if humans_first else groups + [human]
    labels = []
    for x0, (name, vals, color, width, s, alpha) in enumerate(groups):
        strip(ax, x0, vals, color, width, s, alpha, rng)
        box_over_dots(ax, x0, vals, 0.30)
        m, lo, hi = mean_ci(vals)
        ax.errorbar(x0 + 0.23, m, yerr=[[m - lo], [hi - m]], fmt="D", ms=3, color=INK, mfc=INK,
                    mec="white", mew=0.5, elinewidth=0.9, capsize=0, zorder=8)
        labels.append(f"{name}\nn = {len(vals):,}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.tick_params(axis="x", length=0)
    prob_axis(ax)
    handles = [Patch(fc="none", ec=INK, lw=0.7, label="Middle 50%"), Line2D([], [], color=INK, lw=1.2, label="Median"),
               Line2D([], [], color=INK, lw=0.7, label="Middle 90%"),
               Line2D([], [], ls="none", marker="D", ms=2.8, mfc=INK, mec="white", mew=0.3, label="Mean (95% CI)")]
    ax.legend(handles=handles, loc="upper right" if humans_first else "upper left", ncol=2, frameon=False,
              fontsize=5, handlelength=1.1, handletextpad=0.35, columnspacing=0.8, borderaxespad=0.15, labelspacing=0.25)
    heading(ax, "A", "LLMs and AI researchers", "LLMs cluster near 10%; researchers spread from 0% to 100%")


def panel_date(ax, ms, pred, ref):
    scatter_by_region(ax, ms["release_date"], ms)
    yrs = pred.loc[(pred["outcome"] == "probability") & (pred["varied"] == "release_years"), "value"].astype(float).values
    xs = pd.Timestamp("2025-01-01") + pd.to_timedelta(yrs * 365.25, unit="D")
    fit_lines(ax, xs, pred, "release_years")
    prob_axis(ax, label=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(xs[0] - pd.Timedelta(days=120), xs[-1] + pd.Timedelta(days=30))
    ax.set_xlabel("Release date")
    heading(ax, "B", "Release date", "Estimates stay level but spread less over time")


def panel_price(ax, ms, pred, ref):
    scatter_by_region(ax, ms["input_price"].clip(lower=1e-3), ms)
    lp = pred.loc[(pred["outcome"] == "probability") & (pred["varied"] == "log_price"), "value"].astype(float).values
    fit_lines(ax, 10 ** lp, pred, "log_price")
    prob_axis(ax, label=False)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10]))   # free models sit at the $0.001 floor
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "free" if v <= 0.001 else f"${v:g}"))
    ax.set_xlabel("Input price (USD per million tokens)")
    heading(ax, "C", "Input price", "Pricier models, often more capable, state higher risks")


def panel_effort(ax, lv):
    t = {col: lv[lv["outcome"] == col].set_index("level") for col in ["probability", "upper_bound", "lower_bound"]}
    levels = [l for l in LEVELS if l in t["probability"].index]
    x = np.arange(len(levels))
    off = 0.12
    ax.axhline(0, color=INK3, lw=0.5, zorder=1)
    style = {"upper_bound": dict(dx=-off, color=UB_COLOR, lw=1.0, marker="o", ms=3, mfc=UB_COLOR,
                                 mew=0.4, ls=(0, (3, 1.5)), z=3),
             "lower_bound": dict(dx=off, color=UB_COLOR, lw=1.0, marker="s", ms=2.6, mfc="white",
                                 mew=0.7, ls=(0, (3, 1.5)), z=3),
             "probability": dict(dx=0, color=INK, lw=1.2, marker="o", ms=3, mfc=INK, mew=0.4, ls="-", z=4)}
    for col, st in style.items():
        d = t[col].loc[levels]
        yerr = [d["diff_pp"] - d["diff_low"], d["diff_high"] - d["diff_pp"]]
        ax.errorbar(x + st["dx"], d["diff_pp"], yerr=yerr, color=st["color"], lw=st["lw"],
                    marker=st["marker"], ms=st["ms"], mfc=st["mfc"],
                    mec="white" if st["mfc"] != "white" else st["color"], mew=st["mew"],
                    elinewidth=0.6, capsize=0, ls=st["ls"], zorder=st["z"])
    last = x.max()
    y_est = t["probability"].loc[levels[-1], "diff_pp"] + 0.6
    y_up = max(t["upper_bound"].loc[levels[-1], "diff_pp"], y_est + 1.6)   # keep labels apart
    ax.text(last + 0.25, y_up, "Upper\nbound", va="center", fontsize=5.5, color=INK2)
    ax.text(last + 0.25, y_est, "Estimate", va="center", fontsize=5.5, color=INK)
    ax.text(last + 0.25, t["lower_bound"].loc[levels[-1], "diff_pp"] - 1.0, "Lower\nbound",
            va="center", fontsize=5.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.get_xticklabels()[levels.index(REF_LEVEL)].set_fontweight("bold")
    ax.set_xlim(x.min() - 0.4, last + 1.3)
    ax.set_xlabel("Reasoning level")
    ax.set_ylabel("Difference from medium/on\n(percentage points)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    heading(ax, "D", "Reasoning level", "More reasoning raises the upper bound; the estimate barely moves")


def panel_cause(ax, ms):
    """Share of models naming each cause; a model answering differently across
    runs counts fractionally (its share of runs)."""
    groups = [(r, ms[ms["region"] == r]) for r in REGIONS]
    for i, (name, g) in enumerate(groups):
        share = pd.Series({c: g[f"share_{c}"].mean() for c in CAUSES})
        left, y = 0.0, len(groups) - 1 - i
        for c in CAUSES:
            w = share[c]
            if w > 0:
                ax.barh(y, w - 0.004 if w > 0.01 else w, left=left, height=0.62,
                        color=CAUSE_SHADE[c], edgecolor="none", zorder=3)
                if w >= 0.08:
                    ax.text(left + 0.015, y, f"{w*100:.0f}%", va="center", ha="left",
                            fontsize=5.5, color="white" if c == "rogue_ai" else INK)
            left += w
        ax.text(-0.02, y, f"{name}\n(n = {len(g)})", ha="right", va="center", fontsize=6, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(groups) + 0.15)   # room for the cause legend above the bars
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xlabel("Share of answers naming this as the primary cause")
    handles = [mpl.patches.Patch(color=CAUSE_SHADE[c], label=CAUSE_LABEL[c]) for c in CAUSES]
    ax.legend(handles=handles, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.2,
              handletextpad=0.4, borderaxespad=0)
    heading(ax, "E", "Primary cause", "Rogue AI is named most often in every developer region")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "figure1"))
    a = ap.parse_args()

    ms = pd.read_csv(OUT / "model_level.csv", parse_dates=["release_date"])
    ms = ms[ms["region"].isin(REGIONS)]
    pred = pd.read_csv(OUT / "predictions.csv")
    lv = pd.read_csv(OUT / "reasoning_levels.csv")
    humans = load_researchers()
    ref = humans.mean()

    cm = 1 / 2.54
    fig = plt.figure(figsize=(17.8 * cm, 11.6 * cm))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.79], hspace=0.55,
                             left=0.068, right=0.975, top=0.925, bottom=0.105)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.10)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    axA = fig.add_subplot(top[0])
    axB = fig.add_subplot(top[1], sharey=axA)
    axC = fig.add_subplot(top[2], sharey=axA)
    axD = fig.add_subplot(bottom[0])
    axE = fig.add_subplot(bottom[1])

    panel_models_vs_researchers(axA, ms, humans)
    panel_date(axB, ms, pred, ref)
    panel_price(axC, ms, pred, ref)
    panel_effort(axD, lv)
    panel_cause(axE, ms)
    for ax in (axB, axC):
        plt.setp(ax.get_yticklabels(), visible=False)
    pos = axE.get_position()   # room for the region labels left of the bars
    axE.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])

    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=600 if ext == "png" else None)
    print(f"models plotted: {len(ms)} from {ms['developer'].nunique()} developers; "
          f"AI researchers: n = {len(humans)}, mean {ref*100:.1f}% -> {a.out}.pdf/.png")


if __name__ == "__main__":
    main()

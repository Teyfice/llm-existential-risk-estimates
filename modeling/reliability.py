"""
Reliability of the answers across the five runs (GO 1 to GO 5).

Unit: model x reasoning setting (cell), analysed models only (sample.csv), valid answers.
  - ICC(1): share of the variance between cells in a single run's answer (one-way random
    effects, ANOVA estimator for unequal run counts; cells with at least two valid runs,
    since a single run carries no within-cell variance); ICC(k): reliability of the mean over
    the k = 5 runs (Spearman-Brown, the averaged values that are analysed). Computed on the
    probability scale and, as an unreported check, on the logit scale (single-run answers can
    be exactly 0, so they are squeezed into (0, 1) for this check only).
  - mean pairwise Spearman correlation between runs
  - share of cells with identical answers in all five runs (estimate; cause; everything)
  - Fleiss' kappa for the primary cause (variable number of runs per cell)
95% CIs from 4,000 bootstrap resamples of models (all cells of a drawn model kept).

Usage (after Rscript modeling/xrisk_models.R):  python modeling/reliability.py
Writes modeling/output/reliability.csv.
"""
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "modeling" / "output"
OUTCOMES = ["probability", "upper_bound", "lower_bound"]
K = 5
B = 4000
rng = np.random.default_rng(20261108)

# ---------------------------------------------------------------- data: one row per cell
sample = pd.read_csv(OUT / "sample.csv")
keep = set(sample.loc[sample["in_sample"], "model"])
runs = sorted(glob.glob(str(ROOT / "data" / "go_00[1-5]" / "results.csv")))
assert len(runs) == K
calls = pd.concat([pd.read_csv(p, low_memory=False) for p in runs])
d = calls[(calls["outcome"] == "valid") & calls["model"].isin(keep)].copy()
d["cell"] = d["model"] + "|" + d["condition"]
n_all = len(d)
for col in OUTCOMES:
    z = d[col].clip(0, 1) * (n_all - 1) / n_all + 0.5 / n_all
    d[f"logit_{col}"] = np.log(z / (1 - z))

g = d.groupby("cell")
cells = pd.DataFrame({"model": g["model"].first(), "n": g.size()})
VARS = OUTCOMES + [f"logit_{c}" for c in OUTCOMES]
for v in VARS:
    cells[f"mean_{v}"] = g[v].mean()
    cells[f"ssw_{v}"] = g[v].apply(lambda x: ((x - x.mean()) ** 2).sum())
causes = pd.crosstab(d["cell"], d["cause"]).reindex(cells.index)
cells["p_agree"] = ((causes * (causes - 1)).sum(axis=1) / (cells["n"] * (cells["n"] - 1))).where(cells["n"] >= 2)
full = cells["n"] == K
cells["ident_p"] = g["probability"].nunique() == 1
cells["ident_cause"] = g["cause"].nunique() == 1
cells["ident_all"] = cells["ident_p"] & cells["ident_cause"] & (g["upper_bound"].nunique() == 1) & (g["lower_bound"].nunique() == 1)
wide = {c: d.pivot_table(index="cell", columns="go", values=c).reindex(cells.index).to_numpy() for c in OUTCOMES}
causes = causes.to_numpy()
model_idx = cells.groupby("model").indices          # model -> row positions
models = list(model_idx)


# ---------------------------------------------------------------- statistics from cell weights
def compute(w):
    """w: weight (number of times drawn) per cell row."""
    out = {}
    n = cells["n"].to_numpy()
    ok = (n >= 2) & (w > 0)
    wn, nn = w[ok], n[ok]
    N, a = (wn * nn).sum(), wn.sum()
    for v in VARS:
        m = cells[f"mean_{v}"].to_numpy()[ok]
        grand = (wn * nn * m).sum() / N
        msb = (wn * nn * (m - grand) ** 2).sum() / (a - 1)
        msw = (wn * cells[f"ssw_{v}"].to_numpy()[ok]).sum() / (N - a)
        n0 = (N - (wn * nn ** 2).sum() / N) / (a - 1)
        s2b = max((msb - msw) / n0, 0.0)
        tag = v.replace("logit_", "logit_")
        out[f"icc1_{tag}"] = s2b / (s2b + msw)
        out[f"icck_{tag}"] = s2b / (s2b + msw / K)
    rep = np.repeat(np.arange(len(w)), w.astype(int))
    for c in OUTCOMES:
        x = wide[c][rep]
        rs = []
        for i, j in combinations(range(K), 2):
            sel = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            rs.append(np.corrcoef(rankdata(x[sel, i]), rankdata(x[sel, j]))[0, 1])
        out[f"spearman_{c}"] = float(np.mean(rs))
    wf = w * full.to_numpy()
    for k in ["ident_p", "ident_cause", "ident_all"]:
        out[f"share_{k}"] = (wf * cells[k].to_numpy()).sum() / wf.sum()
    pa = cells["p_agree"].to_numpy()
    pbar = np.nansum(w * pa) / w[~np.isnan(pa)].sum()
    pj = (w[:, None] * causes).sum(axis=0) / (w[:, None] * causes).sum()
    pe = (pj ** 2).sum()
    out["kappa_cause"] = (pbar - pe) / (1 - pe)
    return out


est = compute(np.ones(len(cells)))
boot = []
for _ in range(B):
    w = np.zeros(len(cells))
    for m in rng.choice(len(models), size=len(models), replace=True):
        w[model_idx[models[m]]] += 1
    boot.append(compute(w))
boot = pd.DataFrame(boot)
res = pd.DataFrame({"measure": list(est), "estimate": list(est.values()),
                    "ci_low": [boot[k].quantile(0.025) for k in est],
                    "ci_high": [boot[k].quantile(0.975) for k in est]})
res["n_cells"] = len(cells)
res["n_cells_all_runs"] = int(full.sum())
res["n_cells_icc"] = int((cells["n"] >= 2).sum())   # cells entering the ICCs (see compute)
res["n_answers"] = n_all
res["n_models"] = len(models)
res.to_csv(OUT / "reliability.csv", index=False)
print(res.round(3).to_string(index=False))

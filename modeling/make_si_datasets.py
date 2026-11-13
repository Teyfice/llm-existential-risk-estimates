"""
Write the SI Datasets next to the SI Appendix (manuscript/SI_Appendix/).

  Dataset S1  Dataset_S1_models.csv           one row per catalogue model after automatic exclusions:
                                              inclusion decision, covariates, settings, default-setting answer
  Dataset S2  Dataset_S2_answers.csv          one row per model and reasoning setting queried:
                                              valid answers in the five runs, mean answers, causes named
  Dataset S3  Dataset_S3_model_estimates.xlsx all model estimates and derived quantities (one sheet each)

Raw replies, including the explanations, are in data/go_00N/ (GitHub), not here.
Run after Rscript modeling/xrisk_models.R and python modeling/reliability.py:
    python modeling/make_si_datasets.py
"""
import glob
import io
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "modeling" / "output"
SI = ROOT / "manuscript" / "SI_Appendix"
CAUSES = ["rogue_ai", "ai_race", "organizational_risks", "malicious_use"]
LEVEL_OF = {"effort_none": "none/off", "reasoning_off": "none/off", "no_reasoning": "none/off",
            "effort_minimal": "minimal", "effort_low": "low", "effort_medium": "medium/on",
            "reasoning_on": "medium/on", "reasoning_mandatory": "medium/on", "effort_high": "high",
            "effort_xhigh": "xhigh", "effort_max": "max"}
REGION_SOURCE = {"figure script": "developer list of the study",
                 "added automatically (web search, high confidence; region_additions.csv)":
                     "web search (region_additions.csv)"}


def developer(model):
    d = model.split("/")[0]
    return "meta" if d == "meta-llama" else d


# ---------------------------------------------------------------- inputs
lines = (ROOT / "model_inclusion.csv").read_text(encoding="utf-8-sig").splitlines()
hdr = next(i for i, l in enumerate(lines[:5]) if re.match(r"model[;,]included", l.lower()))
incl = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";" if ";" in lines[hdr] else ",", dtype=str)
incl = incl[incl["model"].notna()].copy()
incl["included"] = incl["included"].str.strip().str.lower()
meta = pd.read_csv(OUT / "metadata.csv").drop_duplicates("model").set_index("model")
sample = pd.read_csv(OUT / "sample.csv").set_index("model")
ms = pd.read_csv(OUT / "model_level.csv").set_index("model")
calls = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(glob.glob(str(ROOT / "data" / "go_00[1-5]" / "results.csv")))])

# ---------------------------------------------------------------- S2: one row per model and setting
calls["valid"] = calls["outcome"] == "valid"
v = calls[calls["valid"]]
g = calls.groupby(["model", "condition"], sort=False)
s2 = pd.DataFrame({"n_calls": g.size(), "n_valid": g["valid"].sum(),
                   "is_default_effort": g["is_default_effort"].first().astype(str).eq("True")})
gv = v.groupby(["model", "condition"])
for c in ["probability", "lower_bound", "upper_bound"]:
    s2[f"mean_{c}"] = gv[c].mean()
s2["mean_reasoning_tokens"] = gv["reasoning_tokens"].mean()   # as reported by the provider (model A3)
for c in CAUSES:
    s2[f"n_{c}"] = v.assign(x=v["cause"] == c).groupby(["model", "condition"])["x"].sum()
s2 = s2.reset_index()
s2[[f"n_{c}" for c in CAUSES]] = s2[[f"n_{c}" for c in CAUSES]].fillna(0).astype(int)
s2.insert(2, "reasoning_level", s2["condition"].map(LEVEL_OF))
assert s2["reasoning_level"].notna().all()
enabled = s2["model"].map(meta["reasoning_default_enabled"])
s2["default_setting"] = (s2["is_default_effort"] | s2["condition"].isin(["no_reasoning", "reasoning_mandatory"])
                         | ((s2["condition"] == "reasoning_on") & (enabled == True))     # noqa: E712
                         | ((s2["condition"] == "reasoning_off") & (enabled == False)))  # noqa: E712
s2 = s2.drop(columns="is_default_effort")
s2["in_analysis"] = s2["model"].map(sample["in_sample"]).fillna(False).astype(bool) & (s2["n_valid"] > 0)
n_cells = pd.read_csv(OUT / "model_fit.csv")["n_cells"].iloc[0]
assert s2["in_analysis"].sum() == n_cells, (s2["in_analysis"].sum(), n_cells)

# ---------------------------------------------------------------- S1: one row per model
s1 = incl.rename(columns={"category": "inclusion_category", "reason": "inclusion_reason"})
s1.insert(1, "developer", s1["model"].map(developer))
inc = s1["included"] == "yes"
s1["analysed"] = s1["model"].map(sample["in_sample"]).map({True: "yes", False: "no"}).where(inc, "")
s1["exclusion_reason"] = s1["model"].map(sample["reason"]).fillna("").where(inc, "")
s1["region"] = s1["model"].map(sample["region"]).where(inc)
s1["region_source"] = s1["model"].map(sample["region_source"]).replace(REGION_SOURCE).where(s1["region"].notna(), "")
assert (meta["release_date_source"] == "OpenRouter listing").all(), "release dates must be the OpenRouter listing dates"
for c in ["release_date", "context_length", "input_price", "open_weights"]:
    s1[c] = s1["model"].map(meta[c])
s1 = s1.rename(columns={"release_date": "release_date_openrouter_listing",
                        "input_price": "input_price_usd_per_million_tokens"})
s1["price_source"] = s1["model"].map(sample["price_source"]).where(inc, "")
s1["settings_queried"] = s1["model"].map(s2.groupby("model")["condition"].agg("; ".join))
s1["default_settings"] = s1["model"].map(s2[s2["default_setting"]].groupby("model")["condition"].agg("; ".join))
s1.loc[s1["model"].map(ms["default_source"]).eq("median of conditions"), "default_settings"] = "none listed: mean over settings"
for c in ["probability", "lower_bound", "upper_bound"] + [f"share_{c}" for c in CAUSES]:
    s1[f"default_{c}"] = s1["model"].map(ms[c])
assert s1["default_probability"].notna().sum() == len(ms) == 241

# ---------------------------------------------------------------- S3: model estimates
README = pd.DataFrame({"sheet": [
    "coefficients", "joint_tests", "random_effects", "model_fit", "reasoning_levels", "typical_model", "convergence", "reliability"],
    "contents": [
        "All fixed effects. M1-M3 beta mixed models (estimate, upper, lower bound; logit link, constant precision); "
        "alternative specifications A1 reasoning level as a linear step split into "
        "within- and between-model parts; A2 within-model level indicators; A3 reasoning tokens (log10(tokens + 1), "
        "mean over runs) instead of the reasoning level. part = mean (logit) or precision (log); "
        "exp_estimate = odds ratio (mean part) or precision multiplier; "
        "Wald 95% CIs. Release date per year, context length and input price per tenfold increase.",
        "Joint Wald tests (region; six reasoning-level contrasts) and likelihood-ratio tests of linearity (A1 vs. A2).",
        "Random-effect variances (logit scale) for developers and for models within developers.",
        "Number of cells, log likelihood, AIC and convergence checks per model.",
        "Average marginal differences of each reasoning level from medium/on (percentage points; models answering at "
        "two or more levels), 95% CIs from 4,000 draws of the fixed effects, with the odds ratios of M1-M3.",
        "Model-implied values for a typical model (Fig. 1 B and C): random effects 0, other covariates at their means.",
        "Beta mixed model of one value per model at its default setting: precision ratio per year of release date.",
        "Reliability across the five runs: ICCs (cells with at least two valid runs, n_cells_icc), mean pairwise "
        "Spearman correlations, shares of identical answers, Fleiss' kappa; 95% CIs from 4,000 bootstrap resamples of models."]})
sheets = {"readme": README,
          "coefficients": pd.read_csv(OUT / "coefficients.csv"),
          "joint_tests": pd.read_csv(OUT / "tests.csv"),
          "random_effects": pd.read_csv(OUT / "variance_components.csv"),
          "model_fit": pd.read_csv(OUT / "model_fit.csv"),
          "reasoning_levels": pd.read_csv(OUT / "reasoning_levels.csv"),
          "typical_model": pd.read_csv(OUT / "predictions.csv"),
          "convergence": pd.read_csv(OUT / "convergence.csv"),
          "reliability": pd.read_csv(OUT / "reliability.csv")}
# the binomial models of the primary cause (M4, A1 rogue AI) are fitted but not reported
for name in ["coefficients", "joint_tests", "random_effects", "model_fit"]:
    sheets[name] = sheets[name][~sheets[name]["model"].str.contains("rogue AI")]

# ---------------------------------------------------------------- write
SI.mkdir(exist_ok=True)
s1.to_csv(SI / "Dataset_S1_models.csv", index=False, encoding="utf-8-sig")
s2.to_csv(SI / "Dataset_S2_answers.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(SI / "Dataset_S3_model_estimates.xlsx") as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)
print(f"S1: {len(s1)} models ({inc.sum()} included, {len(ms)} analysed); S2: {len(s2)} model-settings "
      f"({s2['in_analysis'].sum()} in the analysis); S3: {len(sheets) - 1} sheets -> {SI}")

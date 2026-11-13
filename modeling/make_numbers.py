"""
Write every number used in the manuscript as a LaTeX macro.

The macros are written into the manuscript's preamble between the lines
"% BEGIN NUMBERS" and "% END NUMBERS" (the journal submission does not allow \\input), and the
figure is copied next to the manuscript. Run after the R models and the figure:

    python modeling/build_metadata.py data/catalogue/models.json modeling/output/metadata.csv
    Rscript modeling/xrisk_models.R
    python modeling/figure.py
    python modeling/reliability.py
    python modeling/make_numbers.py
"""
import glob
import io
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "modeling" / "output"
MS = ROOT / "manuscript"
TEX = MS / "llm_xrisk_brief_report.tex"
ESPAI = ROOT / "2023 ESPAI - Cleaned and Anonymized Responses - espai2023 2024-01-08 redacted and clean data.csv"
sys.path.insert(0, str(ROOT))
import pipeline as pl  # noqa: E402  (reads model_inclusion.csv as written by Excel)

macros = {}


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = str(value)


def pct(x, d=1):
    return f"{x * 100:.{d}f}\\%"


def num(n):
    return f"{int(n):,}"


def pval(p):
    # math mode: the manuscript class does not load T1 encoding, so a text-mode "<" would print as an inverted "!"
    if p < 0.001:
        return "$P < 0.001$"
    return f"$P = {p:.3f}$" if p < 0.1 else f"$P = {p:.2f}$"


def day(ts):
    d = datetime.fromisoformat(ts[:19])
    return f"{d.day} {d:%B %Y}"


# ---------------------------------------------------------------- data collection
plan = json.loads((ROOT / "data" / "plan.json").read_text(encoding="utf-8"))
included = pl.load_included(pl.Config())
sample = pd.read_csv(OUT / "sample.csv")
ms = pd.read_csv(OUT / "model_level.csv")
runs = sorted(glob.glob(str(ROOT / "data" / "go_00[1-5]" / "results.csv")))
calls = pd.concat([pd.read_csv(p, low_memory=False) for p in runs])
sessions = [json.loads(l) for p in sorted(glob.glob(str(ROOT / "data" / "go_00[1-5]" / "sessions.jsonl")))
            for l in open(p, encoding="utf-8")]
analysed = set(sample.loc[sample["in_sample"], "model"])
valid = calls[calls["outcome"] == "valid"]

put("NCatalogue", num(plan["n_catalogue"]))
put("CatalogueDate", day(plan["created_at"]))
put("NDistinct", num(len(plan["models"])))
put("NIncluded", num(len(included)))
put("NRuns", ["zero", "one", "two", "three", "four", "five"][len(runs)])
put("NConditions", num(len(calls) // len(runs)))
put("NCalls", num(len(calls)))
put("NValid", num(len(valid)))
put("PctValid", pct(len(valid) / len(calls)))
put("NValidAnalysed", num(valid["model"].isin(analysed).sum()))
put("CollectionDate", day(min(s["started_at"] for s in sessions)))
put("TotalCost", f"{calls['cost_usd'].fillna(0).sum():.0f}")
put("MaxTokens", "32{,}768")
put("NAnalysed", num(len(ms)))
put("NDevelopers", num(ms["developer"].nunique()))
for r in ["US", "China", "Other"]:
    put(f"N{r}", num((ms["region"] == r).sum()))
put("NExcluded", num(len(included) - len(ms)))
put("NNoValid", num((sample["reason"] == "no valid answer in any run").sum()))
put("NNoRegion", num((sample["reason"] == "developer region not assigned").sum()))
put("NRegionAddedDevelopers", num(pd.read_csv(ROOT / "modeling" / "region_additions.csv").shape[0]))
put("NRegionAddedModels", num(ms["region_source"].str.startswith("added").sum()))
put("NFree", num(ms["price_source"].str.startswith("free").sum()))
fits = pd.read_csv(OUT / "model_fit.csv")
put("NCells", num(fits["n_cells"].iloc[0]))
lv = pd.read_csv(OUT / "reasoning_levels.csv")
put("NMultiLevel", num(lv["n_models_multi_level"].iloc[0]))
put("NCompleteRuns", num((valid[valid["model"].isin(analysed)].groupby(["model", "condition"]).size() == len(runs)).sum()))

# ---------------------------------------------------------------- LLMs vs. researchers
humans = pd.to_numeric(pd.read_csv(ESPAI, usecols=["extinction_basic_clean"]).iloc[:, 0], errors="coerce").dropna() / 100
p = ms["probability"]
for tag, x in [("LLM", p), ("Human", humans)]:
    put(f"{tag}Mean", pct(x.mean()))
    put(f"{tag}Median", pct(x.median(), 0 if tag == "Human" else 1))
    put(f"{tag}SD", f"{x.std() * 100:.1f}")
    put(f"{tag}IQR", f"{x.quantile(.25) * 100:.1f}--{x.quantile(.75) * 100:.1f}\\%")
    put(f"{tag}ShareFiveFifteen", pct(x.between(0.05, 0.15).mean(), 0))
    put(f"{tag}ShareQuarter", pct((x >= 0.25).mean(), 0))
    put(f"{tag}PctFive", pct(x.quantile(.05), 0 if tag == "Human" else 1))       # middle 90% of answers
    put(f"{tag}PctNinetyFive", pct(x.quantile(.95), 0 if tag == "Human" else 1))
put("NHuman", num(len(humans)))
put("HumanShareZero", pct((humans == 0).mean(), 0))
put("HumanShareHalf", pct((humans >= 0.5).mean(), 0))
n_quarter = int((p >= 0.25).sum())   # small counts as words ("only one gave ...")
put("LLMNQuarter", ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n_quarter]
    if n_quarter < 10 else num(n_quarter))
put("LLMRound", f"{round(p.median() * 100):.0f}\\%")   # the rounded typical answer ("around 10%")
put("LLMMin", pct(p.min(), 2))
put("LLMMax", pct(p.max(), 0))
put("SDRatio", f"{humans.std() / p.std():.0f}")
put("PVarTest", pval(stats.levene(p, humans, center="median").pvalue))
for r in ["US", "China", "Other"]:
    put(f"Mean{r}", pct(ms.loc[ms["region"] == r, "probability"].mean()))
cite = valid[valid["model"].isin(analysed)]["explanation"].fillna("")
put("ShareCiteExperts", pct(cite.str.contains(r"\b(?:surveys?|surveyed|polls?|polled|researchers?|experts?|scholars?|(?:super)?forecasters?)\b",
                                              case=False).mean(), 0))

# ---------------------------------------------------------------- do newer models lie closer together?
# one value per model at its default setting (xrisk_models.R)
conv = pd.read_csv(OUT / "convergence.csv").iloc[0]
assert conv["pd_hessian"]
put("PrecisionRatioDate", f"{conv['precision_ratio']:.2f}")
put("CIPrecisionRatioDate", f"{conv['ci_low']:.2f}--{conv['ci_high']:.2f}")
put("PPrecisionRatioDate", pval(conv["p"]))
# median split on release date (model_level.csv, default settings)
released = pd.to_datetime(ms["release_date"])
median_release = released.median()
put("MedianRelease", f"{median_release.day} {median_release:%B %Y}")
put("SDBeforeMedian", f"{p[released < median_release].std() * 100:.1f}")
put("SDFromMedian", f"{p[released >= median_release].std() * 100:.1f}")
put("NBeforeMedian", num((released < median_release).sum()))
put("NFromMedian", num((released >= median_release).sum()))

# ---------------------------------------------------------------- regression results
coef = pd.read_csv(OUT / "coefficients.csv")
coef = coef[coef["part"] == "mean (logit)"]
# main models (M1-M4): reasoning level categorical, reference medium/on
OUTCOME = {"Est": "M1 estimate", "UB": "M2 upper bound", "LB": "M3 lower bound", "Rogue": "M4 rogue AI (binomial)"}
TERM = {"Price": "log_price", "Date": "release_years", "China": "regionChina", "Other": "regionOther",
        "Context": "log_context", "Open": "open_weights",
        "None": "effortnone/off", "Xhigh": "effortxhigh", "Max": "effortmax"}
# supplementary S1: linear reasoning step split within/between models
OUTCOME_S1 = {"Est": "A1 estimate (within/between, linear)", "UB": "A1 upper bound (within/between, linear)",
              "LB": "A1 lower bound (within/between, linear)", "Rogue": "A1 rogue AI (binomial, within/between, linear)"}
TERM_S1 = {"Within": "step_within", "Between": "step_mean"}
for outcomes, terms in [(OUTCOME, TERM), (OUTCOME_S1, TERM_S1)]:
    for o, model in outcomes.items():
        for t, term in terms.items():
            r = coef[(coef["model"] == model) & (coef["term"] == term)].iloc[0]
            put(f"OR{t}{o}", f"{r['exp_estimate']:.2f}")
            put(f"CI{t}{o}", f"{r['exp_low']:.2f}--{r['exp_high']:.2f}")
            put(f"P{t}{o}", pval(r["p"]))
tests = pd.read_csv(OUT / "tests.csv")
for o, model in OUTCOME.items():
    t = tests[tests["model"] == model]
    put(f"PRegion{o}", pval(t[t["test"].str.startswith("region")]["p"].iloc[0]))
    put(f"PReasoning{o}", pval(t[t["test"].str.startswith("reasoning level")]["p"].iloc[0]))
lin = tests[tests["test"].str.contains("linear vs. categorical")]
put("PLinearityMin", pval(lin["p"].min()))
pred = pd.read_csv(OUT / "predictions.csv")
pp = pred[(pred["outcome"] == "probability") & (pred["varied"] == "log_price")]
put("PredPriceLow", pct(pp["predicted"].iloc[0]))
put("PredPriceHigh", pct(pp["predicted"].iloc[-1]))
put("PriceLow", f"\\${10 ** float(pp['value'].iloc[0]):.2f}")
put("PriceHigh", f"\\${10 ** float(pp['value'].iloc[-1]):.0f}")
vc = pd.read_csv(OUT / "variance_components.csv")
put("ShareModelVar", pct(vc[(vc["model"] == "M1 estimate") & vc["group"].str.startswith("model")]
                         ["share_of_random_variance"].iloc[0], 0))
lvl = lv.set_index(["outcome", "level"])
for o, col in [("Est", "probability"), ("UB", "upper_bound"), ("LB", "lower_bound")]:
    for L, tag in [("max", "Max"), ("none/off", "None")]:
        r = lvl.loc[(col, L)]
        put(f"Diff{o}{tag}", f"{r['diff_pp']:+.1f}".replace("-", "$-$"))
        put(f"CIDiff{o}{tag}", f"{r['diff_low']:.1f} to {r['diff_high']:.1f}".replace("-", "$-$"))
for c, tag in [("rogue_ai", "Rogue"), ("ai_race", "Race"), ("organizational_risks", "Org"), ("malicious_use", "Malicious")]:
    put(f"Share{tag}", pct(ms[f"share_{c}"].mean(), 0))
put("NRogueMajority", num((ms["share_rogue_ai"] > 0.5).sum()))

# ---------------------------------------------------------------- reliability across runs (reliability.py)
rel = pd.read_csv(OUT / "reliability.csv").set_index("measure")
for key, tag in [("icc1_probability", "ICCOneEst"), ("icc1_upper_bound", "ICCOneUB"), ("icc1_lower_bound", "ICCOneLB"),
                 ("icck_probability", "ICCkEst"), ("icck_upper_bound", "ICCkUB"), ("icck_lower_bound", "ICCkLB"),
                 ("icck_logit_probability", "ICCkLogitEst"), ("spearman_probability", "SpearmanEst"),
                 ("kappa_cause", "KappaCause")]:
    r = rel.loc[key]
    put(tag, f"{r['estimate']:.2f}")
    put(f"CI{tag}", f"{r['ci_low']:.2f}--{r['ci_high']:.2f}")
put("ShareIdentCause", pct(rel.loc["share_ident_cause", "estimate"], 0))
put("ShareIdentP", pct(rel.loc["share_ident_p", "estimate"], 0))
put("NCellsAllRuns", num(rel["n_cells_all_runs"].iloc[0]))
# model-setting combinations of the analysed models (incl. those without any valid answer)
comb = calls[calls["model"].isin(analysed)].assign(ok=lambda x: x["outcome"] == "valid") \
    .groupby(["model", "condition"])["ok"].sum()
assert (comb == len(runs)).sum() == rel["n_cells_all_runs"].iloc[0]
put("NCombAnalysed", num(len(comb)))
put("PctCombAllRuns", pct((comb == len(runs)).mean(), 0))

# ---------------------------------------------------------------- SI Appendix only (written to the SI file, not the manuscript)
si = {}


def sput(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name) and name not in macros, name
    si[name] = str(value)


def listing(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# catalogue: automatic exclusions
auto = pd.read_csv(ROOT / "data" / "models_excluded.csv")["reason"].str.replace(r":.*", "", regex=True).value_counts()
AUTO = {"batch_duplicate": "Batch", "alias_duplicate": "Alias", "router": "Router",
        "free_twin_of_paid_model": "FreeTwin", "same_canonical_slug_as": "Slug", "not_text": "NotText"}
assert set(auto.index) <= set(AUTO), auto
sput("NAutoExcluded", num(auto.sum()))
for k, tag in AUTO.items():
    sput(f"NAuto{tag}", num(auto.get(k, 0)))
# hand-coded inclusion decisions (model_inclusion.csv, possibly saved from Excel)
lines = (ROOT / "model_inclusion.csv").read_text(encoding="utf-8-sig").splitlines()
hdr = next(i for i, l in enumerate(lines[:5]) if re.match(r"model[;,]included", l.lower()))
incl = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";" if ";" in lines[hdr] else ",", dtype=str)
incl = incl[incl["model"].notna()]
assert len(incl) == len(plan["models"]) and (incl["included"].str.lower() == "yes").sum() == len(included)
CATEGORY = {"coding": "coding", "roleplay_creative": "roleplay or creative writing", "image_generation": "image generation",
            "narrow_perception_control": "narrow perceptual or control tasks", "multi_model_system": "systems combining several models",
            "web_search": "answers through live web search", "extraction_summary": "single document tasks",
            "privacy_blocked": "blocked by the account's privacy setting", "alias": "aliases of listed models",
            "serving_variant": "faster serving variants of listed models", "safety_filter": "safety classification",
            "translation": "translation", "audio_music": "music output", "audio_speech": "speech output",
            "domain_specific": "a single domain"}
cat = incl[incl["included"].str.lower() == "no"]["category"].value_counts()
assert set(cat.index) <= set(CATEGORY), cat
sput("NNotIncluded", num(cat.sum()))
sput("InclusionExclusions", listing(f"{CATEGORY[k]} ({v})" for k, v in cat.items()))
sput("NCloseCalls", num(incl["close_call"].notna().sum()))
# calls and answers
OUTCOMES = ["valid", "technical_failure", "invalid_json", "schema_violation", "refusal", "moderated", "truncated_no_answer"]
oc = calls["outcome"].value_counts()
assert set(oc.index) <= set(OUTCOMES), oc
for k in OUTCOMES:
    sput("NOut" + "".join(w.capitalize() for w in k.split("_")), num(oc.get(k, 0)))
sput("NLenient", num((valid["parse_mode"] == "lenient").sum()))
sput("NBoundsInconsistent", num((~valid["bounds_consistent"].astype(str).eq("True")).sum()))
first = calls.drop_duplicates(["model", "condition"])
for k, tag in [("study_limit", "Study"), ("provider_limit", "Provider"), ("context_limit", "Context")]:
    sput(f"NLimit{tag}", num((first["limit_source"] == k).sum()))
starts = sorted(datetime.fromisoformat(s["started_at"][:19]) for s in sessions)
sput("RunsFirstStart", f"{starts[0]:%H:%M}")
sput("RunsLastStart", f"{starts[-1]:%H:%M}")
# excluded models
sput("ModelsNoValid", listing(sample.loc[sample["reason"] == "no valid answer in any run", "model"]))
sput("ModelsNoRegion", listing(sample.loc[sample["reason"] == "developer region not assigned", "model"]))
# reasoning settings of the analysed models
ca = calls[calls["model"].isin(analysed)].drop_duplicates(["model", "condition"])
kind = {}
for m, g in ca.groupby("model"):
    c = set(g["condition"])
    kind[m] = ("effort" if g["effort"].notna().any() else "none" if c == {"no_reasoning"}
               else "mandatory" if c == {"reasoning_mandatory"} else "onoff" if c == {"reasoning_on", "reasoning_off"} else "?")
kind = pd.Series(kind)
assert "?" not in set(kind)
meta = pd.read_csv(OUT / "metadata.csv").drop_duplicates("model").set_index("model")
onoff = meta["reasoning_default_enabled"].reindex(kind[kind == "onoff"].index)
for k, tag in [("effort", "Effort"), ("none", "NoControl"), ("mandatory", "Mandatory"), ("onoff", "OnOff")]:
    sput(f"NSet{tag}", num((kind == k).sum()))
sput("NOnOffDefaultOn", num((onoff == True).sum()))   # noqa: E712
sput("NOnOffDefaultOff", num((onoff == False).sum()))  # noqa: E712
sput("NOnOffNoDefault", num(onoff.isna().sum()))
assert onoff.isna().sum() == (ms["default_source"] != "default setting").sum()
sput("NSingleLevel", num(len(ms) - lv["n_models_multi_level"].iloc[0]))
LEVEL_OF = {"effort_none": "none/off", "reasoning_off": "none/off", "no_reasoning": "none/off", "effort_minimal": "minimal",
            "effort_low": "low", "effort_medium": "medium/on", "reasoning_on": "medium/on", "reasoning_mandatory": "medium/on",
            "effort_high": "high", "effort_xhigh": "xhigh", "effort_max": "max"}
va = valid[valid["model"].isin(analysed)]
cl = va.drop_duplicates(["model", "condition"])["condition"].map(LEVEL_OF)
assert cl.notna().all()
LEVELS = ["none/off", "minimal", "low", "medium/on", "high", "xhigh", "max"]
sput("CellsPerLevel", listing(f"{L} ({(cl == L).sum()})" for L in LEVELS))
sput("NCellsFewerRuns", num((va.groupby(["model", "condition"]).size() < len(runs)).sum()))
# cells entering the ICCs: at least two valid runs (reliability.py), recounted here as a check
n_icc = int((va.groupby(["model", "condition"]).size() >= 2).sum())
assert n_icc == rel["n_cells_icc"].iloc[0], "ICC cell count differs from reliability.py"
sput("NCellsICC", num(n_icc))
# providers that served the valid answers
npv = va.groupby("model")["provider"].nunique()
sput("NSingleProvider", num((npv == 1).sum()))
sput("NProviders", num(va["provider"].nunique()))
# covariates
mm = meta.reindex(ms["model"])
assert (mm["release_date_source"] == "OpenRouter listing").all(), "release dates must be the OpenRouter listing dates"
sput("NOpenWeights", num(mm["open_weights"].sum()))
rd = pd.to_datetime(ms["release_date"])
sput("FirstRelease", f"{rd.min().day} {rd.min():%B %Y}")
sput("LastRelease", f"{rd.max().day} {rd.max():%B %Y}")
for r in ["US", "China", "Other"]:
    sput(f"Dev{r}", listing(sorted(ms.loc[ms["region"] == r, "developer"].unique())))
    sput(f"NDev{r}", num(ms.loc[ms["region"] == r, "developer"].nunique()))
pr = pred[(pred["outcome"] == "probability") & (pred["varied"] == "release_years")]["value"]
for tag, yr in [("Low", pr.iloc[0]), ("High", pr.iloc[-1])]:
    dt = pd.Timestamp("2025-01-01") + pd.Timedelta(days=float(yr) * 365.25)
    sput(f"DateGrid{tag}", f"{dt:%B %Y}")
# supplementary models (coefficients.csv, all parts)
allc = pd.read_csv(OUT / "coefficients.csv")


def coef_macros(model, term, tag, part="mean (logit)"):
    r = allc[(allc["model"] == model) & (allc["term"] == term) & (allc["part"] == part)].iloc[0]
    sput(f"OR{tag}", f"{r['exp_estimate']:.2f}")
    sput(f"CI{tag}", f"{r['exp_low']:.2f}--{r['exp_high']:.2f}")
    sput(f"P{tag}", pval(r["p"]))


# A3: reasoning tokens (log10(tokens + 1), mean over runs) instead of the reasoning level
S3 = {"Est": "A3 estimate (reasoning tokens instead of level)", "UB": "A3 upper bound (reasoning tokens instead of level)",
      "LB": "A3 lower bound (reasoning tokens instead of level)"}
for o, model in S3.items():
    coef_macros(model, "log_reasoning_tokens", f"Tok{o}")
# correlation of this measure with the reasoning level (0 = none/off ... 6 = max), cells of the analysed models
tk = va.assign(ltok=np.log10(va["reasoning_tokens"] + 1)).groupby(["model", "condition"], as_index=False)["ltok"].mean()
assert tk["ltok"].notna().all() and len(tk) == len(cl)
tk["step"] = tk["condition"].map(LEVEL_OF).map({L: i for i, L in enumerate(LEVELS)})
sput("CorTokAll", f"{np.corrcoef(tk['step'], tk['ltok'])[0, 1]:.2f}")
tw = tk[tk.groupby("model")["step"].transform("nunique") >= 2].copy()
for c in ["step", "ltok"]:
    tw[c] = tw[c] - tw.groupby("model")[c].transform("mean")
sput("CorTokWithin", f"{np.corrcoef(tw['step'], tw['ltok'])[0, 1]:.2f}")
for o, model in OUTCOME.items():
    if o == "Rogue":
        continue
    t = lin[lin["model"].str.startswith(OUTCOME_S1[o])]
    sput(f"PLinearity{o}", pval(t["p"].iloc[0]))
sput("ShareCiteN", num(len(cite)))
for key, tag in [("spearman_upper_bound", "SpearmanUB"), ("spearman_lower_bound", "SpearmanLB")]:
    r = rel.loc[key]
    sput(tag, f"{r['estimate']:.2f}")
    sput(f"CI{tag}", f"{r['ci_low']:.2f}--{r['ci_high']:.2f}")
sput("ShareIdentAll", pct(rel.loc["share_ident_all", "estimate"], 0))

# ---------------------------------------------------------------- write
def write_block(path, values, label):
    block = [f"% BEGIN NUMBERS (generated by modeling/make_numbers.py; do not edit by hand)"]
    block += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    block += ["% END NUMBERS"]
    tex = path.read_text(encoding="utf-8")
    new, n = re.subn(r"% BEGIN NUMBERS.*?% END NUMBERS", lambda _: "\n".join(block), tex, flags=re.S)
    assert n == 1, f"markers not found in {path.name}"
    path.write_text(new, encoding="utf-8")
    used = set(re.findall(r"\\([A-Za-z]+)", new.split("% END NUMBERS")[1]))
    TEMPLATE = {"Firstpage", "Endparasplit", "Parasplit", "SItext", "FloatBarrier"}
    missing = sorted(u for u in used if u[0].isupper() and u not in values and u not in TEMPLATE)
    print(f"{len(values)} macros written to {path.name}")
    if missing:
        print(f"WARNING ({label}): used in the text but not defined:", missing)


# every number, also without the manuscript files (e.g. in the public repository)
(OUT / "numbers.tex").write_text("\n".join(["% All numbers of the manuscript and SI Appendix (generated by modeling/make_numbers.py)"]
                                           + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in {**macros, **si}.items()]) + "\n",
                                 encoding="utf-8")
print(f"{len({**macros, **si})} macros written to {OUT / 'numbers.tex'}")
if TEX.exists():
    write_block(TEX, macros, "manuscript")
MS.mkdir(exist_ok=True)
shutil.copyfile(OUT / "figure1.pdf", MS / "fig1.pdf")
print(f"figure copied to {MS / 'fig1.pdf'}")
SI_TEX = MS / "SI_Appendix" / "llm_xrisk_SI.tex"
if SI_TEX.exists():
    write_block(SI_TEX, {**macros, **si}, "SI Appendix")

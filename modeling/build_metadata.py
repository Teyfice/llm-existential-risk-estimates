"""Build metadata.csv (one row per OpenRouter model id) from the OpenRouter /models JSON.

Usage: python build_metadata.py models.json metadata.csv [release_date_overrides.csv]
Release dates are the OpenRouter listing dates. The analysis runs WITHOUT the overrides file
(decision 2026-11-11); passing it would replace the listing date by web-verified release dates."""
import json, re, sys
import numpy as np, pandas as pd

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "metadata.csv"
overrides = sys.argv[3] if len(sys.argv) > 3 else None   # release_date_overrides.csv
j = pd.json_normalize(json.load(open(src, encoding="utf-8"))["data"])

def params_from_text(*texts):
    """Total parameters from names like '65b', '397b-a17b', '2.4t-a95b', '8x22b'."""
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        s = t.lower()
        mx = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", s)
        if mx:
            return int(mx.group(1)) * float(mx.group(2)) * 1e9
        mt = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([tbm])(?![a-z])", s)
        if mt:
            return float(mt.group(1)) * {"t": 1e12, "b": 1e9, "m": 1e6}[mt.group(2)]
    return np.nan

def params_from_description(t):
    """'1 trillion total parameters', '1.6T total parameters', 'A 12B parameter model'."""
    if not isinstance(t, str):
        return np.nan
    unit = {"t": 1e12, "trillion": 1e12, "b": 1e9, "billion": 1e9}
    pats = [r"(\d+(?:\.\d+)?)\s?(trillion|billion|t|b)\b\s+total",
            r"(\d+(?:\.\d+)?)\s?(trillion|billion|t|b)\b[\s-]+param",
            r"(?:at|of over|over)\s+(\d+(?:\.\d+)?)\s?(trillion|billion|t|b)\b"]
    for p in pats:
        mm = re.search(p, t.lower())
        if mm:
            return float(mm.group(1)) * unit[mm.group(2)]
    return np.nan

hf = j["hugging_face_id"].fillna("").str.strip()
meta = pd.DataFrame({
    "model": j["id"],
    "release_date": pd.to_datetime(j["created"], unit="s").dt.date,
    "context_length": j["context_length"],
    "input_price": j["pricing.prompt"].astype(float) * 1e6,      # USD per M tokens
    "output_price": j["pricing.completion"].astype(float) * 1e6,
    "open_weights": (hf != "").astype(int),
    "knowledge_cutoff": pd.to_datetime(j["knowledge_cutoff"], errors="coerce").dt.date,
    "reasoning_mandatory": j["reasoning.mandatory"],
    "reasoning_default_enabled": j["reasoning.default_enabled"],
})
# parameters only where weights are public (closed-model names rarely state size)
meta["parameters"] = [params_from_text(h, i.split("/")[-1]) if o else np.nan
                      for h, i, o in zip(hf, j["id"], meta["open_weights"])]
desc = [params_from_description(t) for t in j["description"]]
meta["parameters"] = meta["parameters"].fillna(
    pd.Series(desc, index=meta.index).where(meta["open_weights"] == 1))
# free variants: borrow the price of the paid sibling; otherwise unknown
paid = meta.set_index("model")[["input_price", "output_price"]]
for col in ["input_price", "output_price"]:
    free = meta["model"].str.endswith(":free") | (meta[col] <= 0)
    sib = meta["model"].str.replace(":free", "", regex=False).map(paid[col])
    meta.loc[free, col] = sib[free].where(sib[free] > 0)
# release dates: OpenRouter listing date unless a verified date is given
meta["release_date_source"] = "OpenRouter listing"
if overrides:
    ov = pd.read_csv(overrides).set_index("model")
    hit = meta["model"].isin(ov.index)
    meta.loc[hit, "release_date"] = pd.to_datetime(
        meta.loc[hit, "model"].map(ov["release_date"])).dt.date
    meta.loc[hit, "release_date_source"] = "verified (release_date_overrides.csv)"
meta.to_csv(out, index=False)
print(meta.describe(include="all").T[["count"]].T.to_string())

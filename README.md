# LLM estimates of existential risk from AI

Data and code for

> Kauk, J. *Large language models put existential risk from artificial intelligence at 10%*. Preprint (2026).

**Preprint (paper and SI Appendix): [llm_xrisk_preprint.pdf](llm_xrisk_preprint.pdf)**

We asked 241 large language models (via [OpenRouter](https://openrouter.ai/)) the question on existential risk from AI of the
2023 Expert Survey on Progress in AI, at every reasoning setting they offer and in five runs. This repository contains the
prompt, the raw responses of all calls, the analysis code, the outputs of all models, the preprint and the SI Appendix.

## Contents

| Path | Contents |
|---|---|
| `llm_xrisk_preprint.pdf` | The preprint: paper and SI Appendix in one PDF |
| `PROMPT.txt` | The prompt (the four causes are shuffled per call; see SI Appendix, section 1) |
| `pipeline.py` | Data collection: catalogue download, frozen plan, calls to OpenRouter, parsing of the replies |
| `model_inclusion.csv` | Hand-coded inclusion decision for each of the 344 catalogue models (category, reason, close-call flag) |
| `data/catalogue/` | OpenRouter catalogue (`models.json`) and provider data (`endpoints.json`), downloaded 5 November 2026 |
| `data/plan.json` | Frozen plan: models, reasoning settings, output limits, prompt hash |
| `data/models_metadata.csv`, `data/models_excluded.csv` | Catalogue metadata; entries removed automatically (batch versions, aliases, routers, duplicates) |
| `data/go_00N/raw.jsonl` | Runs 1 to 5: every attempt with request, HTTP status, response headers and full reply, plus generation metadata |
| `data/go_00N/results.csv` | One row per call with the parsed answer (estimate, bounds, cause, explanation) and its outcome |
| `data/go_00N/` (other files) | Per-model availability, spending log, status and session log of each run |
| `modeling/` | Analysis scripts (see below) and `output/`: all model outputs, the figure and `numbers.tex` (every number in the paper) |
| `manuscript/Preprint/` | LaTeX sources of the preprint, with the paper and the SI Appendix also as separate PDFs |
| `manuscript/SI_Appendix/` | SI Appendix (PDF) and Datasets S1 to S3 (models, answers per model and setting, all model estimates) |

## Reproducing the results

Requirements: Python 3.10 (pandas 2.2, numpy 1.26, scipy 1.15, matplotlib 3.10, openpyxl 3.1, httpx 0.28; see
`requirements.txt`) and R 4.5.3 with glmmTMB 1.1.13, data.table and MASS.

1. Download the cleaned, anonymized responses of the 2023 Expert Survey on Progress in AI from OSF
   (<https://osf.io/vmdny>) and save the file in the repository root as
   `2023 ESPAI - Cleaned and Anonymized Responses - espai2023 2024-01-08 redacted and clean data.csv`
   (used for the comparison with AI researchers; not redistributed here).
2. Run, from the repository root:

   ```
   python modeling/build_metadata.py data/catalogue/models.json modeling/output/metadata.csv
   Rscript modeling/xrisk_models.R
   python modeling/figure.py
   python modeling/reliability.py
   python modeling/make_numbers.py
   python modeling/make_si_datasets.py
   ```

`xrisk_models.R` fits the beta mixed models (M1 to M3) and the alternative specifications; `figure.py` draws Fig. 1;
`reliability.py` computes the agreement across runs; `make_numbers.py` writes every number of the paper to
`modeling/output/numbers.tex`; `make_si_datasets.py` writes Datasets S1 to S3. The release date of a model is the date it
was first listed on OpenRouter.

No API key and no model calls are needed to reproduce the results. `python pipeline.py export --number N` rebuilds
`results.csv` of run N from its `raw.jsonl`.

## Collecting new data

`pipeline.py` can repeat the data collection (`python pipeline.py --help`). This sends requests to OpenRouter and costs
money. It needs an OpenRouter API key in a file `secret_key_openrouter.txt` in the repository root; this file is listed
in `.gitignore` and must never be committed.

## Notes

- Account-specific fields were removed from the published logs: the account balance in `sessions.jsonl`, the
  workspace identifier in the generation metadata and cookies in the response headers of `raw.jsonl`. No analyzed value
  is affected.
- A pilot run with an earlier version of the prompt was not analyzed and is not included.

# Mixed beta regressions of LLM estimates of AI existential risk (GO 1 to GO 5).
#
# Main models (main text):
#   M1  estimate        beta GLMM: region + release date + log10 context + log10 input price
#   M2  upper bound       + open weights + reasoning level (categorical, reference medium/on)
#   M3  lower bound       + (1 | developer/model); constant precision
#   M4  rogue AI        binomial GLMM, same fixed and random effects (runs naming rogue AI)
#
# Additional models (SI Appendix; labelled A1-A3 in the output, S1-S3 in the code):
#   A1  M1-M4 with the reasoning level as a linear step split into the model mean (between
#       models) and the deviation from it (within models; Bell & Jones 2015)
#   A2  M1-M3 with within-model level indicators (centred on the model mean, reference
#       medium/on) plus the model mean; LR test A1 vs. A2 = test of within-model linearity
#   A3  M1-M3 with the reasoning tokens reported for the answer (log10(tokens + 1), mean over
#       runs) instead of the reasoning level
#
# Unit of analysis: one row per model x reasoning condition, the 5 runs averaged.
# Reasoning level: none/off = 0, minimal = 1, low = 2, medium/on = 3, high = 4,
# xhigh = 5, max = 6.
#
# Usage (from the project folder):
#   python modeling/build_metadata.py data/catalogue/models.json modeling/output/metadata.csv
#   (release date = OpenRouter listing date for every model; release_date_overrides.csv is not used)
#   Rscript modeling/xrisk_models.R
# Writes CSV tables to modeling/output/.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(data.table)
  library(MASS)
})
set.seed(20261106)

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
ROOT <- dirname(HERE)
OUT <- file.path(HERE, "output")
dir.create(OUT, showWarnings = FALSE)

GOS <- 1:5
CAUSES <- c("rogue_ai", "ai_race", "organizational_risks", "malicious_use")
LEVELS <- c("none/off", "minimal", "low", "medium/on", "high", "xhigh", "max")
REF <- "medium/on"
STEP <- setNames(0:6, LEVELS)
LEVEL_OF <- c(effort_none = "none/off", reasoning_off = "none/off", no_reasoning = "none/off",
              effort_minimal = "minimal", effort_low = "low",
              effort_medium = "medium/on", reasoning_on = "medium/on", reasoning_mandatory = "medium/on",
              effort_high = "high", effort_xhigh = "xhigh", effort_max = "max")

# Developer -> region (as in llm_xrisk_figure.py). Unlisted developers are left out.
REGION <- c(
  openai = "US", google = "US", anthropic = "US", meta = "US", `meta-llama` = "US", `x-ai` = "US",
  nvidia = "US", amazon = "US", microsoft = "US", `ibm-granite` = "US", inception = "US",
  thinkingmachines = "US", reflection = "US", fireworks = "US", liquid = "US", nousresearch = "US",
  writer = "US", perceptron = "US", rekaai = "US", perplexity = "US",
  qwen = "China", `z-ai` = "China", deepseek = "China", `bytedance-seed` = "China", bytedance = "China",
  tencent = "China", moonshotai = "China", minimax = "China", xiaomi = "China", inclusionai = "China",
  stepfun = "China", meituan = "China", `dots-studio` = "China",
  mistralai = "Other", cohere = "Other", upstage = "Other", sakana = "Other")
REGION_SOURCE <- setNames(rep("figure script", length(REGION)), names(REGION))
# Regions found by an automatic web search, flagged as such (see region_additions.csv)
added <- fread(file.path(HERE, "region_additions.csv"), encoding = "UTF-8")
stopifnot(!any(added$developer %in% names(REGION)))
REGION <- c(REGION, setNames(added$region, added$developer))
REGION_SOURCE <- c(REGION_SOURCE, setNames(paste0("added automatically (web search, ", added$confidence,
                                                  " confidence; region_additions.csv)"), added$developer))

# Answers enter the beta models unchanged; fit_beta() stops if any value is exactly 0 or 1.
invlogit <- function(z) 1 / (1 + exp(-z))

# ---------------------------------------------------------------- data
read_inclusion <- function(path) {
  first <- sub("^﻿", "", readLines(path, n = 5, encoding = "UTF-8", warn = FALSE))
  hdr <- which(grepl("^model[;,]included", tolower(first)))[1]
  if (is.na(hdr)) stop("model_inclusion.csv: no header 'model,included' in the first lines")
  sep <- if (grepl(";", first[hdr])) ";" else ","
  x <- fread(path, sep = sep, skip = hdr - 1, encoding = "UTF-8", colClasses = "character")
  bad <- x[!tolower(trimws(included)) %in% c("yes", "no")]
  if (nrow(bad)) stop("model_inclusion.csv: included must be yes/no: ", paste(bad$model, collapse = ", "))
  trimws(x[tolower(trimws(included)) == "yes", model])
}

included <- read_inclusion(file.path(ROOT, "model_inclusion.csv"))
cols <- c("go", "model", "condition", "is_default_effort", "outcome", "probability",
          "lower_bound", "upper_bound", "cause", "reasoning_tokens")
calls <- rbindlist(lapply(GOS, function(g)
  fread(file.path(ROOT, "data", sprintf("go_%03d", g), "results.csv"), select = cols, encoding = "UTF-8")))
stopifnot(setequal(unique(calls$go), GOS))
valid <- calls[outcome == "valid" & model %in% included]
valid[, is_default_effort := as.logical(is_default_effort)]
for (c in CAUSES) valid[, (paste0("share_", c)) := as.numeric(cause == c)]

cells <- valid[, c(list(probability = mean(probability), lower_bound = mean(lower_bound),
                        upper_bound = mean(upper_bound), n_runs = .N,
                        # A3: reasoning tokens reported for the answer, log10(tokens + 1), mean over runs
                        log_reasoning_tokens = mean(log10(reasoning_tokens + 1)),
                        rogue_count = sum(cause == "rogue_ai"),
                        is_default_effort = any(is_default_effort %in% TRUE)),
                   lapply(.SD, mean)),
               by = .(model, condition), .SDcols = paste0("share_", CAUSES)]
cells[, level := LEVEL_OF[condition]]
stopifnot(!anyNA(cells$level))
cells[, step := STEP[level]]
cells[, developer := sub("/.*", "", model)]
cells[developer == "meta-llama", developer := "meta"]
cells[, region := unname(REGION[sub("/.*", "", model)])]
cells[, region_source := fifelse(is.na(region), "not assigned", unname(REGION_SOURCE[sub("/.*", "", model)]))]

meta <- fread(file.path(OUT, "metadata.csv"), encoding = "UTF-8")
meta <- unique(meta, by = "model")
cells <- merge(cells, meta[, .(model, release_date = as.IDate(release_date), context_length,
                               input_price, open_weights, reasoning_default_enabled)],
               by = "model", all.x = TRUE)
# Free models without a paid version have no price in the metadata: they cost nothing, so their
# price is set to 0 (drawn and modelled at the floor of $0.001 per million tokens) and flagged.
cells[, price_source := "OpenRouter listing"]
cells[is.na(input_price) & grepl(":free$", model),
      `:=`(input_price = 0, price_source = "free on OpenRouter, no paid version: set to 0 (floored at $0.001)")]
cells[, release_years := as.numeric(release_date - as.IDate("2025-01-01")) / 365.25]
cells[, log_context := log10(context_length)]
cells[, log_price := log10(pmax(input_price, 1e-3))]

# sample: region assigned and all covariates known
cells[, in_sample := !is.na(region) & !is.na(release_years) & !is.na(log_context) &
        !is.na(log_price) & !is.na(open_weights)]
sample_tab <- cells[, .(cells = .N, region = region[1], region_source = region_source[1],
                        price_source = price_source[1], in_sample = in_sample[1],
                        reason = fifelse(is.na(region[1]), "developer region not assigned",
                                 fifelse(is.na(log_price[1]), "input price unknown",
                                 fifelse(in_sample[1], "", "covariate missing")))),
                    by = .(model, developer)]
no_valid <- setdiff(included, cells$model)
sample_tab <- rbind(sample_tab, data.table(model = no_valid, developer = sub("/.*", "", no_valid),
                                           cells = 0L, region = NA_character_, in_sample = FALSE,
                                           reason = "no valid answer in any run"), fill = TRUE)
fwrite(sample_tab[order(-in_sample, model)], file.path(OUT, "sample.csv"))

d <- cells[in_sample == TRUE]
d[, region := factor(region, levels = c("US", "China", "Other"))]
d[, effort := factor(level, levels = c(REF, setdiff(LEVELS, REF)))]   # medium/on = reference
LV <- setdiff(LEVELS, REF)
EFF <- paste0("effort", LV)
# supplementary terms: linear step split within/between (S1); within-centred level indicators (S2)
d[, step_mean := ave(step, model)]
d[, step_within := step - step_mean]
LV_COL <- paste0("w_", gsub("[^a-z]", "_", LV))
for (i in seq_along(LV)) {
  d[, (LV_COL[i]) := as.numeric(level == LV[i])]
  d[, (LV_COL[i]) := get(LV_COL[i]) - ave(get(LV_COL[i]), model)]
}
stopifnot(!anyNA(d$log_reasoning_tokens))
n <- nrow(d)
cat(sprintf("Sample: %d cells, %d models, %d developers (of %d included models)\n",
            n, uniqueN(d$model), uniqueN(d$developer), length(included)))

# ---------------------------------------------------------------- models
BASE <- c("region", "release_years", "log_context", "log_price", "open_weights")
COVS <- c(BASE, "effort")                          # main models
COVS_S1 <- c(BASE, "step_within", "step_mean")      # supplementary S1
COVS_S2 <- c(BASE, "step_mean", LV_COL)             # supplementary S2
COVS_S3 <- c(BASE, "log_reasoning_tokens")          # supplementary S3: tokens instead of level
FIX <- as.formula(paste("~", paste(COVS, collapse = " + ")))
RE <- "(1 | developer/model)"
OUTCOMES <- c(probability = "M1 estimate", upper_bound = "M2 upper bound", lower_bound = "M3 lower bound")
SHORT <- c(probability = "estimate", upper_bound = "upper bound", lower_bound = "lower bound")

fit_beta <- function(yvar, covs, disp = ~ 1) {
  dd <- copy(d)
  dd[, y := get(yvar)]
  stopifnot(all(dd$y > 0 & dd$y < 1))              # beta distribution: strictly between 0 and 1
  f <- as.formula(paste("y ~", paste(covs, collapse = " + "), "+", RE))
  # constant precision in all models
  glmmTMB(f, dispformula = disp, family = beta_family(link = "logit"), data = dd)
}
fit_rogue <- function(covs)
  glmmTMB(as.formula(paste("cbind(rogue_count, n_runs - rogue_count) ~", paste(covs, collapse = " + "), "+", RE)),
          family = binomial(link = "logit"), data = d)
fits <- lapply(names(OUTCOMES), fit_beta, covs = COVS); names(fits) <- names(OUTCOMES)
fit4 <- fit_rogue(COVS)
fitsS1 <- lapply(names(OUTCOMES), fit_beta, covs = COVS_S1); names(fitsS1) <- names(OUTCOMES)
fit4S1 <- fit_rogue(COVS_S1)
fitsS2 <- lapply(names(OUTCOMES), fit_beta, covs = COVS_S2); names(fitsS2) <- names(OUTCOMES)
fitsS3 <- lapply(names(OUTCOMES), fit_beta, covs = COVS_S3)
names(fitsS3) <- names(OUTCOMES)

LABEL <- c(`(Intercept)` = "Intercept", regionChina = "Region: China (vs. US)",
           regionOther = "Region: Other (vs. US)", release_years = "Release date (per year)",
           log_context = "Context length (per 10x)", log_price = "Input price (per 10x)",
           open_weights = "Open weights (vs. closed)",
           setNames(paste0("Reasoning level ", LV, " (vs. medium/on)"), EFF),
           step_within = "Reasoning level, within model (per step)",
           step_mean = "Reasoning level, model average (per step)",
           log_reasoning_tokens = "Reasoning tokens (per 10x; log10(tokens + 1))",
           setNames(paste0("Reasoning level ", LV, " (vs. medium/on, within model)"), LV_COL))

tidy <- function(fit, name, part) {
  s <- summary(fit)$coefficients[[part]]
  if (is.null(s)) return(NULL)
  data.table(model = name, part = c(cond = "mean (logit)", disp = "precision (log)")[part],
             term = rownames(s), label = unname(fifelse(is.na(LABEL[rownames(s)]), rownames(s), LABEL[rownames(s)])),
             estimate = s[, 1], se = s[, 2],
             exp_estimate = exp(s[, 1]), exp_low = exp(s[, 1] - 1.96 * s[, 2]),
             exp_high = exp(s[, 1] + 1.96 * s[, 2]), p = s[, 4],
             n_cells = nobs(fit), n_models = uniqueN(d$model), n_developers = uniqueN(d$developer))
}
# output labels A1-A3 ("additional"), so that they do not clash with SI Datasets S1-S3
S1_NAME <- c(setNames(paste("A1", SHORT, "(within/between, linear)"), names(OUTCOMES)),
             rogue = "A1 rogue AI (binomial, within/between, linear)")
S2_NAME <- setNames(paste("A2", SHORT, "(within-model categories)"), names(OUTCOMES))
S3_NAME <- setNames(paste("A3", SHORT, "(reasoning tokens instead of level)"), names(OUTCOMES))
all_fits <- c(setNames(fits, OUTCOMES), list(`M4 rogue AI (binomial)` = fit4),
              setNames(fitsS1, S1_NAME[names(OUTCOMES)]), setNames(list(fit4S1), S1_NAME[["rogue"]]),
              setNames(fitsS2, S2_NAME), setNames(fitsS3, S3_NAME))
coef_tab <- rbindlist(lapply(names(all_fits), function(nm)
  rbind(tidy(all_fits[[nm]], nm, "cond"), tidy(all_fits[[nm]], nm, "disp"))))
fwrite(coef_tab, file.path(OUT, "coefficients.csv"))

vc_tab <- rbindlist(lapply(names(all_fits), function(nm) {
  v <- VarCorr(all_fits[[nm]])$cond
  data.table(model = nm, group = names(v),
             variance = sapply(v, function(x) attr(x, "stddev")^2),
             sd = sapply(v, function(x) attr(x, "stddev")))
}))
vc_tab[group == "model:developer", group := "model (within developer)"]
vc_tab[, share_of_random_variance := variance / sum(variance), by = model]
fwrite(vc_tab, file.path(OUT, "variance_components.csv"))

fit_tab <- rbindlist(lapply(names(all_fits), function(nm) {
  f <- all_fits[[nm]]
  data.table(model = nm, n_cells = nobs(f), n_models = uniqueN(d$model),
             n_developers = uniqueN(d$developer), logLik = as.numeric(logLik(f)), AIC = AIC(f),
             converged = f$fit$convergence == 0, pd_hessian = f$sdr$pdHess)
}))
fwrite(fit_tab, file.path(OUT, "model_fit.csv"))

# ---------------------------------------------------------------- tests
wald <- function(fit, terms) {
  b <- fixef(fit)$cond[terms]; V <- vcov(fit)$cond[terms, terms]
  chi <- as.numeric(t(b) %*% solve(V, b))
  c(chi2 = chi, df = length(terms), p = pchisq(chi, length(terms), lower.tail = FALSE))
}
wald_row <- function(fit, name, terms, label) {
  w <- wald(fit, terms)
  data.table(model = name, test = label, statistic = w[["chi2"]], df = w[["df"]], p = w[["p"]])
}
main_fits <- c(setNames(fits, OUTCOMES), list(`M4 rogue AI (binomial)` = fit4))
tests <- rbindlist(c(
  lapply(names(main_fits), function(nm)
    wald_row(main_fits[[nm]], nm, c("regionChina", "regionOther"), "region (China, Other = US), Wald")),
  lapply(names(main_fits), function(nm)
    wald_row(main_fits[[nm]], nm, EFF, "reasoning level (6 contrasts with medium/on), Wald")),
  lapply(names(OUTCOMES), function(o) {
    a <- anova(fitsS1[[o]], fitsS2[[o]])
    data.table(model = paste(S1_NAME[[o]], "vs.", S2_NAME[[o]]),
               test = "within-model reasoning level: linear vs. categorical, LR",
               statistic = a$Chisq[2], df = a$`Chi Df`[2], p = a$`Pr(>Chisq)`[2])
  })))
fwrite(tests, file.path(OUT, "tests.csv"))

# ---------------------------------------------------------------- predictions
# Model-implied values for a typical model (random effects 0), the other covariates and the
# reasoning-level indicators at their means over cells (region dummies at sample shares),
# 95% CI from the fixed-effect covariance.
models_d <- unique(d, by = "model")
base <- colMeans(model.matrix(FIX, d))

predict_grid <- function(fit, var, grid, at = NULL) {
  b <- fixef(fit)$cond; V <- vcov(fit)$cond
  X <- matrix(base[names(b)], nrow = length(grid), ncol = length(b), byrow = TRUE,
              dimnames = list(NULL, names(b)))
  if (!is.null(at)) for (k in names(at)) X[, k] <- at[[k]]
  if (!is.null(var)) X[, var] <- grid
  eta <- drop(X %*% b); se <- sqrt(rowSums((X %*% V) * X))
  data.table(predicted = invlogit(eta), ci_low = invlogit(eta - 1.96 * se), ci_high = invlogit(eta + 1.96 * se))
}
yr_grid <- seq(quantile(models_d$release_years, 0.02), max(models_d$release_years), length.out = 60)
pr_grid <- seq(quantile(models_d$log_price, 0.02), quantile(models_d$log_price, 0.98), length.out = 60)
pred <- rbindlist(c(
  lapply(names(OUTCOMES), function(o) cbind(outcome = o, varied = "release_years", value = yr_grid,
                                            predict_grid(fits[[o]], "release_years", yr_grid))),
  lapply(names(OUTCOMES), function(o) cbind(outcome = o, varied = "log_price", value = pr_grid,
                                            predict_grid(fits[[o]], "log_price", pr_grid))),
  lapply(names(OUTCOMES), function(o) rbindlist(lapply(c("US", "China", "Other"), function(r)
    cbind(outcome = o, varied = "region", value = r,
          predict_grid(fits[[o]], NULL, 0, at = list(regionChina = as.numeric(r == "China"),
                                                      regionOther = as.numeric(r == "Other")))))))),
  fill = TRUE)
pred[varied == "region", n_models := sapply(value, function(r) sum(models_d$region == r))]
fwrite(pred, file.path(OUT, "predictions.csv"))


# ---------------------------------------------------------------- reasoning-level contrasts
# Average marginal difference of each level vs. medium/on (M1-M3), within models that answered
# at >= 2 levels: every cell of those models is set to the level (each model's random effects
# kept), predictions averaged. 95% CI from 4,000 draws of the fixed effects.
multi <- d[, .(k = uniqueN(level)), by = model][k >= 2, model]
contrasts_for <- function(o) {
  f <- fits[[o]]
  e <- d[model %in% multi]
  b <- fixef(f)$cond; V <- vcov(f)$cond
  X0 <- model.matrix(FIX, e)[, names(b)]
  re_part <- predict(f, newdata = e, type = "link", re.form = NULL) - drop(X0 %*% b)
  draws <- mvrnorm(4000, b, V)
  at_level <- function(L) {
    cf <- copy(e); cf[, effort := factor(L, levels = levels(d$effort))]
    X <- model.matrix(FIX, cf)[, names(b)]
    list(point = mean(invlogit(drop(X %*% b) + re_part)),
         draws = colMeans(invlogit(X %*% t(draws) + re_part)))
  }
  ref <- at_level(REF)
  s <- summary(f)$coefficients$cond
  rbindlist(lapply(LEVELS, function(L) {
    nm <- e[level == L, uniqueN(model)]
    if (nm == 0) return(NULL)
    a <- at_level(L)
    diff <- (a$draws - ref$draws) * 100
    term <- paste0("effort", L)
    data.table(outcome = o, level = L, models = nm,
               odds_ratio = if (L == REF) 1 else exp(s[term, 1]),
               or_low = if (L == REF) 1 else exp(s[term, 1] - 1.96 * s[term, 2]),
               or_high = if (L == REF) 1 else exp(s[term, 1] + 1.96 * s[term, 2]),
               p = if (L == REF) NA_real_ else s[term, 4],
               diff_pp = (a$point - ref$point) * 100,
               diff_low = if (L == REF) 0 else quantile(diff, 0.025),
               diff_high = if (L == REF) 0 else quantile(diff, 0.975),
               predicted = a$point, n_models_multi_level = length(multi))
  }))
}
fwrite(rbindlist(lapply(names(OUTCOMES), contrasts_for)), file.path(OUT, "reasoning_levels.csv"))

# ---------------------------------------------------------------- data for the figure
# One value per model at its default setting (default effort, the single setting of
# models without reasoning control, or the on/off state OpenRouter enables by default);
# median over conditions where no default is identifiable.
d[, is_default := is_default_effort | condition %in% c("no_reasoning", "reasoning_mandatory") |
    (condition == "reasoning_on" & reasoning_default_enabled %in% TRUE) |
    (condition == "reasoning_off" & reasoning_default_enabled %in% FALSE)]
model_level <- d[, {
  s <- if (any(is_default)) .SD[is_default == TRUE] else .SD
  c(list(developer = developer[1], region = as.character(region[1]), region_source = region_source[1],
         price_source = price_source[1],
         release_date = release_date[1], input_price = input_price[1],
         default_source = if (any(is_default)) "default setting" else "median of conditions",
         n_conditions = .N, n_runs = sum(s$n_runs),
         probability = median(s$probability), upper_bound = median(s$upper_bound),
         lower_bound = median(s$lower_bound)),
    lapply(s[, paste0("share_", CAUSES), with = FALSE], mean))
}, by = model]
fwrite(model_level, file.path(OUT, "model_level.csv"))

# Convergence between models: beta model of one value per model (default setting), with the
# precision depending on release date. (The precision in M1-M3 is not this: there the random
# intercept per model absorbs the differences between models.)
conv_d <- copy(model_level)
conv_d[, release_years := as.numeric(as.IDate(release_date) - as.IDate("2025-01-01")) / 365.25]
stopifnot(all(conv_d$probability > 0 & conv_d$probability < 1))
fit_conv <- glmmTMB(probability ~ release_years + (1 | developer), dispformula = ~ release_years,
                    family = beta_family(link = "logit"), data = conv_d)
r_conv <- summary(fit_conv)$coefficients$disp["release_years", ]
fwrite(data.table(term = "release_years (per year)", precision_ratio = exp(r_conv[1]),
                  ci_low = exp(r_conv[1] - 1.96 * r_conv[2]), ci_high = exp(r_conv[1] + 1.96 * r_conv[2]),
                  p = r_conv[4], n_models = nrow(conv_d), pd_hessian = fit_conv$sdr$pdHess),
       file.path(OUT, "convergence.csv"))

# ---------------------------------------------------------------- console summary
cat("\nConvergence:\n"); print(fit_tab[, .(model, converged, pd_hessian)])
show <- coef_tab[part == "mean (logit)" & !grepl("^A2", model) & term != "(Intercept)",
                 .(model, label, OR = round(exp_estimate, 2), low = round(exp_low, 2),
                   high = round(exp_high, 2), p = signif(p, 2))]
print(show, nrows = 300)
cat("\nVariance components:\n"); print(vc_tab[grepl("^M", model)])
cat("\nTests:\n"); print(tests)

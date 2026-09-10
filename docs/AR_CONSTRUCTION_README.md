# Building Abnormal Returns: FF5 + Momentum with rolling betas

**What this stage does.** Turn your USD excess returns into abnormal returns (AR) — the dependent variable for the event regressions. AR = realized excess return minus the return predicted by a firm's factor loadings. The loadings (betas) aren't in your data; they're estimated by regressing each firm's returns on the factors.

**The core design choice (already decided).** Betas are estimated on a **trailing rolling window**, not the full sample. A single full-sample fit uses 2013–2025 data to benchmark a 2018 event — future information contaminating a past expectation, which is fatal for an event study. Rolling betas use only data *before* each day, so every AR is a surprise relative to expectations that could actually have been formed at the time.

**The trick that makes it work.** Your 2010–2012 data — contaminated for *analysis* (backloading, Phase 2/3 break) — is clean enough to *warm up* the rolling estimator. Factor loadings are a structural property; estimating them on 2010–2012 just gives the window history to chew on so betas are warm by your first event. So you **estimate on 2010–2025, then trim AR to 2013+**. The early years feed the estimator and never enter your results.

---

##window length

The window sets how much warm-up you need, which sets when  AR series begins.

| Window | Warm by | Covers all 27 events? | Notes |
|---|---|---|---|
| **48 months** | ~Jan 2014 | **Yes** (with 2010 start) | Recommended — keeps the Jan-2014 proposal event |
| 60 months | ~Jan 2015 | No — loses event 1 (2014) | Matches the proposal exactly |
| 24 months | ~Jan 2012 | Yes, comfortably | Shorter, noisier betas |



(Window is counted in **trading days**: 48 months ≈ 48 × 21 ≈ **1,008 trading days**.)

---

## Step 1 — Align the inputs


**End state of Step 1:** one panel, 2010–2025, with `excess_return_usd` and all six factors, no missing rates.

---

## Step 2 — Estimate rolling betas (per firm)

Use `statsmodels` `RollingOLS` — vectorized per firm, so this runs in minutes, not the hours a naive day-by-day loop would take.

- Sort by `(firm, date)`. For each firm, regress `excess_return_usd` on the six factors over the trailing 1,008-day window.
- **Lag the betas one day within each firm** before predicting, so day *t*'s expected return uses betas estimated through *t−1* — strictly out-of-sample. (Over a 1,008-day window the influence of any single day is negligible, but the lag makes the construction clean and a reviewer can't object.)
- Firms thin the estimator honestly: a firm that entered in 2019 has no 2010 warm-up, so its AR starts ~48 months after *its own* entry. That's correct — you can't estimate a beta without history — but it means younger firms contribute fewer AR rows.

---

## Step 3 — Compute AR

For each firm-day with warm betas:

```
predicted_excess = α + β_mkt·Mkt-RF + β_smb·SMB + β_hml·HML
                     + β_rmw·RMW + β_cma·CMA + β_mom·Mom        (using day-t factors, lagged betas)
AR = excess_return_usd − predicted_excess
```

Days inside the warm-up (no full prior window) get `AR = NaN` and drop out — that's the warm-up cost, not a bug.

---

## Step 4 — Trim to the analysis window and validate

**Trim AFTER computing AR**, not before: keep `date >= 2013-01-01`. This is where the 2010–2012 years leave the sample — they did their job as warm-up.

Validation checks before building anything on top:
- `AR.mean()` ≈ 0 and roughly symmetric (a large nonzero mean signals a construction error).
- **Coverage:** how many firm-days have a non-NaN AR, and from which date the series effectively starts. Confirm your 2014 events fall *after* the warm-up boundary (the reason for the 48-month choice).
- Spot-check one known firm (e.g., Shell): its market beta should be sensible (~1, higher for a cyclical), and AR should spike on its own earnings dates, not be flat.
- Confirm no AR rows sit in 2010–2012.

---

## Step 5 — Hand off to the event regression

This is where the AR stage rejoins the main pipeline (the implementation README).

- Merge AR onto the panel and attach the signed treatment from `channel_signs` (the ±1 trading-day event windows — confirmed overlap-free, so no dropping needed).

- Run the regressions: 

Everything downstream — carbon measure (z-score vs quartile), time-varying controls, robustness battery — is Step 5 of the implementation README and unchanged.

---

### Things to keep in mind

- **Warm-up means AR starts later than returns.** Expected; the 2010 head start is what keeps it from eating your early events.
- **Momentum is now baked into the betas.** Your methods section should say FF5+Mom (six-factor model); the panel and the equation must match.
- **`min_nobs` / trading gaps.** Requiring the *full* 1,008-day window is cleanest but a firm with sporadic trading may never warm up; relaxing to ~80% of the window recovers them.

- **Re-running FX on the full sample (1b) is the one easy-to-forget step.** Skip it and the rolling estimator has no warm-up and silently starts your AR in ~2017 instead of 2014.

### Dependency map

```
Step 1 (momentum merge + full-sample FX)
   └─> Step 2 (rolling FF5+Mom betas, 2010–2025, lagged)
          └─> Step 3 (AR = excess − predicted)
                 └─> Step 4 (trim to 2013+, validate)
                        └─> Step 5 (signed-event regressions — main README)
```

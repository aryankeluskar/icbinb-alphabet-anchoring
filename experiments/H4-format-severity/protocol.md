# H4 — Edit Distance vs Biochemical Severity

**Status:** LOCKED (pre-registered before any result was observed)
**Date locked:** 2026-08-05
**Type:** CONFIRMATORY
**Priority:** 2 — the most open angle in the novelty ledger (`literature/survey.md`, Angle D).

## The claim

Zhao et al.'s third axis is *format*: they perturb prompts by inserting, deleting and
modifying tokens, and show performance decays with the amount of perturbation. In their
synthetic world those operations are **semantically empty noise** — there is no sense in
which one insertion is biochemically worse than another.

Proteins are different. A substitution is a real evolutionary operator with a known fitness
distribution. So the format axis splits into two variables that are conflated in the LLM
setting and *separable* here:

- **edit distance** — how many positions changed (k). A pure pattern-matcher should degrade
  with this, because each edit moves the string further from anything it memorised.
- **biochemical severity** — how bad each change is (BLOSUM62). A model using a mechanism
  should degrade with this, because that is what actually determines fitness.

**H4:** pLM error is driven predominantly by **k**; alignment-based model error is driven
predominantly by **severity**.

This is the head-to-head that PLM-GUARD gestures at — it colours its figures by BLOSUM62 —
but never runs as a regression.

## Why this is cheap and unblocked

ProteinGym ships **95 models × 217 assays of precomputed zero-shot scores**, and 72% of the
substitution benchmark is multi-mutant (1.77 M rows, 69 assays, max k = 44). So the whole
experiment is a CPU-only regression over data already on disk. No GPU, no re-scoring — the
data agent verified row counts match the DMS files exactly and that scores are populated at
k = 7.

## Design

**Unit of analysis:** one variant within one assay.

**DV — per-variant rank error.** Within each assay, convert both the DMS score and the model
score to percentile ranks, and take `err = |rank_model − rank_DMS|`. Ranking within assay
removes all cross-assay scale differences and makes assays commensurable.

**IVs, chosen specifically to avoid collinearity:**
- `k` — number of substituted positions.
- `mean_sev` — mean per-substitution severity, `mean(−BLOSUM62(wt_i, mut_i))`.

Note the deliberate choice: **total** severity is ≈ `k × mean_sev` and therefore almost
perfectly collinear with k, which would make the regression uninterpretable. Using k and
*mean* severity keeps the two IVs close to orthogonal, which is the whole point of the
design. We report their empirical correlation; if |r| > 0.5 within an assay, that assay is
excluded and the exclusion is reported.

**Covariates:** position conservation is *not* included in the primary model. It is a
mediator of severity rather than a competitor to it, and conditioning on a mediator would
bias the severity coefficient downward. Reported as a secondary model only.

**Model:** per assay, standardise `k`, `mean_sev` and `err`, then OLS
`err ~ β_k · k + β_sev · mean_sev`. Because both IVs are standardised, β_k and β_sev are
directly comparable *within* an assay.

**Primary statistic:** `D = |β_k| − |β_sev|`. Positive ⇒ edit-distance-driven.
Aggregated across assays by median with a bootstrap CI, and compared **between model
families** (pLM vs alignment-based) with a Mann–Whitney U test on paired-by-assay values.

## Registered predictions

1. `D > 0` for ESM-2 at every size, and `D < 0` for GEMME / EVE / EVmutation /
   Site_Independent.
2. The pLM-vs-alignment difference in `D` is significant (Mann–Whitney, α = 0.05).
3. `D` does **not** decrease with ESM-2 model size — scale does not convert a
   pattern-matcher into a mechanism-user. (Note: per the survey, ESM-2 *saturates* at ~650M;
   do not describe any of this as inverse scaling.)

## Falsification

- If `D < 0` for pLMs, H4 is refuted: pLM error tracks biochemistry, and we report that as a
  positive result about pLMs.
- If `D` is indistinguishable between pLMs and alignment models, the axis does not separate
  the families and H4 is uninformative — which we must say plainly rather than reporting the
  pLM sign alone.
- If k and mean_sev are too collinear across most assays, the design fails and no
  conclusion may be drawn from it.

## Controls

- **C1 — collinearity report.** Empirical corr(k, mean_sev) per assay; exclude |r| > 0.5.
- **C2 — shuffled-severity null.** Permute `mean_sev` within assay. β_sev must collapse to
  ~0; if it does not, the regression is picking up something structural.
- **C3 — singles-only sanity.** Restrict to k = 1, where the k term is constant and must
  drop out. Any residual β_k indicates a coding error.
- **C4 — BLOSUM sign convention assert.** BLOSUM62 gives *higher* scores to *more similar*
  pairs, so severity is its negation. Assert `sev(K→R) < sev(K→W)` before running anything;
  getting this backwards would silently invert every conclusion.
- **C5 — assay-count report.** State how many of the 69 multi-mutant assays survive
  filtering. The inventory warns that 58 of 69 are k=2-only and **two assays hold 58% of all
  multi-mutant rows**, so any unweighted row-level average is really an average over two
  proteins. All aggregation is therefore **per-assay, never per-row**.

## Amendment 1 — 2026-08-05, aggregated DV (registered after the primary null, before running)

The primary analysis above returned a clean null (see `analysis.md`) with two diagnosed
causes: (i) 58 of 69 assays have k_max = 2, so `k` has almost no within-assay variance, and
(ii) per-variant `|rank_model − rank_DMS|` is dominated by DMS measurement noise, leaving
both standardised coefficients at ≈ 0.05–0.10.

**The primary null stands and is reported as the registered result.** This amendment tests a
differently-powered DV; any outcome is reported as a *separate, secondary* analysis and can
never overwrite the primary null.

**Amended DV — within-stratum Spearman.** Partition each assay's variants into cells by
(k × severity tercile). For each cell with ≥ 50 variants compute Spearman(model, DMS) inside
the cell, then regress

    cell_spearman ~ β_k · z(k) + β_sev · z(mean_sev)

Rank correlation over ≥ 50 variants is the standard ProteinGym metric and is far more stable
than a single variant's rank error, so this attacks cause (ii) directly. It does not fix
cause (i), so assays with k_max = 2 remain weakly informative about β_k and are reported
separately from k-rich assays.

**Requires ≥ 6 usable cells per assay** for a 2-predictor fit; assays below that are dropped
and the count reported.

**Registered predictions:** unchanged in sign — D > 0 for pLMs, D < 0 for alignment models,
with a significant family difference.

**Falsification, restated so it cannot be evaded:** if this analysis is *also* null, H4 is
abandoned rather than re-cut a third time. Two differently-powered designs returning null on
the same registered prediction is a negative result about the hypothesis, not about the
method, and the paper will report it that way.

## Deliverable

`results/h4_severity.csv` (`assay, model, beta_k, beta_sev, D, n, corr_k_sev`) plus
`analysis.md`.

# H4 — Analysis: the registered design is uninformative

**Date:** 2026-08-05
**Protocol:** `protocol.md`, locked at commit `1e2ce0f` before this run.
**Status:** **NEGATIVE / UNINFORMATIVE.** Reported under the falsification condition
registered in the protocol, not discovered after the fact.

---

## Result

Primary registered analysis, `D = |β_k| − |β_sev|` over all 69 multi-mutant assays:

**Every model's 95% CI includes zero, and there is no separation between the pLM and
alignment families.** Median D ranges from −0.019 (ESM2_650M) to +0.030 (ProtGPT2); the
pLM-vs-alignment Mann–Whitney gives z = 1.10, far short of significance.

The protocol registered exactly this outcome as a failure mode:

> If `D` is indistinguishable between pLMs and alignment models, the axis does not separate
> the families and H4 is uninformative — which we must say plainly rather than reporting the
> pLM sign alone.

So that is what we say. **H4 is not supported.**

## The controls rule out the boring explanations

- **C4 (sign convention)** passed before anything ran: sev(K→R) = −2 < sev(K→W) = +3.
- **C2 (shuffled-severity null)** passed cleanly: mean |β_sev| = 0.0158 under shuffling
  versus 0.0984 for real severity, a 6× gap. **The severity term is measuring something
  real.** The null is not an artefact of a broken regression.
- **C1 (collinearity)** removed nothing: 0 of 69 assays exceeded |r| = 0.5 between k and
  mean severity. The two IVs are genuinely near-orthogonal, exactly as the design intended.

So the pipeline works and the covariates are sound. The null is a null.

## Why it is null — a design flaw I should have caught

**58 of the 69 multi-mutant assays have k_max = 2.** Within those assays k ranges over
{1, 2} and nothing else, so β_k is estimated from almost no variance. Only **11 of 69**
assays have k_max ≥ 4.

This was knowable in advance: `data/MULTI_MUTANT_INVENTORY.md`, which I had already read,
states plainly that 58 of 69 assays are k=2-only. The protocol's C1 control checked
*collinearity* between the two IVs but never checked whether either IV had enough
**variance** to support a coefficient. That is the gap.

Restricting to the 11 k-rich assays (EXPLORATORY, not registered) does not rescue it:

| | median D | Mann–Whitney |
|---|---|---|
| pLM (n=121) | +0.0048 | z = 1.10, n.s. |
| alignment (n=66) | +0.0019 | |

Every individual model's CI still includes zero. There is a *hint* in the predicted
direction — in the k-rich subset mean |β_k| = 0.064 versus mean |β_sev| = 0.049 — but it is
not distinguishable from noise and must not be reported as support.

## The second, deeper problem: the DV is too noisy

Both standardised coefficients are ≈ 0.05–0.10, meaning the regression explains on the
order of 1% of variance in per-variant rank error. That is expected in hindsight: a single
variant's `|rank_model − rank_DMS|` is dominated by DMS assay measurement noise, and no
covariate should explain much of it.

This is a power problem, not just a variance-in-k problem, and it would have limited the
design even with a better k distribution.

## What this rules out and what it suggests

**Rules out:** the specific claim that per-variant error is driven more by edit distance for
pLMs than for alignment models, at least at the effect size this data can resolve. If that
difference exists it is smaller than ~0.05 standardised units.

**Suggests:** aggregate the DV before regressing. Rank correlation computed *within a
stratum* of ~50+ variants is the standard ProteinGym metric and is far more stable than
per-variant error. Registered as Amendment 1 and run separately.

## Addendum — Amendment 1 refuted H4, and the artifact is confirmed

**Amendment 1 (within-cell Spearman) fixed the power problem and then refuted the
hypothesis.** Effect sizes rose from ≈0.005 to ≈0.3 and most CIs excluded zero — but in the
wrong pattern:

| | median D | |
|---|---|---|
| pLM | +0.270 | |
| alignment | +0.242 | Mann–Whitney z = 1.85, n.s. |

H4 predicted `D > 0` for pLMs and `D < 0` for alignment models. Instead `D > 0` for
*everything*, including GEMME (+0.344), DeepSequence (+0.284) and EVE (+0.223) — precisely
the models that were supposed to carry the opposite sign. In the k-rich subset the family
difference reverses (z = −0.29).

**The universal D > 0 is a data artifact, now confirmed.** A k effect identical across pLMs
and alignment models is better explained by the DV than by anything models do. Direct test
(`code/diag_range_restriction.py`, output in `results/diag_range_restriction.txt`), over all
69 multi-mutant assays:

    corr(k, within-k DMS std): median -1.000, mean -0.521
    negative in 77% of assays

DMS dynamic range **shrinks as k grows** — higher-order mutants are more often uniformly
dead. Range restriction mechanically depresses within-cell Spearman at high k for *every*
scorer, which reproduces the observed `D > 0` without any model-specific explanation.

So the aggregated design measures a property of ProteinGym, not of protein language models.

**Verdict: H4 is abandoned**, per the falsification clause registered in Amendment 1 — two
differently-powered designs failing the same registered prediction is evidence about the
hypothesis, not about the method. Re-cutting it a third time would be fishing.

**A further caution this generates for the rest of the project:** any analysis that
stratifies ProteinGym by mutation depth inherits this range-restriction confound. That
includes the `Depth_k` columns widely cited as evidence about higher-order epistasis, and it
compounds the separate problem that ProteinGym's scoring protocol is additive by
construction (findings.md F5). Both must be controlled for before H3 is attempted.

## Honest scope note

Compute constraints mean this ran on the 17 models named in the registered predictions
rather than all 95, with each assay capped at 8000 rows by a seeded subsample (2.4M rows ×
95 models is ~228M float conversions, which the stock Python 3.6 interpreter cannot do in
reasonable time). Neither choice affects the conclusion: the null is uniform across all 17
models and all 69 assays, and subsampling costs precision on β, not validity.

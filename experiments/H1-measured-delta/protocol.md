# H1 — Performance Tracks *Measured* Distribution Discrepancy, Not Corpus Size

**Status:** LOCKED (pre-registered before any result was observed)
**Date locked:** 2026-08-05
**Type:** CONFIRMATORY
**Priority:** 1 — this is the paper's central novelty claim.

## The claim

Zhao et al.'s Limitation (ii) states, in their own words, that they cannot estimate the
distribution discrepancy between pretraining data and test queries "due to the opaque nature
of training data and model weights", which limits "the precision with which our data
distribution lens can be quantitatively validated in fully realistic and transparent
scenarios." Their workaround is to **construct** a known-unseen distribution (DataAlchemy)
rather than **estimate** an unknown one.

Protein modelling does not have this problem. UniProt records a creation date for every
accession, and AMPLIFY released per-year checkpoints with architecture held constant. So Δ
becomes a *measured*, per-test-item, year-resolved quantity.

**H1:** A pLM's zero-shot fitness-prediction performance on a given protein is a function of
the **measured discrepancy Δ between that protein and the training corpus**, not of corpus
size or calendar year. Conditioning on Δ should absorb the year effect.

## Why this is novel given prior work

The Spinner paper already reports a performance-vs-training-year trend on ProteinGym and
already stratifies it by Neff/L (their Fig. S1). We must not re-derive that. The distinction:

| | Spinner Fig. S1 | H1 |
|---|---|---|
| Explanatory variable | Neff/L of the **present-day** MSA | Homolog count **as of year Y** |
| Resolution | per-assay, time-invariant | per-assay **× per-year** |
| Question | do deep-MSA proteins benefit more? | does Δ **screen off** year? |

Their covariate is a static property of the protein. Ours is a property of the
protein–corpus *pair*, and it changes as the corpus grows. That is what makes it Δ in Zhao
et al.'s sense rather than a difficulty covariate, and it is what licenses the screening-off
test below, which they do not run.

## The measurement trick (avoids a 419 GB download)

Historical UniRef ships as one monolithic XML tarball per release — roughly 419 GB for four
year-points, and UniRef50 cannot be fetched alone. We do not need it.

ProteinGym ships 217 MSAs. Every homolog in an MSA is a UniProt accession, and UniProt
exposes each accession's **integration date** via its REST API. Therefore:

    Δ(protein p, year Y) = f({homologs of p whose UniProt entry existed on or before Y})

computed directly from data already on disk plus a metadata lookup. No historical corpus
snapshot is required.

**Primary Δ:** `n_homologs(p, Y)` = count of MSA members with integration date ≤ Y.
**Secondary Δ:** `neff(p, Y)` = 80%-identity-clustered effective count over that subset,
which controls for redundancy — the difference between "many near-identical sequences" and
"genuine diversity" is exactly what a coverage measure should capture.
**Tertiary Δ:** `max_id(p, Y)` = maximum sequence identity between the query and any homolog
available by year Y. This is the most direct analogue of Zhao et al.'s Δ: nearest-neighbour
distance to the training distribution.

## Design

**Models:** AMPLIFY_120M, HuggingFace `chandar-lab/AMPLIFY_120M`, `revision=AMPLIFY_120M_<YEAR>`
for YEAR ∈ {2011 … 2024}. Architecture, parameter count and step count are held constant;
only the corpus changes.

**Task:** ProteinGym zero-shot substitution DMS, Spearman per assay. Restrict to assays whose
query maps to a UniProt accession we can date.

**Unit of analysis:** (assay, year) pair.

### The screening-off test — this is the actual experiment

Three nested mixed-effects models with a random intercept per assay:

```
M0:  spearman ~ year                      + (1 | assay)
M1:  spearman ~ log Δ                     + (1 | assay)
M2:  spearman ~ log Δ + year              + (1 | assay)
```

**Registered prediction:** M1 ≫ M0 in fit, and in M2 the coefficient on `year` is not
significantly different from zero while the coefficient on `log Δ` remains large and
positive. I.e. **Δ screens off calendar year.**

Reported: ΔAIC, marginal R², and the year coefficient with CI in M1 vs M2.

### Registered predictions

1. Assays whose homolog count was already saturated by 2011 show **no improvement** from
   2011 → 2024 checkpoints.
2. Assays whose homologs mostly entered after 2015 show the **steepest** gains.
3. Δ screens off year (above).
4. The residual year effect after conditioning on Δ is small relative to the Δ effect.

### Falsification

- If year remains significant after conditioning on Δ, H1 is refuted: something about corpus
  growth beyond per-protein coverage is driving performance, and we report that.
- If Δ and year are too collinear to separate, the test is **uninformative rather than
  supportive**, and we must say so. Pre-registering this matters: collinearity is the most
  likely way this experiment fails to answer its question, and it would be easy to
  misreport a null as support.

## Known confounds — all must be reported

- **Steps vs corpus.** Yearly checkpoints run 250K steps while the corpus grows ~33×, so the
  2011 model sees far more epochs per sequence than the 2024 model. Epochs and Δ are
  therefore anti-correlated by construction. **Unfixable with released checkpoints; must be
  stated as a limitation, not buried.** It cuts against H1's prediction (more epochs should
  help the early models), so a confirmed H1 is In-class with respect to it.
- **120M only.** No yearly branches exist for AMPLIFY_350M, so no size sweep.
- **Yearly ≠ main.** Yearly branches are 250K steps, flagship is 1M. Only compare
  yearly-to-yearly.
- **fp32 is mandatory.** The yearly branches ship the pre-fix `rmsnorm.py`, and
  `trust_remote_code` loads code from the *same* revision. This diverges silently under
  bf16/fp16 with **no error raised**. Every run must assert fp32.
- **UniProt integration date ≠ sequence discovery date.** An entry can be re-integrated or
  migrated. We use the earliest available date field and report the field used.
- **ProteinGym assay leakage into AMPLIFY's corpus** is possible for later years; DMS
  *measurements* are not in UniProt but the wild-type sequences are.

## Controls

- **C1 — architecture invariance.** All checkpoints are the same architecture; verify
  parameter count and config hash identical across revisions.
- **C2 — fp32 assert.** Load in fp32, assert dtype, and verify a fixed reference sequence
  gives identical logits across two loads.
- **C3 — shuffled-Δ null.** Permute the Δ values across assays. The screening-off effect must
  vanish; if it survives, the model is fitting something structural rather than Δ.
- **C4 — reproduce the known trend first.** Before testing H1, reproduce the plain
  performance-vs-year trend. If we cannot recover the published trend, our pipeline is wrong
  and nothing downstream is interpretable.

## Deliverable

`results/h1_delta.csv` with `assay, year, spearman, n_homologs, neff, max_id, n_muts`
plus `analysis.md`.

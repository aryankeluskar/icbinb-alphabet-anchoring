# H2 — Control Run Analysis

**Date:** 2026-08-05
**Protocol:** `protocol.md`, locked at commit `d296f37` before this run.
**Script:** `code/test_mrf_invariance.py`
**Raw output:** `results/controls_C0_C2_C3.txt`
**Status:** CONFIRMATORY for C0/C2/C3/C5; C3b is EXPLORATORY.

---

## What this run was for

H2 rests on a mathematical premise: a Potts/MRF coevolution model is **exactly invariant**
under a bijection over the amino-acid alphabet applied consistently across a whole MSA. If
that premise fails — or if our implementation breaks it — the experiment is void, because we
would have no invariant reference against which to measure ESM-2's degradation.

The protocol lists this as control C2 and explicitly frames it as *a hard correctness assert,
not a finding*: there is no statistical route to non-invariance, so a failure here means our
code is wrong. Verifying it on synthetic data costs seconds and gates hours of GPU time.

## Setup

Synthetic MSAs, N=2000 sequences, L=60 columns. Background columns are i.i.d. uniform over
the 20 residues. Fifteen long-range contacts (all |i−j| ≥ 24) are planted by coupling column
j to column i through a fixed random bijection applied with probability 0.9. This produces
genuine pairwise mutual information at exactly the planted positions and nowhere else — the
cleanest available test bed, and one where the ground truth is known by construction rather
than inferred from a structure.

Contact scores: inverse-covariance couplings (Dauparas et al. formulation, λ = 4.5/√N),
Frobenius norm over each 20×20 residue block with the gap state excluded, APC, symmetrised.

## Results

### C0 — the metric can see the signal

recall@L/2 = **1.000**; precision@L/2 = 0.500, which is exactly its ceiling (15 true
contacts / 30 predictions). Chance is 0.0085. All fifteen planted contacts are recovered.

This is worth stating carefully because it caused an error earlier in the session:
**precision@L/k is ceilinged at n_true/n_pred**, so a "0.20" that looks like a weak result
can be perfect recovery. Recall is now the primary control metric and `precision_ceiling()`
is reported alongside precision everywhere.

### C2 — invariance holds exactly

Twelve conditions (3 regimes × m ∈ {2, 6, 10, 20}):

`max|ΔS|` ranges from **3.55e-15 to 5.33e-15** — float64 round-off on scores of order 1.
Recall stays at 1.000 in every condition. Composition check (C5) passes everywhere: a
consistent permutation permutes the residue counts but preserves the multiset.

The residual is not a small effect to be explained; it is the accumulated rounding of a
1260×1260 matrix inversion. The invariance is exact.

**This is independently a theorem, not just a measurement.** Zhang, Wayment-Steele, Brixi &
Ovchinnikov state in the PNAS 2024 text that applying the categorical Jacobian to an MRF or
multivariate-Gaussian model *exactly returns* its coupling tensor W. A consistent relabeling
permutes W's (n,m) alphabet indices; the Frobenius norm over each 20×20 block and the APC
are both invariant to that permutation. So the result follows from their own equations. The
numerical run verifies our implementation rather than establishing the mathematics — which
is precisely what a correctness gate should do.

### C3 — the metric can also detect destruction

Per-column row shuffle (independently permute rows within each column): recall **0.000**,
down from 1.000. Marginals preserved exactly, all inter-column dependency destroyed.

C3 matters as much as C2. Without it, "invariance" would be consistent with a metric that
cannot see anything at all.

### C3b — a per-sequence permutation is not a null

Recall **0.600**, versus 1.000 unperturbed and 0.000 for the true null.

The protocol originally specified a *per-sequence* permutation as the destruction control.
That was wrong, and the error is instructive. Applying a different bijection to each sequence
relabels **both columns of a row identically**. Mutual information between two columns is
invariant under a bijection applied to whole rows, so this operation cannot destroy
coevolution — it only blurs it, by mixing differently-conjugated couplings across rows.

Consequences:

1. The true null is the per-column row shuffle, now C3.
2. The per-sequence case is retained as a **third experimental condition**, not deleted. It
   sits at a genuinely intermediate point (0.600) and asks a real question: the MRF partially
   survives row-wise relabeling — does ESM-2?
3. Had this stood uncorrected, we would have calibrated the pLM comparison against a
   partially-informative condition labelled "destroyed", and the headline comparison would
   have been meaningless.

## Verdict

All controls pass. The H2 premise is verified and the pipeline is trustworthy. Safe to
proceed to ESM-2 once the GPU environment lands.

## What this does and does not establish

- **Does:** our MRF implementation is correct, our metric has both sensitivity and
  specificity, and there is an exactly-invariant reference baseline for the real experiment.
- **Does not:** say anything whatsoever about ESM-2. The interesting measurement has not been
  made yet. This run only guarantees that when it is made, we will be able to interpret it.

## Next

1. Real MSAs (GREMLIN PDB_EXP set) in place of synthetic — confirms the invariance survives
   real gap structure, non-standard residues and phylogenetic correlation.
2. ESM-2 categorical Jacobian under the same six values of m × three regimes.
3. Register the `Cross-class` vs `In-class` contrast at matched m as the mechanism probe:
   if they hurt ESM-2 *equally*, that is the stronger mirage result — it means ESM-2 is keyed
   on token identity and not on biochemistry at all.

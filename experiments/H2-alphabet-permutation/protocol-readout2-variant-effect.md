# H2, second readout — variant effect under alphabet relabeling

**Locked 2026-08-05, before any run.** Separate file so the original H2 protocol stays as
committed; this is a new dependent variable for the same hypothesis, not an amendment to
the old one.

## Why a second readout is needed

Everything measured so far is contact prediction. A reviewer is entitled to ask whether
alphabet non-invariance is a quirk of the categorical-Jacobian contact readout rather than
a property of the representation. If the same relabeling that destroys contacts also
destroys **zero-shot variant effect prediction** — a completely different head, a different
downstream task, and the field's most-used pLM benchmark — the claim generalises. If it does
not, the claim must be narrowed to structure, and we say so.

## The transformation, restated for this task

A consistent bijection π over the 20 amino acids is applied to the wild-type sequence, to
the alignment, and **to the variants themselves**. A variant `K41M` becomes `π(K) 41 π(M)`.
The measured fitness value is untouched: it is a wet-lab number attached to a physical
protein, and we are only renaming the letters we use to describe it.

A model that had learned an alphabet-abstract computation would score π(K)→π(M) on π(WT)
exactly as it scores K→M on WT, so its Spearman against the DMS values would be unchanged.

## Design

**Assays.** Seeded random sample (`numpy.default_rng(0)`) of 15 from the ProteinGym
substitution assays satisfying all of: an `.a2m` MSA exists on disk; WT length ≤ 400; at
least 200 single-mutant rows. Sampling rule fixed here, before looking at any result, so the
set cannot be tuned. All 15 are reported whatever they show.

**Single mutants only.** This deliberately sidesteps the additive-scoring confound recorded
in findings.md F5 and the range-restriction artifact in F8 — both are properties of
multi-mutant scoring. Nothing in this readout depends on either.

**Model scoring.** ESM-2 650M masked marginals, ProteinGym's standard protocol: mask the
mutated position, take `log p(mutant) − log p(wildtype)` at that position. Computed on the
permuted sequence with permuted residue indices.

**Baseline.** A site-independent weighted PSSM built from the same `.a2m` alignment. Chosen
because ICBINB asks for unexpectedly strong simple baselines, because it is one of
ProteinGym's own reference models, and because it is *exactly* invariant to a consistent
relabeling for the same reason the Potts model is — the alignment statistics are unchanged
and only the index labels move. The invariance is asserted numerically, not assumed.

**DV.** Spearman ρ between model score and `DMS_score`, per assay. Reported as
Δρ = ρ(permuted) − ρ(identity), aggregated with SEM over **assays**, not over rows.

**Conditions.** regimes {random, In-class, Cross-class} × m ∈ {2, 6, 10, 20} × 3 seeds,
plus the identity control. Note protocol Amendment 7: In-class and Cross-class saturate near
m=15, so those points are plotted at their realized m.

## Validity check that must pass before any result is believed

Our identity-condition ESM-2 Spearman must reproduce the published `ESM2_650M` column in
`data/proteingym/zero_shot_substitutions_scores` for the same assays, to within rank-tie
noise. We have those scores on disk, so this is free. **If it does not reproduce, the
pipeline is wrong and no permuted number means anything.** This is reported first, as a
pass/fail, regardless of outcome.

## Registered predictions

1. **PSSM Δρ = 0** to floating-point precision at every m. (Theorem, per findings.md F1;
   measured here as a correctness assert.)
2. **ESM-2 Δρ becomes strongly negative under `random` relabeling, monotonically in m.**
3. **In-class degrades less than random at matched realized m** — the same dissociation
   seen in contacts, indicating biochemistry is encoded but alphabet abstraction is not.

## Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| ESM-2 Δρ strongly negative, PSSM exactly 0 | The claim generalises beyond structure to the field's main benchmark. This is the strong outcome. |
| ESM-2 Δρ ≈ 0 at large m | **Prediction 2 refuted.** Alphabet non-invariance is specific to the contact readout and the paper's claim must be narrowed to structure prediction. Report plainly; do not re-cut the readout. |
| ESM-2 degrades but In-class ≈ random | Prediction 3 alone fails. The generalisation holds; the biochemistry story does not extend to this task. |
| Identity-condition ρ does not match published ESM2_650M | Pipeline bug. Nothing is reported until it is fixed. |

**Falsification clause.** As with H4, two readouts are enough. If this one refutes
prediction 2, the paper narrows its claim to contact prediction — it does not go looking for
a third readout that agrees with the first.

---

## Amendment 9 — the aggregate Δρ is confounded by assays with no identity signal (2026-08-05)

**Registered after seeing 3 of 15 identity rows and 0 complete permuted assays beyond the
first two.** Stating that plainly because the amendment is not free of hindsight and a
reader is entitled to know how much I had seen. What I had seen is below; what the
amendment changes is the aggregation rule, which no cell yet observed can determine.

### The problem

The sampling filter was: an MSA exists, WT ≤ 400 residues, ≥ 200 single mutants. It does
**not** require that ESM-2 has any predictive signal on the assay to begin with. The first
three identity rows are:

| assay | n | ESM-2 ρ (identity) | published `ESM2_650M` | gate |
|---|---|---|---|---|
| A0A1I9GEU1_NEIME_Kennouche_2019 | 922 | +0.0297 | +0.0297 | PASS |
| (assay 2) | — | +0.0659 | +0.0659 | PASS |
| (assay 3) | — | +0.7382 | +0.7382 | PASS |

Two of three have effectively **no** identity-condition signal. The gate passes — these are
the numbers ProteinGym itself publishes — so this is not a pipeline bug. It is a property of
the benchmark: ESM-2 650M is near-uncorrelated on a nontrivial fraction of assays.

An absolute Δρ averaged over such a sample is close to meaningless. On an assay with
ρ_identity = 0.03 there is nothing to destroy, so Δρ is bounded near zero by construction
and drags the mean toward zero; and with |Δρ| ≈ 0.1 already observed elsewhere, a permuted
ρ can even sit *above* the identity ρ by noise alone. **This is structurally the same defect
that made H4's registered DV uninformative** (range restriction: a DV whose achievable
magnitude varies systematically across units). Having paid for that lesson once, the
aggregation rule is fixed here rather than after the fact.

### The amended DV

Report all three, in this order:

1. **Pre-registered, unchanged**: mean absolute Δρ over all 15 assays. Reported first,
   whatever it says. The amendment adds analyses; it does not retire the locked one.
2. **Signal-gated absolute Δρ**: mean Δρ over assays with **ρ_identity ≥ 0.20**. The
   threshold is set now, before knowing which assays clear it. It is ≈ 3 standard errors
   for a Spearman correlation at n ≥ 200 (SE ≈ 1/√(n−1) ≈ 0.07), i.e. the smallest identity
   correlation that is distinguishable from zero at all. It is **not** tuned to the observed
   values and must not be moved after the run.
3. **Relative destruction**, on the same gated subset: `Δρ / ρ_identity`, per assay, then
   averaged. This is the quantity the paper actually wants — *what fraction of ESM-2's
   variant-effect signal is alphabet-keyed* — and it is directly comparable to the contact
   readout, where the corresponding fraction is 1.00 (0.763 → 0.053 = chance, F13).

The number of assays clearing the gate is reported explicitly. **If fewer than 6 of the 15
clear it, the gated analysis is declared underpowered and only the pre-registered DV is
reported** — a small gated subset chosen by a threshold is exactly the kind of post-hoc
subgroup this amendment exists to avoid becoming.

### What this does not license

It does not license dropping low-signal assays from the *identity* validity gate: every
assay's gate result is reported, because that is what certifies the pipeline. It does not
license a fourth DV if these three disagree; the falsification clause above still stands.
And it does not change the registered predictions — only the statistic they are read from.

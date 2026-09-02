# H2 — Alphabet-Permutation Invariance

**Status:** LOCKED (pre-registered before any result was observed)
**Date locked:** 2026-08-05
**Type:** CONFIRMATORY

## Claim

Coevolution is a *statistic over an alignment*. If you apply a fixed bijection
$\pi:\mathcal{A}\to\mathcal{A}$ over the 20 amino acids **consistently across every sequence
in a family**, then:

- A Potts / MRF / multivariate-Gaussian coevolution model fitted to the permuted MSA
  recovers **exactly the same coupling structure**, up to the corresponding relabeling of
  the $20\times20$ blocks. Contact prediction is therefore **exactly invariant**:
  $\text{APC}(W_{\pi(\text{MSA})}) = \text{APC}(W_{\text{MSA}})$.
- ESM-2, if it has genuinely internalised the *algorithm* of coevolution, should likewise
  be near-invariant.
- ESM-2, if it has instead stored **motif → contact lookups keyed on literal residue
  identity** (which is precisely what Zhang, Wayment-Steele, Brixi & Ovchinnikov, PNAS
  2024 argue — their Hypothesis 3, supported by the finding that ~85–94 unmasked
  residues of *local* context suffice to recover a contact), should **collapse**.

This turns their descriptive claim into a falsifiable prediction. They did not run it.

## Why this is the right first experiment

- No training. No fine-tuning. Hours of GPU, not days.
- Fully controlled: the permutation changes *nothing* about the information-theoretic
  content of the alignment. Column entropies, mutual information between columns, and the
  true contact map are all **exactly preserved**. So any degradation cannot be explained
  by "the task got harder" — the standard confound that sinks OOD papers.
- It discriminates two mechanisms that ordinary benchmarks cannot separate.

## Design

**Independent variable:** $m$ = number of amino-acid types involved in the permutation,
$m \in \{0, 2, 4, 6, 10, 20\}$. $m=0$ is identity (control). Permutations are derangements
restricted to the $m$ selected types.

**Permutation regimes (secondary IV):**
1. `random` — $\pi$ sampled uniformly at random over the chosen $m$ types.
2. `In-class` — $\pi$ swaps only within BLOSUM62-similar groups (e.g. I↔V, K↔R,
   D↔E). Biochemistry approximately preserved.
3. `Cross-class` — $\pi$ swaps across biochemical classes (e.g. K↔D, I↔E). Biochemistry
   maximally violated.

Regimes 2 vs 3 separate "ESM-2 knows biochemistry" from "ESM-2 knows token identity".

**Dependent variable:** long-range (|i−j| ≥ 24) contact precision @ L/2, computed from:
- (a) the **categorical Jacobian** of ESM-2 (unsupervised — this is the key readout,
  because it is the like-for-like comparison against the MRF couplings);
- (b) ESM-2's supervised contact head (secondary, for reference);
- (c) the MRF/inverse-covariance model of Dauparas et al. (the invariance control).

Ground-truth contacts from the experimental structure: $C_\beta$ distance < 10 Å.

**Dataset:** the GREMLIN PDB_EXP set used by Zhang & Ovchinnikov (filtered: >1000 MSA
sequences, length 200–600, TM<0.5 redundancy filter, <50 missing residues). Target
n ≈ 150 proteins for the pilot, scaling to their full 1431 if the pilot is clean.

## Predictions (registered before running)

| Model | $m=0$ | $m=20$ random | Interpretation if observed |
|---|---|---|---|
| MRF (inverse covariance) | P | **P** (Δ = 0, exactly) | invariance holds — sanity check on our own pipeline |
| ESM-2 categorical Jacobian | P′ | **≪ P′**, toward chance | stores lookups, not the algorithm |
| ESM-2 contact head | P″ | ≪ P″ | same, via the supervised path |

**Primary prediction:** ESM-2 precision decays monotonically in $m$, while MRF precision is
flat at exactly its $m=0$ value.

**Secondary prediction:** at matched $m$, `Cross-class` hurts ESM-2 more than `In-class`.
If instead they hurt *equally*, that is the stronger mirage result — it means ESM-2 is
keyed on token identity and not on biochemistry at all.

## Falsification conditions — what would kill this hypothesis

- If ESM-2 is **also** near-invariant, H2 is refuted and ESM-2 is doing something more
  algorithmic than the lookup account predicts. This would be a genuinely interesting
  positive result and we would report it as such.
- If the **MRF is not exactly invariant** in our implementation, our pipeline is buggy
  (there is no statistical route to non-invariance). This is a hard correctness assert,
  not a finding: `assert np.allclose(contacts_perm, contacts_orig)`.
- If ESM-2 degrades identically under a **within-sequence shuffle control** (which
  destroys the alignment statistics too), then we have not isolated the permutation
  effect. Hence the control below.

## Controls

- **C1 — identity permutation.** $\pi = \text{id}$. Must reproduce baseline exactly.
- **C2 — MRF exact-invariance assert.** As above. Correctness gate on the pipeline.
- **C3 — inconsistent permutation.** Apply a *different* random $\pi$ to each sequence in
  the family. This destroys coevolution, so MRF **and** ESM-2 must both collapse. Confirms
  our metric can detect real signal loss and that C2's invariance is non-trivial.
- **C4 — single-sequence permutation.** Permute only the query, leaving the MSA alone.
  Isolates "ESM-2 is reading the query tokens" from "ESM-2 is reading family statistics".
- **C5 — composition control.** Report amino-acid composition before/after; permutation
  preserves the multiset of counts by construction, so any composition shift = bug.

## Confounds considered

- *Tokenizer artefacts:* ESM's vocabulary includes special tokens; $\pi$ must act only on
  the 20 standard residues, never on `<mask>`, `<cls>`, `<eos>`, `<pad>`, X, B, U, Z, O.
- *BOS/EOS sensitivity:* Zhang & Ovchinnikov had to replace BOS/EOS with mask tokens for
  contact recovery to work. We must fix this choice across conditions and report it.
- *Jacobian step size:* they show a *categorical* (not infinitesimal) perturbation is
  required. Use their released implementation rather than reimplementing.

## Amendment 1 — 2026-08-05, after the control run (C3 was not a null)

The locked text above is left unchanged; this amendment records a correction made *after*
running the controls but *before* any pLM measurement.

**C3 as originally specified is not a destruction control.** Applying a different random π
to each sequence relabels both columns of a row identically, and mutual information between
two columns is invariant under a bijection applied to whole rows. So an inconsistent
permutation does not destroy coevolution — it only blurs it by mixing differently-conjugated
couplings across rows. Measured: recall@L/2 = 0.600, versus 1.000 unperturbed and 0.000 for a
true null.

- **C3 (revised)** — per-**column** row shuffle: independently permute the rows within each
  column. Preserves every column's marginal exactly, destroys all inter-column dependency.
  This is the true coevolution null. Measured recall 0.000.
- **C3b (new, promoted from the old C3)** — per-**sequence** permutation. Retained not as a
  control but as a **third experimental condition**, since it sits at a genuine intermediate
  point and poses its own question: the MRF partially survives row-wise relabeling; does
  ESM-2?

No prediction registered above is weakened by this change — the primary and secondary
predictions concern C1/C2 and the m × regime grid, which are untouched. The falsification
condition that referenced "a within-sequence shuffle control" now refers to C3 (revised).

## Amendment 2 — 2026-08-05, prior art on the technique (Rao et al. 2021, Fig 6)

The claim "this experiment has not been run" as written above is **too strong** and must be
narrowed before submission. Rao et al. 2021 (MSA Transformer, ICML), §5 / Fig 6, already
apply the same rhetorical move — take a symmetry that Potts has by construction, apply it,
and test whether the neural model shares it. They shuffle (a) values within each column
(destroys covariation, Potts → random baseline) and (b) column order (preserves covariation,
Potts invariant by construction, ESM-1b → random baseline).

What remains genuinely unoccupied is the **axis**, not the technique:

- within-column shuffle → covariation destroyed
- column-order shuffle → position destroyed
- **alphabet relabeling (this protocol)** → *residue identity* destroyed, covariation and
  position both preserved

Ours is the only one of the three that leaves the input a statistically valid protein family,
which is what makes the "the task did not get harder" argument airtight rather than merely
plausible. It is also the only axis on which the categorical Jacobian — which did not exist
in 2021 — can be used as the readout, giving a like-for-like comparison against Potts
couplings that Rao could not run.

**Required changes to the write-up:** cite Rao Fig 6 explicitly, present the three-axis table,
and claim novelty on the axis only. Zhang & Ovchinnikov (2024) still contains no relabeling
control of any kind (verified against the full text and the 22-page SI), so the specific claim
"this was never run on the categorical Jacobian" stands.

**Adopted from Rao for comparability:** hhfilter to 1024 sequences, restrict to MSAs with
≥1024 sequences, report top-L long-range precision alongside our top-L/2, and include an
explicit random-guess null.

## Amendment 3 — 2026-08-05, primary DV changed to self-consistency

Registered **before** the first ESM-2 run.

The DV above is contact precision against experimental structures. The GREMLIN server is
dead (404) so those MSAs are unrecoverable, and we do not yet have a matched structure set.
More importantly, precision is the *wrong primary DV for an invariance claim*: it invites
the objection that the permuted sequence is simply harder, which is exactly the confound
this experiment was designed to eliminate.

**New primary DV — self-consistency.** Permute the sequence, recompute the Jacobian contact
map, and measure how far the model's *own* prediction moved:

- `overlap` — fraction of the baseline top-L/2 long-range contact set recovered after
  permutation. Potts reference value: **1.000 exactly**, by construction.
- `spearman` — rank correlation of the two score matrices over all eligible long-range pairs.

An alphabet relabeling leaves positions untouched, so both maps live in the same coordinate
space and are directly comparable. No ground truth is required, and the measurement is immune
to the "task got harder" objection because we are not asking whether ESM-2 is *right*, only
whether it is *consistent with itself* under a transformation that provably preserves every
alignment statistic.

**Registered prediction (unchanged in substance):** overlap decays monotonically in m and
falls far below the Potts value of 1.000. Secondary: at matched m, `Cross-class` ≤ `In-class`;
if they are equal, that is the stronger result.

**Falsification:** if ESM-2's overlap stays near 1.0 across all m, H2 is refuted and ESM-2 is
more algorithmic than the lookup account predicts. We report that as a positive result.

`min_sep` is relaxed from 24 to 12 for this pilot because the pilot proteins are short
(L = 56–162) and |i−j| ≥ 24 leaves too few eligible pairs to rank. To be raised back to 24
for the full-length run.

Precision-against-structure is retained as a **secondary** DV, to be run once structures are
in place; it answers a different question (whether the degradation costs real predictive
performance).

## Amendment 4 — 2026-08-05, matched-Hamming control (C6)

Registered after seeing the first pilot point (m=2 → overlap 0.525) but **before** running
this control.

**The objection this closes.** A bijective relabeling with m=2 still changes roughly 10% of
residue positions. A reviewer will immediately say: ESM-2 is simply sensitive to sequence
edits, and you have shown nothing about *identity* specifically. The MRF comparison does not
fully answer this, because the MRF is fitted to an MSA whereas ESM-2 reads a single sequence
— so "coevolution is preserved" is not automatically an argument about ESM-2's input.

**C6 — matched-Hamming random substitution.** For each permutation condition, construct a
control sequence with **exactly the same Hamming distance** from wild-type, but produced by
random substitutions rather than a bijection. Everything about the amount of change is held
constant; only the *structure* of the change differs.

Registered predictions, and what each would mean:

| Observation | Interpretation |
|---|---|
| permutation ≈ random at matched Hamming | ESM-2 is edit-distance sensitive; H2's identity claim is **not supported** and we must say so |
| permutation **better** than random | ESM-2 retains some relabeling-robust structure — a partial-competence result |
| permutation **worse** than random | strongest mirage result: consistent global relabeling is *more* disruptive than the same number of arbitrary edits, which is what a memorised-lookup account predicts and a statistical account does not |

**This control can refute H2**, and that is the point of registering it before running it. The
m=2 result is not reportable without it.

Also note the honest framing of the asymmetry: the relabeling is applied to whatever input
each model takes, but ESM-2's *implicit* reference statistics are its training corpus, which
is not relabeled. That asymmetry is precisely the claim — a model that had internalised the
algorithm rather than the lookup table would not depend on the corpus's alphabet convention
— but it must be stated plainly rather than glossed.

## Amendment 5 — 2026-08-05, the comparison is unfair as stated; C7 fixes it

Registered before running C7. **This is the most serious threat to H2 and it is of our own
making, not a reviewer's invention.**

**The problem.** The Potts model is *refit* on the permuted MSA. ESM-2 gets no relabeled
training data. So "Potts is invariant, ESM-2 collapses" partly restates the fact that one
model derives its knowledge at inference time and the other has it frozen in weights. A
sharp reviewer will say the comparison is rigged, and they would be partly right.

**What survives the objection.** The honest claim is not "ESM-2 is worse than Potts". It is:

> Potts's coevolutionary knowledge is *recomputed from the input* and therefore transfers
> under relabeling; ESM-2's is *amortised into weights keyed to one alphabet convention*
> and therefore does not. Amortisation buys speed and single-sequence inference at the cost
> of distribution-boundedness.

That is a real claim, it is the ICBINB thesis, and it must be stated in exactly those terms
rather than as a naive head-to-head.

**C7 — MSA Transformer, the fair neural comparison.** MSA Transformer takes an *alignment*
at inference, exactly as Potts does. So it removes the asymmetry entirely: both models see
the permuted family, neither is retrained.

- If MSA-T is invariant → the collapse is an artefact of single-sequence amortisation, and
  H2's interpretation must be narrowed to "single-sequence pLMs" specifically.
- If MSA-T **also** collapses → far stronger result. A neural model *holding the alignment
  in its hands* still cannot perform the alphabet-agnostic computation that inverse
  covariance does trivially and exactly. The failure is then about the learned mechanism,
  not about lack of access to the data.

Either outcome is publishable and they say materially different things, which is what makes
this the right experiment to run next. Rao et al. 2021 already characterise MSA-T under
their two shuffles, giving us a reference point for the third axis.

**Priority: run C7 before writing up H2.** The m-sweep result should not be presented
without it.

## Deliverable

`results/h2_precision.csv` with columns:
`pdb_id, model, regime, m, seed, precision_at_L2, n_contacts, msa_depth, length`
plus `analysis.md` interpreting it.

---

## Amendment 6 — C8, the scale sweep (registered 2026-08-05, before running)

**Why this is the next experiment and not a nice-to-have.** Every measurement so far is at
a single model size. The first question any reviewer asks of "the big model fails at
something the 20-line baseline does exactly" is *does it go away at scale?* If we cannot
answer that, the result reads as a quirk of one checkpoint. If we can, it is a statement
about the model class. ICBINB in particular is a venue about things that do not get better
when they should, so the scale axis is not decoration here — it is the genre.

**Design.** Hold everything fixed except parameter count. Same proteins, same permutations,
same seeds, same DV, across the ESM-2 family:

    esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D,
    esm2_t33_650M_UR50D, esm2_t36_3B_UR50D

The 650M numbers are **re-measured inside this sweep** rather than spliced in from the main
run, so every cell in the table is internally comparable. Regimes restricted to `random`
and `In-class` — those two carry the claim (the contrast between them is the finding
that ESM-2 encodes biochemistry but not alphabet-abstraction), and `Cross-class` costs a third
of the budget to sharpen a point that is already made.

**DV.** Unchanged: top-L/2 long-range contact-set overlap against the same model's own
unpermuted map, plus Spearman over all long-range pairs. Self-consistency, so no ground
truth is needed and the comparison across sizes is not confounded by the fact that bigger
models predict contacts better in the first place. Each model is compared **only to
itself**.

**Chance floor is model-independent** — it is `|top-L/2| / n_pairs`, a function of length
alone — so the same horizontal line applies to every size. That is what makes a single plot
legible.

### Registered predictions

1. **Primary.** Overlap under `random` relabeling at m ≥ 10 stays near chance at *every*
   size. Scale does not buy alphabet abstraction.
2. **Secondary.** The In-class-minus-random gap **widens** with scale: larger models
   have learned more biochemistry, so a In-class swap costs them relatively less, while
   an arbitrary one still destroys the lookup.

### Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| Random collapse flat or worsening across sizes | Prediction 1 holds. The claim is about the model class, not a checkpoint. |
| Random overlap rises monotonically and materially with size | **Prediction 1 is refuted.** The honest report is that the failure is a small-model artifact that scale repairs, and the paper's claim must be scoped to ≤ some size — or the paper does not have this result. |
| Random collapse flat but In-class gap does NOT widen | Prediction 2 fails alone. Report it. The primary claim survives; the biochemistry story weakens to "present at all sizes, not increasing." |
| Non-monotonic in size | Report as non-monotonic and do **not** narrate a mechanism. See `claims_forbidden`: we already refused an inverse-scaling claim on the 650M-vs-15B comparison for exactly this reason. |

**Budget guard.** 3B runs last with a reduced batch size, and results are written
incrementally, so preemption or an OOM at 3B still leaves a complete 8M–650M sweep. If 3B
does not finish, the table is reported as 8M–650M and the omission is stated in the caption
rather than quietly leaving a gap.

---

## Amendment 7 — the x-axis is wrong at m=20 (registered 2026-08-05, reporting correction)

Found by auditing `src/permute.py` against the figure, not by looking at results. It changes
no experiment and no prediction; it changes what the horizontal axis is allowed to say.

**The requested `m` is not always the delivered `m`.** Measured directly by counting
non-fixed points of the permutations the runs actually used (seeds 0, 1, 2):

| regime | asked 2 | asked 6 | asked 10 | asked 20 |
|---|---|---|---|---|
| random | 2 | 6 | 10 | **20** |
| In-class | 2 | 6 | 10 | **15** |
| Cross-class | 2 | 6 | 10 | **14–16** |

At m ∈ {2, 6, 10} all three regimes deliver exactly what was asked, so every cell the
headline claims rest on is faithful. Only the m=20 cell is not.

**This is a fact about the amino-acid alphabet, not a bug.** `In-class_GROUPS`
(IVLM, KR, DE, NQ, ST, FYW) covers 15 of 20 residues; A, C, G, H and P have no
biochemically similar partner to swap with. **A fully In-class relabeling of the
complete alphabet does not exist.** The same ceiling binds `Cross-class_perm`, whose pair list
shares residues across pairs so a non-conflicting matching saturates around 15.

**Correction adopted.** Plot every point at its *realized* m, computed deterministically
from the same seed formula the runs used, and label the axis "residue types actually
relabeled". The In-class curve therefore ends at 15 and there is simply no In-class
point at 20 — which is the honest statement, and a more interesting one than a flat segment
from 10 to 20 that was really the builder saturating rather than the model coping.

**What this does NOT license.** It is not a post-hoc reason to drop the m=20 random cell,
which is genuine and is where ESM-2 reaches chance. Random remains the regime that carries
the primary claim precisely because it is the only one that can relabel the whole alphabet.

---

## Amendment 8 — C9, true contact precision (registered 2026-08-05, before running)

**This is not a new DV; it is the ORIGINAL one.** The locked protocol's dependent variable
was contact precision against experimental structures. We substituted self-consistency
because no structures were on disk, and `run_h2_esm2.py` records that the accuracy question
"remains worth running later; it answers a different question." `src/structures.py` now
exists and all four targets pass its sanity bar, so the substitution is no longer necessary
and the original DV is measured.

**Why both DVs are kept rather than one replacing the other.** They answer different
objections and each is weak where the other is strong.
- *Self-consistency* is immune to "the permuted sequence is simply a harder input" — it asks
  only whether the model agrees with itself under a transformation that provably preserves
  every alignment statistic. But it cannot distinguish a model that degrades from one that
  was never right.
- *True precision* shows the degradation costs real predictive performance, and grounds the
  claim in the quantity the field actually reports. But it invites the harder-input
  objection, which self-consistency has already closed.

Reporting both, and saying which answers which, is stronger than either alone.

**Targets.** Ubiquitin (1UBQ_A) and protein G B1 (1PGB_A) only. Both are 100% identical to
their deposited structure. Cytochrome c is excluded on purpose: our sequence is human and
1HRC is horse heart at 88.5% identity, so its contact map is an ortholog's, and lysozyme T4
is a pseudo-wild-type. Two clean targets beat four with caveats when the DV is accuracy.

**Conditions.** regimes {random, In-class} × m ∈ {2, 10, 20} × 2 seeds, plus identity.
Reduced from the full grid because four other jobs are contending for the GPU; the cells
kept are the ones the claim rests on. Cross-class is dropped here — it is a rhetorical
reinforcement, not part of the primary claim.

**Metric.** Long-range (|i−j| ≥ 12) P@L/2 against CB–CB < 8 Å, via
`structures.precision_at_topk`. Chance is the long-range contact density.

### Registered predictions

1. **Identity-condition precision reproduces the already-measured values** — ubiquitin
   ≈ 0.76, protein G ≈ 0.50. This is a pipeline gate: if it does not reproduce, nothing
   else in this run is reported.
2. **Precision falls toward the chance density under `random` relabeling at m = 20.**
3. **In-class retains materially more precision than random at matched realized m.**

### Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| Precision falls to chance under random, In-class retains | Both DVs agree. The degradation is real predictive loss, not just self-inconsistency. |
| Self-consistency collapses but precision holds up | **Important and against us.** The model would be finding the same contacts by a different route, and "collapse" would be the wrong word. Report prominently; the paper's verb changes. |
| Precision falls under BOTH In-class and random equally | Prediction 3 fails. The biochemistry dissociation is specific to self-consistency and must not be claimed for accuracy. |
| Identity precision does not reproduce | Pipeline bug. Nothing reported until fixed. |

---

## Amendment 10 — C10: does the collapse cross the classical baseline? (2026-08-05)

**Registered before running.** C9 established that ESM-2 650M's true long-range contact
precision on 1UBQ falls from 0.763 (14× chance) to 0.053 (**exactly** chance) under an m=20
random relabeling. The obvious reviewer question that C9 cannot answer is: *so what — is
0.053 bad relative to anything a practitioner would actually use?* Potts is exactly invariant
by theorem (`theory.md` Prop. 2), so it has a single precision number that holds under every
relabeling. If that number exceeds ESM-2's permuted number, the claim upgrades from

> "ESM-2 degrades to chance"

to

> "under a transformation that provably preserves every alignment statistic, the neural model
> falls **below the classical model it replaced**, which is unaffected by construction."

That is a crossing, and it is the strongest single sentence the experiment can support.

### Design

- **Target**: ubiquitin / 1UBQ chain A, the C9 headline target. Only one target, because
  only one has both a deep MSA and a 100%-identity PDB chain.
- **MSA**: `RL40A_YEAST_full_11-26-2021_b01.a2m` from ProteinGym (depth 16,228). Its query
  is yeast RL40A, a ubiquitin–ribosomal fusion; the ubiquitin region is match columns 1–67,
  which correspond to **human ubiquitin residues 10–76**, differing at exactly the three
  known positions (P19S, E24D, A28S). Columns 68–89 are the ribosomal fusion and are cut.
- **Coverage restriction, applied identically to both models.** Ubiquitin residues 1–9 (the
  β1 strand) are outside the MSA. They are removed from the candidate set for *both* Potts
  and ESM-2 by setting `mapping[i] = -1`, so the two models are scored on the identical pair
  set with the identical `k = int(76 × 0.5) = 38`. ESM-2's full-length 0.763 is reported
  alongside its restricted number so the cost of the restriction is visible and not hidden
  inside the comparison.
- **Conditions**: Potts identity; Potts under m=20 random (2 seeds, expected bit-identical);
  ESM-2 identity; ESM-2 under m=20 random (2 seeds) — the same permutations, same builder,
  same seeds as C9.

### Registered predictions

1. **Potts under relabeling is bit-identical to Potts at identity** — max|ΔS| < 1e-9,
   enforced as an in-run assert. This is Proposition 2, not a finding.
2. **Potts identity precision is well above chance** on the restricted set. If it is not,
   see the refuting branch.
3. **ESM-2 restricted identity > Potts** — the ordering everyone expects, and the one that
   makes the crossing interesting rather than a story about a weak neural model.
4. **ESM-2 at m=20 random < Potts** — the crossing.

### Declared limitation, stated before the numbers

`src/mrf.py` fits couplings with a plain ridge and **no sequence reweighting** (no Meff
downweighting of redundant homologues), which is standard practice and typically *raises*
Potts accuracy. Our Potts is therefore a floor, not a tuned baseline. This cuts in the
In-class direction for prediction 4 and in the *anti*-In-class direction for
prediction 3, and both readings are reported.

### Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| Potts identity ≫ chance, ESM-2 identity > Potts > ESM-2 permuted | The crossing. Report as the headline framing of C9. |
| Potts identity ≫ chance, but ESM-2 permuted ≥ Potts | **Prediction 4 refuted.** ESM-2 permuted is not below the classical baseline; C9 keeps its "falls to chance" phrasing and no crossing is claimed. |
| Potts identity ≈ chance | The MSA or the untuned estimator is too weak to serve as a baseline. The comparison is declared underpowered and **withdrawn** — we do not go shopping for a different MSA to make Potts look better. |
| Potts identity > ESM-2 identity | Reportable and interesting, but it means ESM-2 was never the better model here, and the "neural model falls below the classical one" framing is dropped as trivially true. |

### Amendment 10a — C10 gets a second target (2026-08-05, registered after C10's ubiquitin result)

Amendment 10 restricted C10 to ubiquitin and gave the reason: *"only one has both a deep MSA
and a 100%-identity PDB chain."* **That reason was wrong, and I had not checked it.** Protein
G B1 is 100% identical to 1PGB chain A (verified in C9's identity gate), and ProteinGym ships
`SPG1_STRSG_full_b0.1.a2m` at depth 3,109. So the sample size of the crossing result is one
purely because I did not look, which is not a defensible reason to leave it at one.

Registering the extension before running it, and stating what I already know: on ubiquitin the
crossing confirmed at ESM-2 2.33× Potts → 0.17× Potts. That knowledge is why this amendment is
not free of hindsight, and a reader should discount it accordingly.

**Design.** Identical to Amendment 10, with three differences forced by the data:

- The alignment query is the **full-length 448-residue** SPG1_STRSG, of which the B1 domain is
  a sub-range. Protein G carries B1/B2/B3 repeats that differ at a handful of positions, so
  the located region will not be 100% identical to our B1 query the way ubiquitin's was
  (95.5%). The measured region identity is **reported**, and if it falls below 0.80 the target
  is dropped rather than patched.
- Potts is fit on the **extracted domain columns only**, not the full 448. Fitting 448×21 =
  9,408 parameters from 3,109 sequences is rank-deficient and would produce a meaningless
  baseline.
- Depth 3,109 against ubiquitin's 16,228, on a 56-residue domain. If Potts lands near chance
  here, **Amendment 10's withdrawal branch applies to this target alone** — the ubiquitin
  crossing is not retracted, and we do not go looking for a third MSA.

**Registered predictions.** Same four as Amendment 10. Prediction 3 (ESM-2 identity > Potts)
is the one at genuine risk: ESM-2's identity precision on protein G is 0.500 at 6.3× chance,
much weaker than ubiquitin's 14.4×, so a competent Potts could plausibly beat it. If it does,
Amendment 10 branch 4 applies **to this target**: reportable, but the crossing framing is not
claimed for protein G.

---

## Amendment 11 — C11: a third architecture (AMPLIFY 120M) (2026-08-05)

**Registered before running.** The generalisation objection is one of the three the review
evidence makes non-optional (findings.md F16): *are pLMs uniquely fragile, or is this
representative of deep sequence models in general?* We currently have ESM-2 (five sizes, C8)
and MSA Transformer (C7). Both are Meta models trained on UniRef with the same 33-token
alphabet-plus-specials vocabulary. A reviewer can reasonably read that as one lineage.

AMPLIFY 120M is a different lineage: SwiGLU feed-forward, RoPE, RMSNorm, a **27-token
vocabulary** with a different residue ordering, and a different training corpus. The
checkpoints are already on disk.

### Design

- **Model**: `AMPLIFY_120M_2024` (the full-corpus checkpoint), loaded by `src/amplify_jac.py`.
  That wrapper subclasses `jacobian.ESM2Jacobian` and overrides **only** `_tokenize` and
  `_logits`, so the Jacobian reduction — mean-centring, symmetrisation, per-block Frobenius,
  APC — is byte-for-byte the code that produced every ESM-2 number. A cross-architecture
  comparison is worth nothing if the readout differs, so this is a design requirement, not an
  implementation convenience.
- **Comparison point**: ESM-2 150M from the C8 scale sweep. 118M vs 148M parameters is the
  closest matched pair available, so architecture and corpus vary while capacity roughly does
  not.
- **Proteins**: the C8 subset — ubiquitin, protein G B1, cytochrome c — for direct
  comparability with the scale sweep.
- **Conditions**: `random` and `In-class`, m ∈ {2, 6, 10, 20}, seeds {0, 1}. Plotted at
  realized m per Amendment 7.
- **Primary DV**: top-L/2 long-range contact-set overlap between the unpermuted and permuted
  predictions, identical to C8.

### Validity gate, and why it is here

Readout 2 taught this the expensive way: an aggregate degradation statistic is meaningless on
a unit that had no signal to begin with. **Before any permuted cell is reported, AMPLIFY's
*unpermuted* long-range P@L/2 against experimental structure is measured on ubiquitin (1UBQ_A)
and protein G B1 (1PGB_A)** — the two 100%-identity targets from C9.

**If AMPLIFY's identity precision is below 3× chance on both targets, the overlap sweep is
declared uninformative and withdrawn.** A model that cannot predict contacts has no contact
prediction to destroy, and reporting its overlap collapse would be exactly the error Amendment
9 exists to prevent. This threshold is fixed now, before the gate is run.

### Registered predictions

1. **AMPLIFY's contact-set overlap collapses under random relabeling**, monotonically in m,
   as ESM-2's does.
2. **In-class degrades substantially less than random at matched realized m** — the
   dissociation replicates in a third architecture.
3. Potts remains the invariant reference by Proposition 2; no new Potts run is needed.

### Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| AMPLIFY collapses like ESM-2, with the In-class/random dissociation | The strong outcome. The violation is a property of the model class, not of one lab's tokenizer or corpus. This is what closes the generalisation objection. |
| AMPLIFY collapses but In-class ≈ random | Prediction 2 fails for this architecture. The collapse generalises; the biochemistry reading is ESM-specific and must be stated as such. |
| **AMPLIFY is invariant or nearly so** | **Prediction 1 refuted, and this is the most interesting possible result.** It would mean alphabet-equivariance is achievable by a pLM and ESM-2's failure is a design or corpus artifact rather than a property of the approach. Report it as the headline and rewrite the paper around it. Do not bury it. |
| AMPLIFY identity precision < 3× chance on both targets | Withdrawn as uninformative. Do not substitute a different checkpoint to get a usable gate. |

---

## Amendment 12 — C12: dose–response between biochemical cost and contact-map damage (2026-08-05)

**Registered before running.** The venue's third scoring criterion asks whether the paper
"goes beyond reporting a bad number to characterize **causes**" (findings F22). Our causal
account says ESM-2's contact map is produced by residue-identity-conditioned pattern matching
rather than column-statistic inference, and the evidence for the *biochemistry* half of that
is currently **three ordinal levels**: `In-class` < `random` < `Cross-class`. Three points is
a contrast, not a mechanism.

If the account is right, damage should be a *continuous function of how far the relabeling
moves each residue in biochemical space*. That is a dose–response prediction, and it is
falsifiable in a way the three-level contrast is not.

### Design

- **Held constant:** m = 20 (every residue relabeled), a single model, a single protein, the
  same top-L/2 long-range DV. Every permutation is a full derangement of all 20 residues, so
  **every condition changes the same number of positions**. This is C6's matched-Hamming logic
  extended from two points to a continuum: the intervention size does not vary, only its
  biochemical character does.
- **Independent variable:** sequence-weighted BLOSUM62 cost of the permutation,
  `cost(π) = mean_i −BLOSUM62[s_i, π(s_i)]` over the residues actually present in the
  sequence. Sign convention is H4's (`severity` = negated BLOSUM62, higher = more disruptive)
  and is asserted in-run against `sev(K→R) < sev(K→W)`.
- **Sampling:** 60 permutations per protein. Sample random derangements, compute cost, and
  bin-stratify so the cost range is covered rather than clustered at the random-derangement
  mode. The `In-class` and `Cross-class` builders are included as labelled anchors.
- **Model / proteins:** ESM-2 650M (the headline model), ubiquitin and protein G B1.

### Primary DV and registered prediction

Spearman ρ between permutation cost and top-L/2 self-overlap, across permutations, per protein.

**Prediction: ρ ≤ −0.6 on both proteins** — damage rises monotonically with biochemical cost at
constant intervention size.

### Outcome interpretation — written before seeing numbers

| Observation | Reading |
|---|---|
| ρ ≤ −0.6 on both | Dose–response confirmed. The three-level contrast becomes a curve, and "keyed on biochemical identity" is a measured functional relationship rather than an ordinal claim. This is what criterion 3 asks for. |
| ρ ≤ −0.6 on one, weaker on the other | Report both. C9 already established the In-class/random dissociation is **protein-dependent**, so a split is consistent with what we know and must not be averaged away. |
| **\|ρ\| < 0.3 on both** | **Prediction REFUTED, and this matters.** It would mean the In-class-vs-random gap is not a biochemical-similarity effect but something coarser (e.g. driven by a few high-frequency residues, or by whether specific tokens move at all). The causal account in `paper/OUTLINE.md` §4 would be wrong as stated and must be rewritten, not softened. Report the null. |
| ρ positive | Something is broken. Check the BLOSUM sign assert before believing it. |

### Registered confound check

Cost is sequence-weighted, so a permutation could score low simply by leaving *rare* residues
badly mismatched while sparing common ones. Unweighted cost (uniform over the 20 letters) is
recorded alongside, and both correlations are reported. If they disagree materially, the
weighted one is primary — damage should track the residues the model actually sees — and the
disagreement is reported rather than resolved silently.

### Amendment 12a — two design corrections, found in a dry run before any model was loaded

Registered before C12 produced a single number. Both were caught by checking the sampler's
output rather than by reading the code.

**(a) The anchors must be excluded from the correlation.** Amendment 12 said the
`In-class` and `Cross-class` builders would be included as labelled anchors. Their **realized
m is 15 and 16, not 20** — `In-class_GROUPS` covers only 15 of 20 residues (A, C, G, H, P
have no partner) and `Cross-class_PAIRS` covers 16. Putting them into a Spearman alongside 60 full
derangements would let intervention *size* vary with cost, which is precisely the confound the
constant-m design exists to remove. **ρ is computed over the m=20 derangements only**; the
anchors are recorded and plotted as labelled reference points and excluded from every
correlation. The CSV keeps `m_realized` per row so this is checkable.

**(b) The low-cost tail needs searching for, not sampling.** Random derangements of all 20
residues span weighted cost ≈ 0.09 to 2.29 on ubiquitin, while the In-class anchor sits at
−2.33 — entirely outside the sampled range. A dose–response fitted only over 0.09–2.29 would
leave the interesting end of the axis untested, and the only low-cost points available would be
the ones with the wrong m.

So C12 **searches** for biochemically In-class *full* derangements by simulated annealing
on permutation cost, subject to the no-fixed-point constraint, and pools them with the random
sample before bin-stratifying. This is a strictly better test than Amendment 12 described: it
compares In-class against Cross-class relabelings **at exactly matched m = 20**, decoupling
"biochemically gentle" from "fewer residues moved" in a way even C6 did not. C6 matched Hamming
distance in the sequence; this matches the number of relabeled residue *types*.

Predictions and the refuting branch from Amendment 12 are unchanged.

---

## Amendment 13 — C13: does a *pseudolikelihood* Potts model stay invariant? (2026-08-05)

**Registered before running. CPU-only by design** — the GPU is saturated and this needs none.

`theory.md` flags this gap itself, in the section arguing that Proposition 2 carries the paper:

> **Proposition 1 is an optimality statement.** It says the relabelled parameters *are a
> maximiser* of the relabelled objective. That leaves a live objection: nobody solves the Potts
> maximum-likelihood problem exactly. Real pipelines run pseudolikelihood with early stopping,
> or a particular optimiser, or a particular initialisation — and an argument about maximisers
> says nothing about which maximiser a given solver returns.

Every Potts number in this project comes from the mean-field / inverse-covariance estimator,
which Proposition 2 covers **exactly** and closed-form. But the classical models a reviewer has
in mind — plmDCA, GREMLIN, CCMpred — are *pseudolikelihood* fits with a real optimiser. We have
never tested one. If our claim is "the classical comparator is invariant", that claim currently
holds for an estimator nobody in the field actually runs.

### Design

Implement asymmetric pseudolikelihood Potts (`src/plm.py`): per-column multinomial logistic
regression of σ_i on all other columns, isotropic L2 penalty, L-BFGS; symmetrise
`J_ij ← (J_ij + J_jiᵀ)/2`; zero-sum gauge; Frobenius norm per block; APC. Same MSAs, same
targets and the same DV as C10 (ubiquitin / RL40A_YEAST, protein G B1 / SPG1_STRSG).

Fit twice per target — on the alignment, and on the globally relabelled alignment — and compare.

### The initialisation contrast, which is the actual point

Run **two** initialisation regimes, because they test different things:

- **(a) zero init.** The optimiser starts at a relabeling-equivariant point, so the entire
  optimisation *trajectory* of the relabelled problem is the exact image of the original
  trajectory under conjugation by `Q = I_L ⊗ P_π`. Invariance should hold to floating-point
  noise regardless of whether the solver converged. **Prediction: max|ΔS| < 1e-8 relative to
  score scale, top-L/2 set identical.**
- **(b) random init**, drawn independently for the two fits. The trajectories are no longer
  images of one another, so invariance can only hold up to convergence. This is the honest
  worst case for the objection above. **Prediction: top-L/2 overlap ≥ 0.95**, with ΔS orders
  of magnitude below the ESM-2 effect.

### Registered predictions and refuting branches

1. Both regimes are invariant to within the stated tolerances → the objection "your baseline is
   invariant only because it is closed-form" is answered empirically, on the estimator class
   the field actually uses. Report (a) and (b) separately; do not average them.
2. **(a) passes but (b) fails materially (overlap < 0.95)** → invariance is a property of the
   *equivariant trajectory*, not of the fitted model. That is a genuinely interesting and
   publishable qualification, and it must be stated as one rather than buried; the paper would
   then say classical DCA is invariant *given a symmetric initialisation*, which is what every
   standard implementation does anyway.
3. **(a) itself fails** → either the implementation is wrong or a hidden non-isotropy exists
   (theory.md's load-bearing condition). Debug before reporting; a failure here is an
   implementation bug until proven otherwise, and `test_theory_conditions.py` already shows an
   anisotropic ridge produces exactly this signature.

### Correctness gate, fixed before the run

The plm implementation must reproduce known behaviour before its invariance means anything: its
unpermuted top-L/2 long-range precision against the deposited structure must be **at least as
good as the mean-field estimator's** on the same alignment (C10 measured 0.3158 on ubiquitin,
0.1786 on protein G). A pseudolikelihood fit that is *worse* than mean-field is a broken fit,
and its invariance would then be the invariance of noise. If the gate fails, C13 is withdrawn
and reported as withdrawn.

### Amendment 13a — the correctness gate did its job: a regularisation bug (2026-08-05)

**Recorded before any corrected number exists.** Amendment 13's gate failed on its first cell,
and Amendment 13 registered in advance what to do about that: *"a failure here is an
implementation bug until proven otherwise. Debug before reporting."* This is that debugging,
written down because the implementation was changed **after** seeing a result and that must be
visible rather than quietly folded in.

**First-run result (ubiquitin, zero init, λ nominal 0.01):**

| quantity | value | expected |
|---|---|---|
| mean-field P@L/2 | 0.3333 | — |
| plm P@L/2 | **0.2727** | ≥ 0.3133 → **GATE FAIL** |
| max\|ΔS\| relative | **1.29e-02** | < 1e-8 |
| top-L/2 overlap | 0.970 | ≈ 1.000 |

**The bug.** In `_neg_pll_and_grad` the L2 penalty was added to the *summed* negative
log-likelihood and the total then divided by `n`:

```python
ll += lam * (j * j).sum()          # added to the SUM
return ll / n, ...                 # then everything divided by n
```

so the penalty's effective coefficient against the mean log-likelihood was `lam / n` =
0.01 / 3000 ≈ **3.3e-6**. The fit was, to three significant figures, unregularised — with
29,568 free parameters per column and 3000 sequences.

**One cause, both symptoms.** An unregularised fit at that parameter-to-data ratio overfits, so
precision falls below mean-field (gate fail); and the objective is ill-conditioned, so L-BFGS
converges slowly and the two fits stop at meaningfully different points, so invariance comes
out at 1e-2 relative instead of near machine precision. Neither symptom implicates
Proposition 1.

**Why the synthetic test missed it.** `plm_test` used L=12 and a strong planted signal — 5,292
parameters per column against 800 well-determined sequences. That problem is essentially
determined with or without a penalty, so it recovered 4/4 couplings and showed 5.4e-06
invariance regardless. **A correctness test on an easy instance cannot detect a
regularisation bug.** This is the same lesson as the earlier real-Pfam control that passed
because our own gap filter had removed the characters it existed to exercise: construct the
awkward case, or the control certifies nothing.

**The fix.** `lam` now multiplies the penalty against the *mean* log-likelihood, the usual
plmDCA convention, making it n-independent and directly interpretable. λ is then chosen on the
gate — a small sweep over {0.001, 0.01, 0.1} against mean-field precision on ubiquitin.

**Registered constraint on that choice, so it cannot become gate-fitting.** λ is selected on
the **unpermuted** fit's precision only. The permuted fit is not run during selection and
invariance is not consulted. Selecting a hyperparameter on the DV would make C13 circular; the
gate is a *fitness-for-purpose* check on the estimator and nothing more. Whichever λ is chosen
is then fixed for both proteins and both initialisation regimes, and reported.

Predictions and refuting branches from Amendment 13 are otherwise unchanged. If the gate still
fails at every λ, C13 is withdrawn as registered.

## Amendment 14 — C14: is the collapse just "the input is out-of-distribution"? (2026-08-05)

**Registered before writing the measurement code. No C14 number exists yet.**

**The objection this exists to answer, in its strongest form.** Everything so far shows the
contact map collapses under relabeling and that a *In-class* relabeling costs far less
than a *random* one at matched edit distance (C6). A reviewer can accept all of that and still
say: a randomly relabeled sequence is simply **not a protein** as far as the model is
concerned — its amino-acid composition is wrong, `A→W` everywhere makes tryptophan the most
common residue — so the model is being fed gibberish and any degradation is unremarkable.
Under that reading, the In-class-vs-random dissociation reduces to *In-class
relabelings stay closer to the training distribution*, and the alphabet-abstraction claim adds
nothing.

C6 answers the **edit-distance** version of this confound. It does not answer the
**model-perceived** version, because Hamming distance in sequence space is not the model's own
notion of distance. That is the gap this closes.

**The measurement, and why it is nearly free.** The categorical Jacobian's first step is
already one forward pass on the unmutated sequence, returning logits over the 20 standard
residues at every position. The model's own average log-likelihood of a sequence is a
log-softmax and a gather away:

  LL(s) = (1/L) Σ_i log softmax(logits(s)[i])[s_i]

restricted to the 20-residue support the Jacobian itself uses, so the two quantities are
computed under the same convention. Define the **model-perceived shift**

  Δ_LL(π) = LL(x) − LL(π(x))   (nats/residue; positive = the relabeled sequence is less
                                protein-like *to this model*)

This is label-free, structure-free and alignment-free, exactly like the audit itself.

**Design.** Cells are not chosen fresh; they are **read from the committed main-sweep CSV**
(`h2_esm2_invariance.csv`) and the permutations regenerated with the same seeding convention
(`default_rng(1000*seed + m)`, then `PERM_BUILDERS[regime](m, rng)`), so every Δ_LL joins to an
`overlap` that was measured before Δ_LL was conceived of. ESM-2 650M, all main-sweep proteins,
regimes random/In-class/Cross-class, m ∈ {2,6,10,20}, seeds as recorded. One forward pass per
cell plus one per protein.

**The dependent variable is the mediation question**, not the correlation. Marginally, Δ_LL
and overlap will obviously be related — the arms differ on both. The question is whether
regime still matters **at matched Δ_LL**:

1. Fit overlap on Δ_LL alone (rank-based, pooled across regimes) → R²_base.
2. Add a In-class-vs-random indicator → R²_full.
3. Report ΔR² = R²_full − R²_base, and the partial Spearman of overlap with regime holding
   Δ_LL fixed.

**Registered prediction: NOT fully mediated — two curves, not one.** ΔR² ≥ 0.15 and partial
|ρ| ≥ 0.3. Reasoning: In-class relabeling preserves biochemical *class*, so the local
hydrophobicity pattern that drives contact prediction survives even where the sequence is
equally surprising to the model.

**Both branches are committed in advance, and both are publishable. This is registered
explicitly because the outcome I did *not* predict is the one that flatters the paper's
framing, and that asymmetry is exactly when a result needs to be locked first.**

- **Branch A — MEDIATED** (ΔR² < 0.05 and partial |ρ| < 0.2). My prediction is wrong and the
  single curve is the finding: contact-map competence is a function of the model's own
  perceived distribution shift, and biochemistry matters only insofar as it keeps the input
  near the training distribution. This is Zhao et al.'s thesis instantiated in biology with a
  Δ that is **measured rather than constructed** — the thing findings F14 recorded we could not
  deliver. Outline §4 gets reorganised around it and it is stated as the mechanism.
- **Branch B — NOT MEDIATED** (thresholds above). Perceived-OOD-ness does not explain the
  dissociation; the failure is more specific than generic distribution shift, and §4's current
  three-line causal account stands with Δ_LL added as a fourth line that rules out the
  cheapest alternative explanation.
- **Branch C — ambiguous** (between the two). Report the scatter and the numbers, claim
  neither, and say so in one sentence.

**A negative-control cell that must hold in every branch.** Potts on the same relabeled
alignment is exactly invariant (max|ΔS| ≈ 1e-14, Proposition 2). So a large Δ_LL does **not**
imply the coevolutionary computation is impossible on that input — a model that does it
correctly is unaffected. Whatever branch fires, the shift is a fact about the model's
familiarity with the input, never about the information content of the input. Any wording that
lets Δ_LL read as "the task got harder" is wrong and must not survive review of the draft.

**What C14 cannot license.** LL is computed on a single sequence with BOS/EOS masked and
restricted to 20-residue support; it is not a calibrated likelihood and no absolute value of it
is interpretable. Only *differences* between a sequence and its relabeling, under one model, are
used. Do not compare Δ_LL across model sizes or across architectures — the supports and
temperatures differ.

### Amendment 13b — zero-init met half its registered prediction; the debug branch (2026-08-05)

**Registered before the diagnostic is run.** Amendment 13(a) predicted **two** things for the
zero-initialised fit: `max|ΔS| < 1e-8` relative to score scale, **and** an identical top-L/2
set. The first corrected cell (ubiquitin, λ=0.001, maxiter=150) delivers exactly one of them:

| quantity | registered prediction | measured | verdict |
|---|---|---|---|
| gate: plm P@L/2 vs mean-field | ≥ 0.3133 | **0.5152** vs 0.3333 | PASS |
| top-L/2 set identical | overlap 1.000 | **1.000** | MET |
| permuted precision | — | 0.5152, identical | — |
| relative max\|ΔS\| | **< 1e-8** | **1.88e-03** | **MISSED, by 5 orders** |

Amendment 13 branch 3 applies — *"a failure here is an implementation bug until proven
otherwise"* — so this is not reported as a result until it is diagnosed. We are **not**
relaxing the threshold to fit the number.

**The competing explanations.** (i) A real implementation bug or a hidden non-isotropy, which
is what branch 3 presumes. (ii) Chaotic amplification of floating-point rounding: in exact
arithmetic the zero-init trajectory of the relabelled problem *is* the conjugated image of the
original, but the permuted design matrix has its columns in a different order, so every GEMM
accumulates in a different order and the two gradients differ at ~1e-16 relative from the first
iteration. L-BFGS is not contractive, and 150 iterations on an ill-conditioned objective can
amplify that. Note λ=0.001 was selected precisely because it is the **least** regularised and
therefore worst-conditioned candidate, so this explanation predicts we are seeing close to the
worst case we could have chosen.

**The diagnostic, and what each outcome means.** Re-fit ubiquitin, zero init, λ=0.001, at
maxiter ∈ {2, 10, 40} and compare against the maxiter=150 cell already measured.

- **Divergence grows monotonically with maxiter** (e.g. ~1e-12 at 2 iterations rising to 1e-3
  at 150) → explanation (ii). Rounding is being amplified by the optimiser, the theory is
  untouched, and the honest report is that *the prediction is exactly invariant while the raw
  score matrix is not*. That distinction is then stated in the paper, not hidden.
- **Divergence is already ~1e-3 at 2 iterations** → explanation (i). Something is structurally
  asymmetric between the two fits and it is a bug. Debug further; do not report C13.
- **Divergence does not depend on maxiter in any clean way** → inconclusive; C13 reports only
  the top-L/2 result and explicitly declines to claim score-level invariance for the
  pseudolikelihood estimator.

**Committed in advance: what C13 may and may not claim if (ii) holds.** It may claim that a
pseudolikelihood Potts fit driven by a real optimiser produces an *identical contact set and
identical precision* under relabeling. It may **not** claim score-level invariance to machine
precision for that estimator — that remains a property of the closed-form estimator
(Proposition 2, ≤ 2.3e-14) alone. The paper must state both, because the gap between them is
the honest content of C13: the theory is about the computation, and a numerical solver only
inherits it up to its own conditioning.

## Amendment 15 — C15: does the Δ_LL saturation replicate, and at every scale? (2026-08-05)

**Registered before the measurement code is written. No C15 number exists.**

**Why this experiment exists.** C14's registered mediation test came out ambiguous and the
paper claims neither branch. But the most informative thing in C14 was not the registered
statistic — it was a pattern spotted while reading the output table: in the random arm the
model-perceived shift **saturates** by m=6 (Δ_LL 0.187 → 0.288 → 0.275 → 0.278) while contact
overlap keeps collapsing over the same range (0.711 → 0.342 → 0.079 → 0.010). That is currently
**exploratory** — one model, one protein set, noticed after the fact. An exploratory observation
carrying that much argumentative weight is the weakest link in §4, so it gets tested properly
instead of being written up with a hedge.

**Out-of-sample by construction.** Cells are read from the committed **C8 scale** CSV
(`h2_c8_scale.csv`: 4 models × 4 proteins × 2 regimes × 4 m × 2 seeds = 256 cells), not from
the main sweep C14 used. C8's proteins are `PF00018|BOI2_YEAST, cytochrome_c, protein_G_B1,
ubiquitin`; C14's were six mostly-Pfam sequences, and **only PF00018 is shared**. So even the
650M row is a near-independent replication, and the 8M/35M/150M rows ask something C14 could
not: is the saturation a property of one model or of the model class?

Δ_LL is computed exactly as in C14 (same `mean_loglik`, same 20-residue support, same BOS/EOS
convention), under *each* model in turn. Permutations are regenerated from the recorded seeds
with C8's convention `default_rng(1000*seed + m)`, so each Δ_LL joins an overlap measured long
before Δ_LL existed.

**Registered predictions, random arm, evaluated separately at each of the four scales:**

- **P1 — saturation:** `mean Δ_LL(m=20) − mean Δ_LL(m=6) ≤ 0.05` nats/residue.
- **P2 — continued collapse:** `mean overlap(m=6) − mean overlap(m=20) ≥ 0.10`.

Both must hold at a given scale for that scale to count as replicating. The claim is the
*conjunction*: perceived unfamiliarity stops increasing while the prediction keeps degrading.

**Refuting branch, committed now.** If P1 fails at three or more of the four scales — i.e. Δ_LL
keeps climbing materially alongside the overlap collapse — then C14's saturation was specific to
the 650M model and/or that protein set. In that case the exploratory paragraph is **removed
from §4.4**, not softened, and C14 reverts to reporting only its ambiguous registered result.

**Reported but not registered** (descriptive, and labelled as such): the In-class arm's
Δ_LL trajectory. Note its realized m saturates at 15/20 by construction, so a plateau there is
partly a property of the permutation builder and is **not** evidence for anything. This is
exactly the trap §3.1 already flags for the In-class overlap curve, and it is why the
registered test is confined to the random arm.

**What C15 cannot license.** Δ_LL is not comparable *across* models — different supports,
different temperatures, different training corpora. Every comparison here is within one model:
that model's Δ_LL at m=20 against its own Δ_LL at m=6. No cross-model Δ_LL magnitude is
interpreted, and no ranking of architectures by Δ_LL is permitted.

---

## Amendment 16 — C16: does ESM-2's *own* contact predictor show the same anchoring? (2026-08-05)

**Locked before the script was written and before any number was produced.**

**The hole this closes.** Every contact result in this project so far uses the *categorical
Jacobian* (Zhang & Ovchinnikov): `J[i,a,j,b] = logits(mutate i→a)[j,b] − logits(wt)[j,b]`,
mean-centred, symmetrised, Frobenius per 20×20 block, APC. That is **our** instrument. A
reviewer is entitled to ask whether alphabet anchoring is a property of ESM-2 or a property of
the readout we built on top of it. ESM-2 ships a *different* contact predictor — the supervised
attention-map logistic regression of Rao et al. (2020), exposed as `predict_contacts()` and
distributed as the `-contact-regression.pt` sibling checkpoints (present on disk for all five
ESM-2 sizes and for MSA Transformer). It is a different function of the same network: attention
maps rather than logit differences under single-site mutation, and supervised rather than
unsupervised. If the anchoring is real it must appear in both.

It is also ~1500× cheaper: one forward pass per sequence instead of L×20.

**Design.** Exactly the main sweep's cells, so the two instruments are compared on matched
inputs: proteins from the committed main-sweep protein set, regimes {random, In-class,
Cross-class} × m ∈ {2, 6, 10, 20} × seeds {0, 1, 2}, plus identity. ESM-2 650M is the registered
model; other sizes are descriptive. Permutations are rebuilt with the sweep's seeding
convention, `np.random.default_rng(1000 * seed + m)`, so the permutation is the one that
produced the Jacobian number in the same cell.

**DV.** Identical to the main sweep and therefore directly comparable: top-L/2 long-range
(|i−j| ≥ 24) contact-set self-consistency overlap between the identity and permuted conditions,
computed by the same `top_contacts` / `set_overlap` code. `predict_contacts()` already applies
symmetrisation and APC internally; no extra post-processing is applied, and none is permitted
after seeing results.

**Registered predictions (ESM-2 650M, mean over proteins and seeds):**

- **P1 — collapse.** `random, m=20` overlap ≤ 0.15.
- **P2 — biochemical grading.** `In-class, m=20` overlap − `random, m=20` overlap ≥ 0.20.
- **P3 — instrument agreement.** Spearman ρ between native-head overlap and categorical-Jacobian
  overlap, over all matched (protein, regime, m, seed) cells, ≥ +0.6.

**Pre-committed interpretation.**

| outcome | what the paper does |
|---|---|
| P1, P2, P3 all hold | State that the anchoring is a property of the model, not of our readout, and that it reproduces in ESM-2's own published contact predictor. The categorical Jacobian remains the primary instrument because it is unsupervised; the native head becomes the robustness check. |
| **P1 fails** (native head robust where the Jacobian collapses) | **The anchoring claim is instrument-specific.** Every contact claim in the paper is narrowed to "the categorical-Jacobian readout", in the abstract and in §3, not in a footnote. This is a refutation of the general claim and is reported as one. |
| P1 holds, **P2 fails** | Both readouts collapse, but the biochemical grading is Jacobian-specific. §4.1's regime story is narrowed to the Jacobian; the collapse claim survives unchanged. |
| **P3 < 0.3** | The two instruments disagree cell by cell. Report both, side by side, and claim nothing about their agreement. Do not average them. |
| 0.3 ≤ P3 < 0.6 | Report the correlation with its value; describe the agreement as partial. Do not call it a replication. |

**Registered in advance: the objection we expect, and our answer.** The contact head is
*supervised* — trained on solved structures written in the standard alphabet. A reviewer may
say it therefore has no reason to be alphabet-equivariant and the test is unfair. Our answer,
fixed here so it cannot be invented after the fact: the necessary condition in §2 applies to
any function *claimed to predict contacts from sequence*, and the supervised head makes the
stronger claim, since it is the shipped, published predictor that downstream users call. A
predictor that fails the condition is not thereby proven to have learned nothing — it is proven
not to be computing a function of alignment column statistics, which is what the coevolutionary
story asserts. We will **not** argue that the head "ought" to be invariant by construction.

**What C16 cannot license.** It cannot compare the two instruments' *absolute* overlap levels
as a measure of quality — different predictors, different top-L/2 sets, and the head is
supervised while the Jacobian is not. Only the within-instrument pattern across cells, and the
rank correlation between instruments, are interpreted.

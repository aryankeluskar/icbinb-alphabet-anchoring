# Findings

**Project:** Is protein-language-model competence a distribution-bounded mirage?
**Venue:** ICBINB-BIO @ NeurIPS 2026 — deadline 2026-08-29, 8 pages, non-archival.

---

## Current Understanding

The framing question is *not* "does the CoT-mirage result apply to ESM." ESM emits no
reasoning trace, so that transfer is a category error. What ports from Zhao et al. is
narrower and more useful:

1. **A dialable Δ knob.** Their contribution over prior "CoT is brittle" papers (GSM-Symbolic,
   Illusion of Thinking) is not the conclusion but the *controlled environment* — you can
   turn distribution discrepancy up and watch performance track it with everything else
   pinned. That is what we are importing.
2. **The trace/answer decomposition.** Their strongest evidence is dissociation
   (100% correct trace / 0.01% correct answer, and the reverse via commutativity), because
   failure alone is consistent with "the task is hard" while dissociation is not.
3. **Train-from-scratch leakage control.**

And one asymmetry works *in biology's favour*: Zhao et al.'s stated Limitation (ii) is that
they cannot measure Δ for real LLMs because training corpora are opaque. Protein corpora
are public and versioned. We can measure Δ. That is the paper's novelty hook.

---

## Confirmed Results

### F1 — The MRF invariance premise underpinning H2 is verified (2026-08-05)

**Status:** CONFIRMATORY (matched the pre-registered prediction in
`experiments/H2-alphabet-permutation/protocol.md`, committed before the run).

A Potts/MRF coevolution model is **exactly invariant** under a consistent bijective
relabeling of the amino-acid alphabet applied across a whole MSA.

Measured on synthetic alignments with 15 planted long-range couplings (N=2000, L=60,
min_sep=24), across three permutation regimes (random / In-class / Cross-class) and
m ∈ {2, 6, 10, 20} residue types permuted:

| Control | Result |
|---|---|
| C0 recovery | recall@L/2 = **1.000**, precision at its 0.500 ceiling (chance 0.0085) |
| C2 invariance | `max|ΔS|` ≤ **5.3e-15** in all 12 conditions — exact to float64 |
| C3 true null (per-column row shuffle) | recall **0.000** — dependency genuinely destroyed |
| C5 composition | multiset of residue counts preserved in all conditions |

This is the correctness gate for H2. It establishes that we have an **invariant reference**
against which any ESM-2 degradation can be measured. Independently corroborated from the
PNAS text: Zhang & Ovchinnikov state that applying the categorical Jacobian to an MRF or
multivariate-Gaussian model *exactly returns* its coupling tensor W, so this invariance is a
theorem that follows from their own equations rather than an empirical accident.

**Extended to real and adversarial data (same day).** The synthetic result above is not on
its own persuasive, since real alignments are messier in ways that route through the
one-hot encoder's catch-all branch. Two follow-ups:

- **Real Pfam seed alignments** (PF00072, PF00076, PF00018, PF00027; N = 52–199,
  L = 41–97): `max|ΔS|` ≤ 2.5e-16 and — the stronger check — the predicted **top-L/2
  long-range contact set is literally identical**, not merely numerically close. That set
  is the quantity the paper reports, so it is the one that has to be invariant.
  (PF00013 dropped: only 12 columns survived gap filtering.)
- **Adversarial injection**: 15.1% gaps plus ~0.7% each of X, B, Z, U, O and `*`.
  Invariance holds at `max|ΔS|` ≤ 1.8e-15 with the top-L/2 set identical, and the planted
  signal survives the corruption (recall 1.000). A positive control confirms this has
  power: relabeling those same non-standard characters *while bypassing the guard* moves
  the map by **0.157** and changes the contact set.

### F2 — A per-sequence permutation is NOT a coevolution null (2026-08-05)

**Status:** EXPLORATORY — discovered while debugging C3, not pre-registered.

Applying a *different* bijection to each sequence relabels both columns of a row
identically. Mutual information between two columns is invariant under a bijective
relabeling, so this operation does **not** destroy coevolution — it only blurs it by mixing
conjugated couplings across rows. Measured: recall 1.000 → **0.600**, versus 0.000 for the
true null.

Consequence for the design: the correct null is a **per-column row shuffle** (preserves
marginals, destroys dependency). Had we kept the original control, we would have been
calling a partially-informative condition "destroyed" and the pLM comparison would have
been meaningless. Retained as a third experimental condition rather than deleted, since
"MRF partially survives; does ESM-2?" is now a question worth asking.

### F3 — The technique is not novel; the AXIS is. (2026-08-05)

**Status:** LITERATURE — a near-miss that would have been fatal in review.

Rao et al. 2021 (MSA Transformer, ICML), Section 5 / Fig 6, **already ran input-shuffling
controls with exactly our rhetorical move**: find a symmetry a Potts model has by
construction, apply it, and check whether the neural model has it too. They do two:

| Shuffle | Covariation | Position | Identity | Potts | ESM-1b / MSA-T |
|---|---|---|---|---|---|
| within-column (Rao) | **destroyed** | preserved | preserved | → random baseline | — |
| column-order (Rao) | preserved | **destroyed** | preserved | invariant by construction | → random baseline |
| **alphabet relabeling (ours)** | preserved | preserved | **destroyed** | **exactly invariant** | ? |

So a blanket "nobody has scrambled pLM inputs" claim is **false**, and a reviewer who knows
this paper would kill the submission over it. We must cite Rao Fig 6 explicitly and position
against it.

The position that survives: ours is the **only remaining axis** of the three, and the only
one that leaves the input a *statistically valid protein family* — the within-column shuffle
destroys the family's covariation, and the column-order shuffle destroys its sequence
semantics, whereas an alphabet relabeling yields an alignment that is internally consistent
in every respect except the arbitrary naming of the letters. That is what makes "the task did
not get harder" airtight rather than merely plausible.

Bonus: Rao's protocol is a ready-made template we should adopt for comparability —
hhfilter to 1024 sequences, restrict to MSAs with ≥1024 sequences, top-L long-range
precision, and an explicit random-guess null.

**VERIFIED against the primary sources (2026-08-05).** PMLR main PDF + ICML supplementary
+ all three bioRxiv versions, downloaded to `literature/`. Three things came back:

1. **The numbers are real and now quotable.** Verbatim from the Fig 6 caption: *"Average
   Top-L long-range precision drops from 52.9 (no ablation) to 15.9 (shuffled covariance)
   and 27.9 (shuffled positions) respectively."* All three are **MSA Transformer long-range
   P@L**, long-range meaning s ≥ 24. Report them with their population: averages over only
   the MSAs with ≥1024 sequences, subsampled to 1024 by hhfilter — which is why 52.9 sits
   below the paper's headline 57.4 in Table 1. Quoting 52.9 against 57.4 as if they were the
   same measurement would be an error.
2. **No alphabet relabeling anywhere.** Exhaustive search of *alphabet / relabel / bijection
   / remap / swap / vocabulary permutation* across main text, supplementary and bioRxiv
   v1–v3 returns zero. Every occurrence of *permut\*/shuffl\*/scrambl\** is one of the two
   conditions above, random MSA subsampling, or BERT masking. **Our axis is genuinely open.**
3. **One of my own descriptions was wrong.** I had the position shuffle as "shuffling columns
   within each sequence." It is a **single global column reordering applied identically to
   every row**. The distinction is load-bearing: a per-sequence independent shuffle would
   destroy covariance too, collapsing Rao's two conditions into one. Their claim that it
   "preserves all covariance information between pairs of columns" and that Potts is
   unaffected only holds for the global version. The table above is correct as written; the
   prose gloss I was carrying was not.

**The near-miss we should actually cite — and it cuts against us.** Appendix A.4 (A.12 in
bioRxiv v1), *"Attention to Amino Acids"*, measures KL divergence between column-attention
over amino acids and the background, and concludes the model *"stops focusing on the amino
acid identities in favor of focusing on other properties"* in later layers. That is an
**observational** claim that predicts the opposite of what we measure: if identity stops
mattering, relabeling should be cheap. Our intervention says it is not. Citing this as a
tension we resolve is far stronger than ignoring it and hoping no reviewer recalls it —
it converts a hostile prior into the motivation for doing the intervention at all.

### F4 — Novelty ledger: which angles survive (2026-08-05)

**Status:** LITERATURE, from a 123-paper survey in `literature/`.

| Angle | Verdict | Consequence |
|---|---|---|
| **A** measured-Δ scaling (H1) | **largely taken** — Spinner ran the AMPLIFY-yearly × ProteinGym setup; Hou computed our exact covariate; Gordon established causality via influence functions | demote from headline; keep only the narrow residue (regress *performance* not likelihood, per-protein date-of-entry, screening-off test). **Drop any "first to link corpus statistics to pLM behaviour" claim.** |
| **B** alphabet permutation (H2) | **OPEN** | promote to headline |
| **C** epistatic order (H3) | partially taken empirically; the *framing* is untouched | reframe, see below |
| **D** edit distance vs biochemical severity (H4) | **OPEN — least contested**; all ingredients exist, nobody runs the head-to-head regression | promote to second experiment |
| **E** ESM-3 order dependence (H5) | partially taken; crowded 2026 text-diffusion subfield | keep as an extension only if time allows |

**Reprioritisation:** the paper's spine becomes **B + D**, not A. This is a real change of
plan driven by evidence, and it is a good trade — B and D are both fully unblocked, need no
historical corpus download, and B is already producing results.

### F5 — ProteinGym's scoring protocol cannot express epistasis at all (2026-08-05)

**Status:** LITERATURE, verified in ProteinGym's `compute_fitness.py` source.

The standard protocol masks **one position at a time** and **sums** per-site log-ratios. So
for a k-mutant, the reported score is additive by construction. The `Depth_k` columns that
everyone cites as evidence about higher-order epistasis are therefore **comparing two
additive models** — the representation is never asked to express an interaction.

This reframes H3 from "can pLMs compose?" to the sharper and more defensible **"does the
scoring protocol project away structure the representation actually contains?"** Tsui's
Walsh–Hadamard R² of 0.72/0.66 suggests it does.

### F6 — Three claims we must NOT make

Recorded because each is refutable as stated and would hand a reviewer an easy kill:

1. **Not "inverse scaling."** Paired bootstrap on ESM-2 650M − 15B gives +0.0130, 95% CI
   [−0.0015, +0.0273] — not significant. The correct wording is **"saturates at ~650M"**.
   (Two confounds strengthen saturation: the 15B ran 270K steps vs 500K for its smaller
   siblings, and ESM-2's own Tables S1/S3 show it merely tying the 3B.)
2. **Not a blanket "mirage."** Candido et al. (EvolutionaryScale) report wet-lab-validated
   nanomolar binders. The claim must be scoped to **distribution-bounded**.
3. **Not "pLMs learn nothing beyond additivity."** Dieckhaus shows ESM-1v joint masking
   beating its own additive mode (0.08 vs 0.05), and RITA/Progen2/EVE score non-additively
   yet still fall to baseline — so additive scoring is not the diagnosis.

### F7 — ESM-2 collapses under alphabet relabeling, but the reason is NOT what it first looked like (2026-08-05, IN PROGRESS)

**Status:** CONFIRMATORY for the m-sweep (registered in protocol Amendments 3–4). **Not yet
reportable** — two registered controls are still running and one has already qualified it.

ESM-2 650M categorical Jacobian, contact-set self-consistency vs wild-type (Potts reference
= 1.000 exactly; chance ≈ 0.02). First protein of nine:

| m | random | In-class | Cross-class |
|---|---|---|---|
| 2 | 0.525 | **0.980** | 0.354 |
| 6 | 0.141 | **0.879** | 0.020 |
| 10 | 0.010 | **0.818** | 0.010 |
| 20 | 0.010 | **0.828** | — |

**The headline is the In-class/random gap, not the collapse.** At m=20 — every
biochemically groupable residue type relabeled — In-class swaps retain **0.828** while
random swaps fall to **0.010**, *below* the 0.021 chance line. So ESM-2 is not a pure
identity-keyed lookup table: it has a real biochemical similarity metric, and I↔V costs it
almost nothing. What it lacks is the alphabet-abstract computation that inverse covariance
performs exactly and for free.

**Confound 1 — edit distance (C6), already qualifying this.** First C6 cell (cytochrome_c,
random m=2, Hamming d=10): permuted **0.840**, matched-Hamming uniform substitution
**0.795**, composition-matched **0.853**. Within 0.06 of each other. At low m the
degradation is largely an **edit-distance effect**, not an identity effect. Registered in
advance as a possible refutation, and at low m that is what it is showing.

This is why the *unconfounded* comparison is In-class vs random at matched m: both arms
relabel the same number of residue types, so the contrast is internal to ESM-2 and does not
depend on the Potts baseline at all.

**Confound 2 — Potts refits, ESM-2 does not (C7).** The Potts model is refit on the permuted
alignment; ESM-2 gets no relabeled training data. A naive head-to-head is rigged. The
defensible claim is narrower and better: *Potts's coevolutionary knowledge is recomputed
from the input and therefore transfers; ESM-2's is amortised into weights keyed to one
alphabet convention and therefore does not.* C7 (MSA Transformer, which reads an alignment
at inference exactly as Potts does) removes the asymmetry and can decide this.

### F8 — H4 is refuted: nothing separates pLMs from alignment models on this axis (2026-08-05)

**Status:** NEGATIVE. Two differently-powered designs, both registered in advance, both
failing the same registered prediction.

**Design 1 (per-variant rank error, primary).** Every model's 95% CI on
`D = |β_k| − |β_sev|` includes zero; pLM vs alignment gives Mann–Whitney z = 1.10. Null.

**Design 2 (within-cell Spearman, Amendment 1).** Aggregating the DV worked exactly as
intended — effect sizes rose from ≈0.005 to ≈0.3 and most CIs now exclude zero. But the
result **refutes** H4 rather than supporting it:

| | median D | |
|---|---|---|
| pLM | **+0.270** | |
| alignment | **+0.242** | Mann–Whitney z = 1.85, n.s. |

H4 predicted `D > 0` for pLMs and `D < 0` for alignment models. Instead `D > 0` for
*everything*, including GEMME (+0.344), DeepSequence (+0.284) and EVE (+0.223) — the
alignment-based models that were supposed to show the opposite sign. In the k-rich subset the
family difference reverses outright (z = −0.29).

**Per the pre-registered falsification, H4 is abandoned rather than re-cut a third time.**

**The universal D > 0 is a data artifact — confirmed, not conjectured.** A k effect identical
across pLMs and alignment models is better explained by the DV than by anything models do.
Direct test over all 69 multi-mutant assays
(`experiments/H4-format-severity/results/diag_range_restriction.txt`):

    corr(k, within-k DMS std): median -1.000, mean -0.521; negative in 77% of assays

DMS dynamic range **shrinks as k grows** — higher-order mutants are more often uniformly
dead. Range restriction mechanically depresses within-cell Spearman at high k for *every*
scorer, reproducing the observed `D > 0` with no model-specific explanation. The aggregated
design measures a property of ProteinGym, not of protein language models.

**This confound propagates.** Any analysis that stratifies ProteinGym by mutation depth
inherits it — including the `Depth_k` columns widely cited as evidence about higher-order
epistasis. Combined with F5 (the scoring protocol is additive by construction), **two
independent artifacts sit underneath the standard depth-stratified epistasis analysis**, and
both must be controlled before H3 is attempted.

**Why the controls make this trustworthy rather than a suspected bug:** C4 (sign convention)
asserted before any run; C2 (shuffled-severity null) gave 0.0158 vs 0.0984 real, a 6× gap, so
the severity term measures something real; C1 excluded nothing, so k and mean severity are
genuinely near-orthogonal.

**The design flaw I should have caught.** 58 of 69 multi-mutant assays have k_max = 2, so `k`
ranges over {1,2}. `data/MULTI_MUTANT_INVENTORY.md` said so and I had already read it. C1
checked *collinearity* between the IVs but never checked whether either had enough
*variance* to carry a coefficient. **Lesson: a collinearity control is not a variance
control.**

### F9 — C6 complete for cytochrome c: Hamming distance is not the controlling variable (2026-08-05)

**Status:** CONFIRMATORY. Protein 1 of the C6 sweep complete; protein 2 running. ESM-2 650M,
cytochrome c, L=104, chance ≈ 0.012. `perm` = bijective relabeling; `unif` = uniform random
substitution at the **same positions and same Hamming distance**; `comp` = the same but
composition-matched.

| condition | Hamming d | permuted | uniform | composition |
|---|---|---|---|---|
| random m=2 | 10 | 0.840 | 0.795 | 0.853 |
| random m=6 | 16 | 0.532 | 0.372 | 0.385 |
| random m=10 | 55 | 0.218 | 0.128 | 0.058 |
| random m=20 | 104 | 0.000 | 0.019 | 0.013 |
| In-class m=2 | 20 | 0.904 | 0.853 | 0.840 |
| In-class m=6 | 36 | 0.859 | 0.673 | 0.603 |
| **In-class m=10** | **60** | **0.731** | **0.026** | **0.006** |
| **In-class m=20 (realized 15)** | **76** | **0.699** | **0.013** | **0.013** |

**Correction to what I wrote from the random arm alone.** I previously headed this finding
"the collapse is NOT merely edit distance" on the strength of the random rows, where
permutation is *spared* relative to matched substitution (0.218 vs 0.128 at d=55). That
reading was wrong in a way that matters. If a bijective relabeling damages the model *less*
than arbitrary edits of the same size, then the observed degradation is bounded above by a
generic edit-distance effect — which makes edit distance a **sufficient** explanation for the
magnitude, not an excluded one. The random arm alone does not close the confound.

**The In-class arm does close it, decisively.** Compare the middle of the table at
constant Hamming distance:

| intervention | Hamming d | overlap |
|---|---|---|
| uniform substitution | 60 | 0.026 |
| random relabeling | 55 | 0.218 |
| In-class relabeling | 60 | **0.731** |

Three interventions that change 55–60 of 104 positions produce outcomes spanning **0.026 to
0.731, a factor of 28**. At d=76 the In-class relabeling still holds 0.699 against 0.013
for its own matched control — a factor of 54, at a Hamming distance *larger* than the random
m=10 cell that took the model to 0.218. **Hamming distance is not the controlling variable;
what controls the outcome is whether the substitution preserves biochemistry.**

That is the precise form the artifact control has to take (F16): not "the permuted sequence is
different, therefore harder", because a *larger* perturbation that respects biochemistry is
nearly free. It is the same structure C9 shows against experimental ground truth on ubiquitin,
reached by a different route.

One row to report rather than smooth over: at random m=20 the relabeling gives **0.000** while
its matched controls give 0.019 and 0.013. All three are at the chance floor for L=104
(≈0.012), so the ordering there carries no information.

---

### F10 — The In-class regime cannot reach the whole alphabet (2026-08-05)

**Status:** MEASURED, and it is a fact about amino acids rather than a bug.

Auditing `src/permute.py` against the figure showed the requested `m` is not always the
delivered `m`. Counting non-fixed points of the permutations the runs actually used
(seeds 0, 1, 2):

| regime | asked 2 | asked 6 | asked 10 | asked 20 |
|---|---|---|---|---|
| random | 2 | 6 | 10 | **20** |
| In-class | 2 | 6 | 10 | **15** |
| Cross-class | 2 | 6 | 10 | **14-16** |

Every cell the headline rests on (m = 2, 6, 10) is faithful in all three regimes. Only m=20
was mislabelled, and only for In-class and Cross-class.

**Why.** `In-class_GROUPS` (IVLM, KR, DE, NQ, ST, FYW) covers 15 of 20 residues. A, C, G,
H and P have no biochemically similar partner to swap with, so **a fully In-class
relabeling of the complete alphabet does not exist.** `Cross-class_perm` hits the same ceiling
for a different reason: its pair list reuses residues across pairs, so a non-conflicting
matching saturates near 15.

**Consequence for the figure** (registered as protocol Amendment 7 before the corrected plot
was produced): every point is drawn at its *realized* m, recomputed deterministically from
the seed formula the runs used. The In-class curve therefore ends at 15 and there is no
In-class point at 20. That is more informative than the flat 10 to 20 segment it
replaces, which was the builder saturating rather than the model coping — a reviewer who
noticed that before we did would have been entitled to distrust the rest of the panel.

**What it does not license.** It is not a reason to drop the m=20 random cell. That one is
genuine, it is where ESM-2 reaches chance, and random is the only regime that can rewrite the
whole alphabet — which is exactly why it carries the primary claim.

### F11 — C7 COMPLETE: MSA Transformer also fails, with the asymmetry removed (2026-08-05)

**Status:** CONFIRMATORY. Registered as protocol Amendment 5, which called this "the most
serious threat to H2 and it is of our own making, not a reviewer's invention."

The threat was real: Potts is *refit* on the permuted alignment while ESM-2 gets no relabeled
training data, so "Potts invariant, ESM-2 collapses" partly restates that one model derives
knowledge at inference and the other has it frozen. MSA Transformer removes the asymmetry —
it reads an alignment at inference, exactly like Potts, and neither is retrained.

4 Pfam families (PF00018, PF00027, PF00072, PF00076), 136 cells, top-L/2 long-range
contact-set overlap against each model's own unpermuted map:

| regime | m=2 | m=6 | m=10 | m=20 |
|---|---|---|---|---|
| In-class | 0.935 ± 0.013 | 0.795 ± 0.023 | 0.640 ± 0.057 | 0.536 ± 0.086 |
| random | 0.775 ± 0.038 | 0.471 ± 0.067 | 0.244 ± 0.061 | **0.147 ± 0.056** |
| Cross-class | 0.687 ± 0.033 | 0.381 ± 0.059 | 0.239 ± 0.080 | 0.150 ± 0.049 |

SEM over families. Chance is 0.013–0.046 depending on length.

**Potts on the same permuted alignments: exactly 1.000000 in all 136 cells** — min = max =
mean, not "approximately invariant". That is an in-run assert, so every MSA-T number above is
certified to come from an alignment that is statistically unchanged.

**This is the strong branch of the registered interpretation.** A neural model holding the
alignment in its hands, retrained on nothing, still cannot do the alphabet-agnostic
computation that inverse covariance does exactly and for free. H2 does **not** have to narrow
to single-sequence pLMs.

**Two honest qualifications.**
- MSA-T degrades severely but does **not** reach chance (0.147 vs ~0.031). The defensible
  word is *severe degradation*, not *collapse*. ESM-2 650M does reach chance; MSA-T does not.
- The In-class-vs-random dissociation replicates here (0.536 vs 0.147 at m=20), so the
  "encodes biochemistry, not alphabet abstraction" reading is not an artifact of the
  single-sequence setting.

### F12 — Ground-truth contacts are in place, and ESM-2 is genuinely good before permutation (2026-08-05)

**Status:** INFRASTRUCTURE + a validity check the paper needs.

`src/structures.py` (numpy only, no new dependencies) fetches and parses RCSB structures,
builds CB–CB < 8 Å contact maps, aligns them to our query sequences and scores top-L/2
long-range precision. All four targets pass the pre-set sanity bar (≥90% of query positions
mapped; long-range contact count 0.5–2× chain length):

| target | PDB | mapped | long-range contacts | identity |
|---|---|---|---|---|
| ubiquitin | 1UBQ_A | 100% | 111 | 100% |
| protein_G_B1 | 1PGB_A | 100% | 79 | 100% |
| cytochrome_c | 1HRC_A | 100% | 129 | 88.5% |
| lysozyme_T4 | 2LZM_A | 100% | 183 | 98.8% |

**The number that matters for the paper.** ESM-2 650M's categorical Jacobian, scored against
real structure: ubiquitin **P@L/2 = 0.763** (14× chance), protein G **0.500** (6× chance).
So the unpermuted maps are genuinely good, and the collapse we report is a collapse *from
competence*, not from noise. A deliberately scrambled residue mapping drops these to
0.107 / 0.158, i.e. chance — confirming the alignment is load-bearing rather than decorative.

**Two caveats that must appear in any writeup using these.**
- `cytochrome_c` in our `EXTRA` dict is the **human** sequence; 1HRC is **horse heart**
  (88.5% identity, 12 substitutions). Its contact map is an ortholog's. Either state this or
  swap for a human structure. `lysozyme_T4` is the cysteine-free pseudo-wild-type versus true
  WT 2LZM, differing only at T54C/A97C, which is harmless.
- A silent-corruption bug was found and fixed during construction: plain Needleman–Wunsch
  with linear gaps lets a deletion fragment into scattered gaps at *identical* score,
  producing a mapping that is 100% identical at every mapped position while anchoring
  residues onto the wrong copies of a repeated letter. Fixed with a Gotoh three-state
  recursion plus a provably score-preserving contiguity surcharge. This is the second time on
  this project that a check passed while being wrong (see Lessons) — the pattern is that
  agreement on the values you inspect does not certify the indices you did not.

### F13 — The collapse is real predictive loss: 14x chance to chance (2026-08-05)

**Status:** CONFIRMATORY, registered as protocol Amendment 8. **COMPLETE — 26 cells, both
targets, both regimes.** Full table below the ubiquitin-random headline.

This restores the locked protocol's **original** dependent variable. Self-consistency was a
substitute adopted because no structures were on disk, and it has a known weakness: it cannot
tell a model that degrades from one that was never right, and it leaves open the reading
"maybe it finds the same contacts by another route." True precision closes that.

ESM-2 650M categorical Jacobian, long-range (|i−j| ≥ 12) P@L/2 against CB–CB < 8 Å from
1UBQ_A (100% coverage, 100% identity, 111 true long-range contacts, chance 0.053):

| condition | P@L/2 | × chance | Δ |
|---|---|---|---|
| unpermuted | **0.763** | 14.4× | — |
| random, m=2 | 0.829 | 15.5× | +0.066 |
| random, m=10 | 0.461 | 8.6× | −0.303 |
| random, m=20 | **0.053** | **1.0×** | −0.711 |

**The model goes from 14× chance to exactly the base rate** under a transformation that
provably leaves every alignment statistic unchanged, and to which the classical estimator is
invariant *by theorem* (see `theory.md`, Propositions 1–2).

The registered identity gate passed to three decimals (0.763 measured vs 0.763 expected from
the independent structure-module validation), so this is not a pipeline artifact.

**Three things worth noting honestly.**
- m=2 shows **no damage at all** (0.829 ≥ 0.763, within seed noise). This is a genuine
  dose–response, not brittleness to any perturbation whatsoever — which is the more
  defensible and more interesting claim.
- m=10 has large seed variance (±0.303 over 2 seeds): one permutation retained much more
  than the other. With only 2 seeds this cell is weak and should be reported with its spread
  rather than as a point estimate, or given more seeds.
- n_eval = 38 pairs at top-L/2 for L=76. Small, and the CI on a precision estimate from 38
  pairs is wide. The m=20 cell is nonetheless unambiguous because it sits *on* the base rate.

**Consequence for how the paper is built.** The headline becomes a precision number against
experimental structure, and self-consistency demotes to the control it always was — the one
that closes "the permuted sequence is simply a harder input", which precision alone cannot.

#### Complete C9 table — and the dissociation is protein-dependent

| | ubiquitin (1UBQ_A, chance 0.053, n_eval 38) | protein G B1 (1PGB_A, chance 0.080, n_eval 28) |
|---|---|---|
| **identity** (gate PASS both) | **0.763** (14.4×) | **0.500** (6.3×) |
| random m=2 | 0.829 (15.5×) | 0.625 (7.8×) |
| random m=10 | 0.461 ± 0.303 (8.6×) | 0.054 ± 0.054 (**0.7×**) |
| random m=20 | **0.053** (**1.0×**) | 0.107 (1.3×) |
| In-class m=2 | 0.868 (16.3×) | 0.607 (7.6×) |
| In-class m=10 | 0.882 (16.5×) | 0.268 (3.4×) |
| In-class m=20 → **realized 15** | 0.855 (16.0×) | 0.071 (**0.9×**) |

**On ubiquitin the dissociation is total and it is the cleanest result in the project.**
In-class relabeling of 15 of 20 residues does **not damage the model at all** — 0.855 vs
an unpermuted 0.763 — while random relabeling of 20 takes it to exactly the base rate. Same
edit distance regime, same number of tokens changed, opposite outcome. This is a stronger
version of the In-class/random contrast than the self-consistency readout gives, and it
is precisely the artifact control F16 says must lead: *the relabeled sequence is not simply a
harder input, because a relabeling of the same size that respects biochemistry costs nothing.*

**On protein G it is only partial, and that must be reported.** In-class degrades more
slowly than random (0.268 vs 0.054 at m=10) but still reaches chance by realized m=15
(0.071 = 0.9× chance). So "In-class is harmless" is a ubiquitin statement, not a general
one. Protein G is also the weaker starting point — identity 0.500 at 6.3× chance versus
ubiquitin's 14.4× — and has n_eval = 28, where one hit is worth 0.036. The honest general
claim is **In-class degrades substantially less than random at matched m**, with
ubiquitin as the extreme case rather than the typical one.

Two further honesties. Every m=2 cell, in both regimes and both proteins, comes out *above*
identity (0.829, 0.868, 0.625, 0.607). With n_eval of 28–38 a single pair is worth 0.026–0.036,
so these are small-sample fluctuations and should not be read as improvement. And protein G's
random arm is **non-monotone** (0.054 at m=10, 0.107 at m=20); both cells sit at the floor, so
the ordering there is noise.

### F14 — The invariance theorem's conditions are load-bearing, and one control was measuring a blind spot (2026-08-05)

**Status:** THEORY + verification. `theory.md`, `code/test_theory_conditions.py`.

The Potts invariance is now proved rather than asserted: Proposition 1 for maximum-entropy
Potts, Proposition 2 for the regularised inverse-covariance estimator we actually run, where
relabeling acts as orthogonal conjugation by `Q = I_L ⊗ P_π` and every scoring step
(mean-centring, symmetrisation, per-block Frobenius, APC) commutes with it. The corollary is
the paper's actual claim: **a model whose contact predictions change under a global
relabeling is not computing a function of alignment column statistics.** That is a *necessary
condition*, so it needs no claim about what the model represents instead.

A proof that names conditions invites the question of whether those conditions are real, so
both are tested in both directions rather than only the happy path:

| condition | isotropic / guarded | violated |
|---|---|---|
| (i) ridge must be isotropic | max\|ΔS\| = **4.9e-15** | per-amino-acid ridge spanning only 1.0–1.5× λ → **0.355**, 11.8% of score scale, top-L/2 overlap **0.850** |
| (ii) non-standard chars excluded | max\|ΔS\| = **1.8e-15** | `gap → K` on an MSA with 10% gaps + 1.5% `X` → **0.182** |

Verdict line from the run: *all stated conditions are LOAD-BEARING and verified*. The
isotropy requirement is not bookkeeping — a mildly non-uniform regulariser destroys
exactness — and the `PROTECTED` guard is not defensive coding, it is a hypothesis of the
proposition.

**Condition (ii) exposed an error in my own control, not in the theory.** `mrf.ALPHABET` is
`AA20 + GAP` and `one_hot_msa` folds *every* non-standard character into the single gap
state. So swapping `-` with `X` is **invisible to the estimator** — both already occupy the
same slot — and a control that observes "invariant" there is measuring the encoder's blind
spot rather than the theorem. The guard forbids *protected ↔ standard* mappings, which move
probability mass between slots. Test corrected to exercise `gap → K`.

That is the third time on this project that a check passed while being wrong. The unifying
lesson, now in Lessons: **agreement on the values you inspect does not certify the indices,
slots, or encodings you did not.**

### F15 — Diagnosis alone clears the bar, but only if the diagnostic is shipped as an instrument (2026-08-05)

**Status:** VENUE CALIBRATION, from primary sources. `literature/iclr2026-plm-exemplars-STRUCTURAL.md`.

I had been treating "does this need a constructive half?" as an open question and leaning
toward yes. The evidence says no, and it says so from the only sample that matters: the four
genuinely analysis-driven pLM papers accepted to **ICLR 2026 main conference** (out of ~49
accepted bio/protein papers — the rest are method/generative). Venue confirmed two ways:
OpenReview `venueid == ICLR.cc/2026/Conference`, and `\iclrfinalcopy` uncommented in three of
the four arXiv sources.

**The existence proof is thinner than our current evidence base.** *Towards Understanding the
Shape of Representations in PLMs* is **two experiments**, 0 tables, **two results figures in
the entire main body** (two of its four main figures are schematics), no related-work
section, no limitations section, no error bars, no seeds, no significance test — against a
null its smallest models nearly lose to. Its one actionable recommendation was tested, failed,
and is disclosed in a subordinate clause: *"Initial attempts to show this with linear models
or small networks failed to generalize (data not shown)."*

What bought it was not the finding. It was an **instrument** that makes a previously ill-posed
question well-posed — square-root-velocity representation so variable-length proteins share a
sphere, and a *k*-filtration replacing an Ångström threshold that does not exist in
representation space. **That reframes what we are shipping.** The alphabet-equivariance test
is exactly this kind of object: a checkable necessary condition with a proof, a guard, a
positive control and a negative control. The paper should present it as *a named, reusable
diagnostic with a protocol*, and the ESM-2 collapse as its first application — not as one
experiment that happens to have controls.

**The constructive half is a choice, not a requirement.** Both diagnosis+fix papers in the
sample land at ~30/65, and in both the diagnosis is structurally demoted — Reverse
Distillation runs **zero** diagnostic experiments (five citations substitute for a scaling
curve), and Controlling Repetition puts **zero** diagnostic experiments in its Experiments
section and still carries `\label{sec:preliminaries}` on the section it promoted to a
contribution. There is no 50/50 paper in the sample. Critically, **neither paper ever argues
that diagnosis would have been insufficient** — the transition is asserted by sequencing
("the first step is…"). So: no failure-predictor half. Fork, don't blend.

**Our baseline situation is inverted relative to all four, and that is the strongest card.**
In every one of these papers the simple baseline is absent (Reverse Distillation runs no
one-hot, no BLOSUM, no PSSM, no HMM, no MSA method), or wins and is absorbed by adverb
("generally", "nearly always"), or wins and is neutralised by introducing a new axis (the
utility-constraint checkmark), or wins and is never mentioned at all (a generalist model
beating every specialised Sci-LLM). We have a baseline that satisfies the property **exactly,
by construction** — the same rhetorical asset Reverse Distillation had to buy with Theorem 1,
except ours is free and provable.

Three craft rules taken directly from the sample:

- **No numerals in the abstract.** None of the four abstracts contains a single number. All
  lead with a phenomenon or a gap. Two coin a named pathology in sentence two and then run
  the coinage through every section title.
- **Appendix mass, not main-text mass, is the bar.** Shape: 8 of 12 figures in the appendix.
  Controlling Repetition: 26 of 28 tables. Our C6/C7/C8/C10 belong there.
- **Pre-empt the three predictable asks**, which the rebuttal-added appendices in the sample
  answer verbatim: *mechanism* (where in the model does it break), *generalisation* (does it
  hold across families/architectures), *artifact* (is it just the tokenizer or the decoding
  procedure). Ours: layer/head localisation; ESM-2 + MSA-Transformer + a third; and the
  edit-distance and composition controls.

**Update from the second exemplar set** (`literature/iclr2026-diagnostic-exemplars-STRUCTURAL.md`,
five ICLR 2026 main-track diagnostic papers across domains). Two craft moves we already had
and were not using as such:

- **Prove an identity on the computed output, not an optimality property.** *t-SNE exaggerates
  clusters, provably* rests its whole cluster-salience half on a two-line invariance lemma, and
  the decisive choice is that it proves **set equality of all stationary points** rather than
  something about optima — which makes *"but real pipelines run gradient descent"*
  **structurally unavailable** rather than something to argue against. Our Proposition 2 is
  already this: `S^π = S` is an equality between two matrices a closed-form computation emits,
  with no solver, no convergence and no initialisation. Proposition 1, which *is* an optimality
  statement, is the weaker one and must not be what the result rests on. Written up in
  `theory.md` under "Why Proposition 2 carries the paper".
- **Universal quantification over the nuisance parameter.** t-SNE proves failure for *all*
  perplexities at once instead of sweeping a grid. Proposition 2 holds for **every** π in the
  symmetric group on 20 residues — all 20! ≈ 2.4 × 10¹⁸ — simultaneously, because nothing in
  the argument depends on which permutation matrix `P_π` is. The sampled sweep is an
  illustration, not the claim, and a reviewer cannot ask about a permutation we did not try.

And the cautionary case, which is the sharpest evidence for putting the theorem first: the EEG
paper's ratings were **8/8/4/2**, and **no reviewer raised the tuning objection its authors had
carefully pre-empted** — zero hits for hyperparameter/sweep across all 21 notes. The entire
fight was novelty. Eighteen classical baselines, eleven datasets and 50,000-resample
permutation testing did not substitute for a theorem. Its abstract is still worth copying
structurally: sentence 1 grants the promise, sentence 2 pivots on "Yet" and names the
comparison class in an em-dash aside — *"particularly classical non-neural approaches."*

**One caution that changes budget allocation.** The revealed rigor standard in this sample is
low — no error bars against one's own null, a causal claim whose two variables are never
correlated, single-run results cherry-picked against visible counterexamples in the authors'
own tables. Our controls already clear it comfortably. **Remaining effort should go to
framing and to the reusability of the instrument, not to more rigor.**

### F16 — 14,164 decided ICLR 2026 submissions: the theorem is the headline, not the support (2026-08-05)

**Status:** VENUE CALIBRATION, quantitative. `literature/iclr2026-diagnostic-genre-reviews.md`,
`literature/iclr2026-review-threads/`, `data/openreview/`.

OpenReview is Cloudflare-gated from this host, so the numbers come from the HuggingFace mirror
`JerMa88/ICLR_Peer_Reviews_2026` — **76,199 verbatim review notes across 19,814 submissions**.
**Known gap: the mirror carries Official_Review notes only, no author rebuttals**, so nothing
below is a rebuttal→score-movement claim.

Accept rate by mean rating (0/2/4/6/8/10 scale): 4.0 → 11%, 4.5 → 28%, **5.0 → 53%**,
5.5 → 78%, 6.0 → 93%. Overall **37.8%**.

| bucket | n | accept |
|---|---|---|
| all submissions | 14,164 decided | 37.8% |
| diagnostic / negative-framed abstract | 274 | 39.8% |
| diagnosis **with** a proposed fix | 84 | 36.9% |
| **pure diagnosis, no fix** | 190 | **41.1%** |
| **diagnosis + a theorem** | — | **47.4%** — best bucket in the corpus |

This independently confirms F15 and sharpens it: **adding a fix is measurably *worse* than
not**, because it opens a second novelty attack surface. And the single best predictor
available is the one asset we already have.

**The kill that matters is not the one I was defending against.** Reading 11 rejected papers
in full, the dominant objection (6/11) is *"your intervention could itself explain the
effect"* — VAR-MATH, a near-identical invariance intervention on AIME, died at mean 2.67 on
exactly this: *"It could very well be the case that the mutated questions are simply harder,
causing the score drop."* An explicit demand for a fix appears in only **2 of 11** and is the
weakest recurring kill.

But the objection that ends diagnostic papers with strong empirics is **"this is measurement,
not a contribution."** The EEG-foundation-model paper ran 6 model families, 11 task-datasets,
**18 classical non-neural decoders**, and permutation tests at 50,000 resamples with
Bonferroni correction — and still took a 2, with the AC meta-review recording that it *"reads
more like a benchmarking/engineering study than an ICLR-level research contribution."* The
papers in the same set that never faced this objection are the two with a theorem.

**Consequence for our structure: Proposition 2 leads.** It is not support for the empirical
sweep; it is the thing that makes the sweep a research contribution rather than a benchmark.

**Three additions the evidence makes non-optional**, each traceable to a specific kill:

1. **An artifact control ruling out "the renamed alphabet is simply harder."** This is
   verbatim what killed VAR-MATH. Our instrument is the In-class-vs-random contrast at
   matched realized *m* (C6, C7) — and it must **lead**, not sit in an appendix.
2. **A comparator that is not a refit classical model**, so the result cannot be read as
   "the baseline got retrained and the neural model didn't." The TSFM rejection is our exact
   risk: *"leaving unclear whether TSFMs are uniquely fragile or simply representative of
   general deep forecasting vulnerabilities."* We have two answers: MSA-Transformer reads an
   alignment at inference and is retrained on nothing (C7), and Proposition 2 shows the Potts
   refit produces a **bit-identical** score matrix, so refitting is not doing any work.
3. **A named consequence.** In protein specifically, "pLMs memorise more than they
   generalise" is *already priced in as unsurprising* — a reviewer said exactly that while
   rejecting a paper with real wet-lab data. And the DFT counter is the shape of the attack
   we will face: *"standard KS-DFT also does not correctly model the H2 dissociation limit,
   yet DFT is widely used in production."* Our answer must be stated in the abstract, and the
   defensible ones are: **the test is label-free and structure-free** (one sequence, L×20
   forward passes, no ground truth) so it is an audit any lab can run on any new pLM; and
   **scale does not close the gap** (C8: 8M→3B buys biochemistry-sensitivity, not alphabet
   abstraction), which is the "do not blindly upscale" consequence reviewers credited in the
   EEG thread.

**Scope is not the gate.** Mirage-or-Method was accepted at mean 6.0 on **two models at one
scale** with 3 of 4 reviewers flagging narrowness; a rejected paper ran 34 VLMs and another
ran 10 models / 4 families / 5 datasets, was praised for *"commendable experimental breadth"*,
and scored 3.33. Scope complaints are a review reflex; **comparator class** is what gates.

**Our specific exposure — synthetic intervention — is survivable but must be framed.**
"Unlikely to occur naturally" appears in 22.6% of rejected reviews and 17.9% of accepted ones
(ratio 1.26 — near-useless as a discriminator). Illusion-of-Diminishing-Returns took the
harshest possible version (*"an extremely synthetic and narrow task… a more fundamental flaw
that challenges the paper's entire premise"*) and finished at **mean 6.0**, because other
reviewers credited the same design for what it *bought*: two variables decoupled that were
otherwise confounded. **The artificiality must be presented as the instrument that isolates
the confound, in those words, in the abstract.**

### F17 — C10: the crossing. ESM-2 falls below the classical model it replaced (2026-08-05)

**Status:** CONFIRMATORY, protocol Amendments 10 and 10a. **COMPLETE, both targets.**
`results/h2_c10_potts_crossing.csv` (14 rows).

C9 said ESM-2 falls to chance. It could not say whether chance is bad relative to anything a
practitioner would use. C10 answers that on the same target, the same structure, and — this
is the design point — the **identical candidate pair set and identical k** for both models.

Ubiquitin / 1UBQ chain A. The ProteinGym `RL40A_YEAST` alignment (depth 16,228) covers human
ubiquitin residues **10–76** at 95.5% identity — differing at exactly the three known
positions P19S / E24D / A28S. Residues 1–9 are outside the MSA and are dropped for **both**
models by setting `mapping[i] = -1`, leaving 1,540 candidate pairs, 70 true long-range
contacts, and a chance rate of 0.0455.

| | P@L/2 | × chance | × Potts |
|---|---|---|---|
| chance (base rate) | 0.0455 | 1.0 | 0.14 |
| **Potts, under *any* relabeling** | **0.3158** | 6.9 | 1.00 |
| ESM-2 650M, identity | 0.7368 | 16.2 | **2.33** |
| ESM-2 650M, m=20 random (2 seeds) | 0.0526 | 1.2 | **0.17** |

Potts is bit-identical across relabelings — max\|ΔS\| = 2.3e-14 and 1.8e-14, an in-run assert,
which is Proposition 2 rather than a finding. The full-length ESM-2 identity number is 0.7632,
reproducing C9's 0.763 exactly, so the restriction to 67 of 76 residues costs 0.026 and is not
hiding anything.

**The sentence the paper is built on:** under a relabeling that provably leaves every column
and column-pair statistic unchanged, ESM-2 650M goes from **2.3× the precision of a 20-line
inverse-covariance Potts model to 0.17× it**, while the Potts model does not move at all.

Two caveats to carry. `src/mrf.py` uses a plain ridge with **no sequence reweighting**, so our
Potts is a floor rather than a tuned baseline — declared in Amendment 10 before the run, and
it cuts *against* the crossing (a stronger Potts widens the gap) while cutting *for* the
claim that ESM-2 legitimately beat it at identity. And n_eval = 38 pairs; the two permuted
seeds are 0.0789 and 0.0263, straddling chance.

#### Second target (Amendment 10a): the crossing replicates, more weakly

Amendment 10 limited C10 to ubiquitin on the stated ground that only one target had both a
deep MSA and a 100%-identity PDB chain. **That was wrong and I had not checked it.** Protein G
B1 is 100% identical to 1PGB_A, and ProteinGym ships `SPG1_STRSG_full_b0.1.a2m` at depth
3,109. The B1 domain sits at match columns 226–281 of the full-length 448-residue query, at
96.4% identity to our B1 sequence (it is a neighbouring repeat, differing at 2 of 56).

| | ubiquitin | protein G B1 |
|---|---|---|
| MSA depth (rows kept) | 16,228 | 3,043 of 3,109 |
| chance | 0.0455 | 0.0798 |
| **Potts, under any relabeling** | 0.3158 (6.9× chance) | 0.1786 (**2.2× chance**) |
| ESM-2 identity | 0.7368 = **2.33× Potts** | 0.5000 = **2.80× Potts** |
| ESM-2 m=20 random | 0.0526 = **0.17× Potts** | 0.1071 = **0.60× Potts** |
| Potts max\|ΔS\| under relabeling | 2.3e-14 / 1.8e-14 | 1.1e-14 / 1.3e-14 |

**The crossing replicates: on both targets ESM-2 starts well above the classical baseline and
ends below it.** But the protein G version is much weaker and must not be quoted as if it
matched ubiquitin. Its Potts sits at 2.2× chance — barely clear of the withdrawal threshold
Amendment 10 set at 1.5× — on a 5× shallower alignment, and its permuted ESM-2 lands at 0.60×
Potts rather than 0.17×, with n_eval = 28 where one pair is worth 0.036 and the two seeds are
0.0714 and 0.1429. The defensible summary is: **the crossing holds on both targets, with a
margin that tracks how good the classical baseline is.**

### F18 — Readout 2 interim: the dissociation replicates on variant effect (2026-08-05)

**Status:** PARTIAL — 3 of 15 assays complete (a 4th is mid-run). **Do not cite the numbers**;
cite only that the pattern is present. `code/analyze_readout2.py` implements Amendment 9.

DV1, the pre-registered aggregate, over the first 3 assays (9 cells per condition):

| m | In-class | random | Cross-class |
|---|---|---|---|
| 2 | −0.008 | −0.058 | −0.107 |
| 6 | −0.041 | −0.249 | −0.228 |
| 10 | −0.090 | −0.295 | −0.308 |
| 20 | **−0.122** | **−0.333** | **−0.332** |

Two registered predictions are landing on a readout that has nothing to do with structure.
Prediction 2: ESM-2 Δρ is strongly negative and monotone in m under random relabeling.
Prediction 3: **In-class degrades ~2.7× less than random at m=20**, and Cross-class tracks
random rather than In-class — which is what "biochemistry is encoded, alphabet abstraction
is not" predicts, since Cross-class swaps also violate biochemistry.

Prediction 1 holds exactly: PSSM max\|Δρ\| is **0.0e+00** across every permuted cell. That is
Proposition 1 with `J = 0`, so it is a correctness assert on the pipeline rather than a result.

**The Amendment 9 risk is real and currently biting.** Only 1 of the first 3 assays clears the
`ρ_identity ≥ 0.20` gate (the other two sit at 0.030 and 0.066 — both reproducing ProteinGym's
published `ESM2_650M` value exactly, so this is the benchmark, not a bug). The analysis script
correctly suppresses DV2 and DV3 under the underpowered clause. The 4th assay's identity ρ is
0.737, so the rate over 4 is 2-in-4; if that holds, 7–8 of 15 clear and the gated analysis
survives. If it does not, only DV1 gets reported, exactly as registered.

## Patterns and Insights

- The distinction that matters throughout is **statistic-over-alignment** vs
  **lookup-keyed-on-identity**. Every experiment in this project is a variation on
  constructing an operation that preserves the former and destroys the latter.
- Preserving *information content* while changing *surface tokens* is the general recipe.
  It defuses the standard confound that sinks OOD papers — "the task just got harder" —
  because the true contact map, column entropies, and inter-column MI are all provably
  unchanged.

---

## Lessons and Constraints

- Stock cluster Python is **3.6.8** (torch 1.10, numpy 1.19). No PEP 585 builtin generics
  (`dict[str, str]`), no `from __future__ import annotations`. `src/mrf.py` and
  `src/permute.py` are deliberately kept 3.6-compatible so controls can run without the GPU
  environment. Experiment code that needs a pLM may assume the new env.
- **Precision@L/k is ceilinged** at n_true/n_pred. With few planted contacts it looks like
  a weak result when it is in fact perfect. Always report recall alongside it.
- bioRxiv returns 403 to naive fetches; PMC efetch + the Europe PMC `supplementaryFiles`
  endpoint works and also yields SI appendices.
- Run long scripts with `python3 -u` when redirecting to a file. Block buffering on a box
  loaded with background agents made a working script look hung for two minutes.
- **A control that passes on clean data proves nothing about dirty data.** The real-Pfam
  run reported `gaps=0.0%, non-standard=none` because our own >30%-gap column filter had
  stripped exactly the columns the test existed to exercise. Construct the awkward cases
  rather than hoping a download contains them.
- **A failing control's self-diagnosis can be wrong.** The adversarial positive control
  first reported "injected chars too rare, test has no power". The actual cause was that
  `apply_perm` enforces `PROTECTED` at the application site, so a deliberately broken
  permutation could not express the breakage. Read why a control failed before believing
  what it says about itself.

### F19 — A third architecture is loadable: AMPLIFY 120M, verified end to end (2026-08-05)

The generalisation objection (F16) needs a pLM that is not a Meta model on UniRef. AMPLIFY
120M is one — SwiGLU, RoPE, RMSNorm, a **27-token vocabulary in a different residue order**,
a different corpus — and the yearly checkpoints were already on disk. Loadability was
unverified: a CPU probe had timed out at 420s with no output. It is now verified.

`src/amplify_jac.py` loads it, and the probe passes:

```
loaded, params 118.3M
aa_idx [7, 25, 15, 11, 20, 8, 23, 14, 17, 6, 22, 19, 16, 18, 12, 10, 13, 9, 24, 21] mask 2
tokens (1, 78)   logits (1, 76, 20)
wildtype recovery (unmasked) 1.000
```

Wildtype recovery of 1.000 is the load correctness check that matters: at unmasked positions
the model reproduces the input residue at every one of 76 sites. A mis-mapped vocabulary or a
partially-loaded state dict does not do that. Note the residue ordering — AMPLIFY's `A` is
token 7 and `C` is token 25, nothing like ESM-2's contiguous block, which is exactly why the
architecture is worth testing and exactly what a hand-written index map gets wrong silently.

**Three blockers, recorded because each would cost an hour again.**

1. `AutoTokenizer`/`AutoModel` with `trust_remote_code=True` **hangs indefinitely** against the
   local checkpoints — observed twice, >7 min, zero output, no traceback, even under
   `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Bypass the Auto\* machinery entirely; the
   checkpoint ships `amplify.py`, `config.json`, `model.safetensors` and `tokenizer.json`.
2. `amplify.py` hard-imports `xformers`, which has no wheel for torch 2.11+cu128. Only two
   symbols are used. `SwiGLU`'s packed layout is confirmed by the checkpoint's own key shapes
   (`ffn.w12.weight [3424, 640]`, `ffn.w3.weight [640, 1712]`), and `memory_efficient_attention`
   is written to be the same computation as amplify.py's *own* CPU branch, which already calls
   `scaled_dot_product_attention` — so the shim agrees with the shipped code by construction
   rather than by approximation.
3. `from .rmsnorm import RMSNorm` needs `amplify.py` loaded as a package submodule. Synthesise
   a module with `__path__ = [ckpt_dir]` and the normal import machinery resolves it.

**The design point, not an implementation detail.** `AMPLIFYJacobian` subclasses
`jacobian.ESM2Jacobian` and overrides **only** `_tokenize` and `_logits`. Mean-centring,
symmetrisation, per-block Frobenius and APC are byte-for-byte the code that produced every
ESM-2 number. A cross-architecture comparison in which the readout also changes measures
nothing.

Registered as Amendment 11 and running as C11, against ESM-2 150M from C8 — 118M vs 148M is
the closest matched pair we have, so architecture and corpus vary while capacity roughly does
not. The amendment fixes a validity gate before the fact (AMPLIFY's *unpermuted* P@L/2 must
clear 3x chance on 1UBQ_A or 1PGB_A) and registers the refuting branch: if AMPLIFY turns out
invariant, that is the most interesting possible outcome and becomes the headline.

---

### F20 — C11: the violation and the dissociation both replicate in a third architecture (2026-08-05)

**COMPLETE, both registered predictions confirmed, validity gate passed on both targets.**
This closes the generalisation objection that F16 recorded as non-optional.

The worry was legitimate: ESM-2 (C8, five sizes) and MSA Transformer (C7) are both Meta models
trained on UniRef with the same 33-token vocabulary, so a reviewer could fairly read all our
evidence as one lineage rather than as a statement about pLMs. AMPLIFY 120M is a different
lineage — SwiGLU, RoPE, RMSNorm, a 27-token vocabulary in a **different residue order**
(`A`→7, `C`→25, nothing like ESM-2's contiguous block), a different corpus — and at 118M vs
148M parameters it is the closest capacity match to ESM-2 150M available. Architecture and
corpus vary; size roughly does not.

**Validity gate first, because Amendment 11 fixed it before the run.** AMPLIFY's *unpermuted*
long-range P@L/2 against experimental structure:

| target | P@L/2 | chance | ratio | verdict |
|---|---|---|---|---|
| ubiquitin (1UBQ_A) | 0.237 | 0.053 | **4.4×** | PASS |
| protein G B1 (1PGB_A) | 0.321 | 0.080 | **4.0×** | PASS |

Both clear the 3× threshold, so the permuted cells are reportable. Note AMPLIFY is a *much*
weaker contact predictor than ESM-2 650M (0.237 vs 0.763 on ubiquitin). That is the point of
having a gate rather than an eyeball: 0.237 sounds poor and is nonetheless 4.4× chance, which
is real signal with room to be destroyed.

**The matched-capacity comparison.** Same three proteins, same permutations, same seeds, and
— because `AMPLIFYJacobian` overrides only `_tokenize` and `_logits` — the same reduction code
byte for byte. Mean top-L/2 self-overlap over proteins × seeds:

| model | regime | m=2 | m=6 | m=10 | m=20 | chance |
|---|---|---|---|---|---|---|
| AMPLIFY 120M (118M) | random | 0.559 | 0.222 | 0.124 | **0.056** | 0.020 |
| AMPLIFY 120M (118M) | In-class | 0.770 | 0.671 | 0.493 | **0.414** | 0.020 |
| ESM-2 150M (148M) | random | 0.797 | 0.341 | 0.126 | **0.010** | 0.020 |
| ESM-2 150M (148M) | In-class | 0.892 | 0.767 | 0.666 | **0.578** | 0.020 |

**Prediction 1 confirmed.** AMPLIFY's contact predictions collapse under random relabeling,
monotonically in m, to 0.056 — 2.8× chance, from a baseline of 1.0. Per protein at m=20:
ubiquitin 0.026 (1.4× chance), cytochrome c **0.000** (not one contact in the top-L/2 survives),
protein G 0.143 (5.1×, and the noisiest cell in the table at ±0.107).

**Prediction 2 confirmed.** In-class degrades far less than random at matched realized m,
and the gap widens with m — ratio 1.4× / 3.0× / 4.0× / **7.3×** across m = 2/6/10/20. ESM-2
150M's corresponding ratios are 1.1× / 2.3× / 5.3× / 55.9×. The ESM-2 m=20 ratio is inflated by
a near-zero denominator (0.010) and should not be quoted against AMPLIFY's 7.3× as though the
two were commensurable; the honest statement is that **both architectures show the same
monotone widening, and both retain an order of magnitude more structure under In-class
relabeling than under random relabeling of the same size.**

**Why this matters more than a third data point.** Alphabet equivariance is a necessary
condition on *any* function of alignment column statistics (theory.md). The contrapositive
rules out a mechanism, and until now it ruled it out for one lab's models. It now holds across
two vocabularies, two tokenizations, two corpora and two feed-forward designs, at matched
capacity, measured by identical code. The remaining sceptical reading — "you found a quirk of
ESM" — is no longer available.

**One thing C11 does not show.** AMPLIFY is *not* uniformly more fragile than ESM-2. It is more
fragile at small m in the random regime and less collapsed at m=20 random (0.056 vs 0.010).
Nothing here supports ranking architectures by equivariance, and the paper should not try; the
claim is that both violate it, not that one violates it more.

---

### F21 — C8: scale buys biochemistry sensitivity and *zero* alphabet abstraction (2026-08-05)

**Complete for 8M / 35M / 150M / 650M. 3B hit the registered budget guard and is retrying.**
This is the evidence for the third of F16's non-optional additions — the *named consequence*,
the thing a reviewer gets to take away besides "the model is not doing what you thought".

Everything is held fixed except parameter count: same four proteins, same permutations, same
seeds, same DV, and each model compared only to itself, so the fact that bigger models predict
contacts better cannot confound the invariance measurement. Mean top-L/2 self-overlap over
proteins × seeds, n = 8 per cell, chance ≈ 0.023:

| size | regime | m=2 | m=6 | m=10 | m=20 | m=20 as × chance |
|---|---|---|---|---|---|---|
| 8M | random | 0.375 | 0.172 | 0.143 | **0.042** | 1.8× |
| 8M | In-class | 0.592 | 0.433 | 0.293 | **0.211** | 9.3× |
| 35M | random | 0.538 | 0.118 | 0.075 | **0.027** | 1.2× |
| 35M | In-class | 0.853 | 0.722 | 0.424 | **0.367** | 16.2× |
| 150M | random | 0.712 | 0.276 | 0.160 | **0.028** | 1.2× |
| 150M | In-class | 0.909 | 0.795 | 0.640 | **0.623** | 27.5× |
| 650M | random | 0.743 | 0.272 | 0.165 | **0.017** | 0.8× |
| 650M | In-class | 0.829 | 0.755 | 0.508 | **0.501** | 22.1× |

**Read the random column downward.** 1.8× → 1.2× → 1.2× → **0.8×** chance. Over an 80×
parameter range the collapse under random relabeling does not improve by any amount, and the
largest model is the most completely destroyed — 650M lands *below* chance, meaning its
permuted top-L/2 set is very slightly anti-correlated with its own unpermuted one. Whatever
capacity buys, it is not the equivariance that a function of alignment column statistics would
have for free.

**Read the In-class column downward and it does move.** 0.211 → 0.367 → 0.623 → 0.501, and
the In-class-minus-random gap grows +0.170 → +0.340 → +0.596 → +0.484. Larger models
distinguish a biochemically In-class relabeling from an arbitrary one far more sharply than
small ones do. That is a real capability and it appears with scale.

**So the consequence is specific, and it is not "scaling doesn't work".** Scaling buys
*sensitivity to biochemical similarity* — a genuinely useful thing — while buying *nothing at
all* on the axis that a coevolutionary computation would be invariant along. Those are
different axes, and conflating them is what "bigger models understand proteins better" does.
The defensible sentence for the abstract is: **more parameters make ESM-2 better at knowing
which residues resemble each other, and no better at treating the alphabet as arbitrary
labels.**

**The 150M → 650M dip is not evidence of anything.** In-class falls 0.623 → 0.501 and the
gap narrows. With n = 8 per cell and two seeds this is not a trend, and the honest reading is
that the In-class arm plateaus somewhere after 150M. Do not tell a story about it.

**Why 3B is missing, stated rather than hidden.** `run_c8_scale.py`'s registered guard caught
`CUDA out of memory ... 49.81 MiB free` with three other users holding ~38 GB, wrote the
complete 8M–650M table, and named the omission. The 5.7 GB checkpoint is cached, so only
memory was missing; `run_c8_3b_retry.py` polls for headroom at batch 4 and appends when the box
frees up, giving up after 10 hours rather than hammering a shared GPU. If it never lands, the
claim is made over 8M–650M and the range is quoted as 80×, not 375×.

---

### F22 — The venue publishes its own rubric, and it is not the ICLR bar we calibrated to (2026-08-05)

Fetched from icbinb-bio.github.io. This supersedes F16 **as the organising principle**, though
F16's craft lessons still apply *within* sections. We had been optimising against ICLR
conference proxies while the target venue publishes both a required structure and a five-point
scoring rubric.

**Full submissions must have four named sections: Problem, Proposed approach, Observed outcome,
and Reason for failure.** That is a submission requirement, not a suggested outline, and it is
a different spine from the one our findings accumulated in (theorem → measurement → controls).
Reorganise; do not append.

**The five scored criteria, verbatim:**

1. *Clarity* — "Is the biological task, modeling approach, and claimed failure mode clearly
   stated?"
2. *Technical Rigor* — "Are methods and conclusions well-supported by evidence? Are results
   documented carefully enough to reproduce?"
3. *Failure Analysis Depth* — "Does the paper go beyond reporting a bad number to characterize
   causes (model, data, experimental design, or deployment)?"
4. *Novelty of Negative Result* — "Does the paper offer new insight into where AI for biology
   fails or misleads?"
5. *Workshop Alignment* — "Does the paper present a reasonable argument aligned with failure
   modes of AI in biology?"

Plus an explicit anti-criterion: "Submissions that primarily report improved state-of-the-art
performance without meaningful analysis of limitations or failures should receive a lower score
on workshop alignment."

**The single most important consequence.** Criterion 3 asks for **causes**, and it is the one
criterion our current narrative under-serves. C6, C8 and C11 are written in findings.md as
*controls* — instruments for closing reviewer objections. Against this rubric they are the
**contribution**, because together they are a causal account: the contact map is produced by
residue-identity-conditioned pattern matching rather than column-statistic inference (C6 shows
it is not input difficulty; C8 shows scale sharpens the biochemistry axis and never touches the
abstraction axis; C11 shows it is not one lab's tokenizer). The paper must present them that
way. Section 4 of `paper/OUTLINE.md` does.

**Second consequence: C10 is not an appendix result.** Criterion 4 says "fails **or misleads**".
ESM-2 going from 2.33× Potts to 0.17× Potts on the identical candidate set is the misleads case
in a single number, and burying it forfeits the criterion.

**Third: F16's numbers were measuring a harder bar than we face.** Pure diagnosis accepted at
41.1% and diagnosis+theorem at 47.4% *at ICLR main track*. Here the guidelines actively
penalise papers that lead with SOTA and skip failure analysis. Our lack of a proposed fix is
not a liability at this venue — it is the format. This retires the last of the residual anxiety
recorded in F15 about shipping diagnosis alone.

**Format facts that constrain execution.** 8 pages full / 4 tiny, excluding references and
appendices; workshop LaTeX template; double-blind **and "linked material must also preserve
anonymity"**, so the audit tool ships as anonymised supplementary code and *not* as a GitHub
link; non-archival; concurrent submission welcome including to the NeurIPS 2026 main track;
deadline 2026-08-29 AoE.

---

### F23 — Review mining round 2: shipping the tool does NOT help, and our real exposure is "vacuous" (2026-08-05)

215 hand-confirmed score-raise events across 194 scraped ICLR 2026 threads, plus a 349-thread
corpus for the framing questions. Three results, two of which **overturn** decisions recorded
in F15 and F16. Full file: `literature/iclr-rebuttal-and-instrument-calibration.md`.

**1. "Ship the diagnostic as an instrument" is refuted as an acceptance strategy.** F15
concluded that diagnosis clears the bar "only if the diagnostic is shipped as an instrument."
The data does not support it:

- Conditional on the novelty objection, artifact language moves accept by **+2.0pp (p = 0.73)**.
- Papers whose reviewers *explicitly credited the artifact* accept at **36.4% (n = 99)** — the
  ICLR baseline. Being praised for the tool is worth nothing.
- **In 10 of 10 threads where both appeared, the same reviewer credited the tool and called the
  paper not-a-contribution in the same review.**
- "Our contribution is the instrument" was tried in 20 of 349 threads and conceded **once**, by
  a reviewer who could not lower a score on a paper already at 6/6/8/10 *with a theorem*.
- Corpus-wide, probes accept **below** benchmarks (33.7% vs 41.8%, p = 0.008) while drawing the
  novelty objection half as often — so the objection is not even the mechanism.

**Keep `src/alphabet_audit.py`** — it is cheap, it is true, and it makes the work reproducible
for criterion 2. **Stop treating it as the pitch.** This retires the "instrument" half of F15.

**2. The one framing that does work, and it is a knife-edge.** Reviewers spontaneously credit
generality when it **unlocks a measurement that was otherwise impossible** (`jv1a8MPymM`,
accepted). The mirror image is `eDxZ6MR2FL` — a cheap, label-free, model-agnostic probe —
**rejected** on *"Running ADF or Levene tests on LLM outputs is relatively simple experiment."*

So the audit's cheapness must be framed as **access**, never convenience. Not "this is easy to
run"; rather "no labels, no structures and no alignment are required, so this can be measured
where nothing else can." Identical fact, opposite reception.

**3. Our anticipated objection is the wrong one, and the real one is dangerous.** theory.md and
the outline both pre-empt "a necessary condition does not say the model is bad." Across **192
invariance-violation threads that objection appears twice.** Explicit necessary-vs-sufficient
attacks: four. Keep the sentence; cut the paragraph.

What actually kills invariance papers, by accept rate:

| Objection | Accept | n |
|---|---|---|
| **"vacuous / true by construction"** | **15.0%** | 20 |
| "what does the model do instead?" | 29.2% | 24 |
| "so what / no consequence" | 48.1% — *above* baseline | — |

**"Vacuous / true by construction" is our largest single exposure and nothing in the project
currently answers it.** A reviewer can say: *you defined a property that only a column-statistic
model has, then showed a neural model lacks it — the conclusion is in the premise.*

**The winning counter is an attainability existence proof** (`cMGJcHHI7d`), and we already have
one without having recognised it as such. Potts is a model that (i) **satisfies the condition
exactly** — max|ΔS| ≈ 1e-14, and now under a *pseudolikelihood* solver too (C13) — and
(ii) **is genuinely accurate**, 6.9× chance on ubiquitin. So the condition is not vacuous: it is
satisfiable by a real, useful model on the same input. The property is a discriminating one,
not a tautology, and the proof is a working system rather than an argument.

This must be stated **where the theorem is introduced**, not in a discussion section.

**4. Two calibration facts worth keeping.** 39% of all score raises — and **46% for diagnostic
papers** — involved **no new empirical work**; reframing alone was the second-largest winning
category (19%) after new experiments (41%). Diagnostic papers are ~1.3× more likely than method
papers to win on argument alone. And **not rebutting is fatal: 0 raises in 48 threads.** Also a
clean null worth honouring: no objection type is statistically distinguishable from the 9.6%
per-reviewer raise base rate. Do not over-model which objection to fear; fear the two above,
answer everything, and answer it in writing.

---

### F24 — A registered correctness gate caught a real bug that every test we had missed (2026-08-05)

**Method finding, not a result. Worth its own entry because it changed what I trust.**

C13's first cell failed its own pre-registered gate: pseudolikelihood P@L/2 = 0.2727 against
mean-field's 0.3333, and zero-init invariance came out at 1.29e-02 relative where the theory
predicts floating-point noise. Amendment 13 had committed the response in advance — *"a failure
here is an implementation bug until proven otherwise; debug before reporting"* — so the branch
was taken rather than argued about.

**The bug.** `plm.py` added its L2 penalty to the *summed* negative log-likelihood and then
divided the total by `n`. Effective coefficient: `lam/n` = 0.01/3000 ≈ 3.3e-6. The fit was
unregularised, with 29,568 free parameters per column against 3000 sequences.

**One cause produced both symptoms**, which is why it was diagnosable at all. Overfitting drove
precision below mean-field; ill-conditioning made L-BFGS halt at materially different points
for the two fits, so invariance looked loose. Neither symptom implicated Proposition 1, and
without the gate the 1.29e-02 would have been reported as *"pseudolikelihood Potts is only
approximately invariant"* — a false negative against our own theory, in the direction of
understating our result.

**After the fix, all three candidate λ clear the gate on ubiquitin** (mean-field 0.3333):
λ=0.001 → **0.5152**, λ=0.01 → 0.4848, λ=0.1 → 0.3636. Pseudolikelihood now beats mean-field
by the margin plmDCA is known for, which is the independent sign that the estimator is right.
λ=0.001 is selected by the registered rule (highest *unpermuted* precision; the permuted fit
was never run during selection). That choice is also the least-regularised and therefore
worst-conditioned option — the hardest case for invariance — so it cannot be accused of buying
the result with conditioning.

**The transferable lesson, and it is the second time this project has learned it.** The
synthetic test `plm_test` passed at 5.4e-06 invariance and recovered 4/4 planted couplings,
because L=12 with a strong planted signal is well-determined with or without a penalty. **A
correctness test on an easy instance cannot detect a regularisation bug.** The earlier
instance was the real-Pfam control that passed because our own gap filter had already stripped
the characters it existed to exercise. Both times the test was real, ran green, and certified
nothing. What caught this was a gate defined against an *independent estimator on real data*,
with a threshold and a response fixed before the run. Design controls to fail on the
instance you would find awkward, and register what a failure obliges you to do — otherwise the
failure gets explained away at the moment it is least convenient.

### F25 — C14 registered: closing the "it's just OOD" objection with a *measured* Δ (2026-08-05)

Registered as Amendment 14 before any measurement code existed; **no result yet**, logged here
so the prediction is on the record.

The strongest surviving alternative explanation for everything in H2 is that a randomly
relabeled sequence simply is not a protein to the model — `A→W` everywhere makes tryptophan
the commonest residue — so the collapse is unremarkable and the In-class-vs-random
dissociation just means In-class relabelings stay nearer the training distribution. C6
answers the **edit-distance** form of that objection; Hamming distance is not the model's own
notion of distance, so C6 does not answer this form. That is the gap.

C14 measures the **model-perceived** shift, `Δ_LL = LL(x) − LL(π(x))` in nats/residue, from
the forward pass the categorical Jacobian already runs on the unmutated sequence — a
log-softmax and a gather away, under the same 20-residue support the Jacobian uses. Cells are
read from the committed main-sweep CSV and permutations regenerated from the recorded seeds,
so every Δ_LL joins to an overlap measured before Δ_LL was conceived of.

**The DV is mediation, not correlation.** Marginally the two must be related — the arms differ
on both. The question is whether regime still predicts overlap at *matched* Δ_LL.
**Registered prediction: NOT mediated** (ΔR² ≥ 0.15, partial |ρ| ≥ 0.3), because In-class
relabeling preserves biochemical class and so preserves the local hydrophobicity pattern even
where the sequence is equally surprising.

**The branch I did not predict is the one that flatters the paper, which is exactly why it is
locked.** If it comes out **mediated**, contact-map competence is a function of the model's own
perceived distribution shift — Zhao et al.'s thesis instantiated in biology with a Δ that is
*measured rather than constructed*, the thing F14 recorded we could not deliver, and outline §4
gets reorganised around it. Either way the control that holds in every branch is that Potts on
the same relabeled alignment is exactly invariant, so a large Δ_LL never means the
coevolutionary computation is impossible on that input — only that this model finds the input
unfamiliar. No wording may let Δ_LL read as "the task got harder".

### F26 — C13: a pseudolikelihood Potts fit *is* invariant, at the level of its predictions (2026-08-05)

This is the result that closes the last retreat available to the "your baseline is invariant
only because it is closed-form" objection. Ubiquitin, L=67, depth 3000, λ=0.001, random
relabeling at realized m=20 `[results:c13_run.log]`:

| init | gate (plm vs mean-field 0.3333) | rel max\|ΔS\| | top-L/2 overlap | permuted P@L/2 |
|---|---|---|---|---|
| zero | **0.5152** PASS | 1.88e-03 | **1.000** | 0.5152 (identical) |
| random | **0.5152** PASS | 1.85e-03 | **0.970** | 0.4848 |

Amendment 13 registered two separate predictions. For **(b) random init** it predicted top-L/2
overlap ≥ 0.95; measured 0.970, **met**. For **(a) zero init** it predicted *both* an identical
top-L/2 set *and* rel max\|ΔS\| < 1e-8. The set is exactly identical; the score tolerance was
missed by five orders of magnitude. We did not relax the threshold — Amendment 13's branch 3
obliges a debug, and Amendment 13b registered the diagnostic before running it.

**The maxiter ladder settles it** `[results:c13b_maxiter_ladder.txt]`, ubiquitin, zero init,
λ=0.001:

| maxiter | rel max\|ΔS\| | top-L/2 overlap |
|---|---|---|
| 2 | **1.1e-15** | 1.000 |
| 10 | 1.4e-14 | 1.000 |
| 40 | 1.2e-09 | 1.000 |
| 150 | 1.9e-03 | 1.000 |

At two iterations the two fits agree to **machine precision**. Twelve orders of magnitude of
monotone growth follow, with the contact set exactly identical at every rung. Nothing about the
formulation is asymmetric; the divergence is grown by the optimiser. The mechanism is plain:
the relabelled design matrix has its columns in a different order, so every GEMM accumulates
differently and the gradients differ at ~1e-16 from the first step, and L-BFGS is not
contractive. λ=0.001 was chosen as the *least* regularised and so worst-conditioned candidate,
making this close to the worst case available rather than a flattering one.

**COMPLETE, both proteins, 4 fits, gate PASS on all four** `[csv:h2_c13_plm_invariance]`.
Protein G replicates ubiquitin: zero init gives P@L/2 = 0.2143 against mean-field 0.2143, top-L/2
overlap **1.000**, permuted precision identical, rel 1.33e-03. Aggregated: zero init max
rel\|ΔS\| = 1.88e-03 with min overlap **1.000**; random init max rel\|ΔS\| = 3.52e-03 with min
overlap **0.970**.

*One thing the run's own summary line overstates and the paper must not repeat.* The script
prints "zero-init is 2× tighter than random-init in rel\|ΔS\|". A factor of two between 1.9e-03
and 3.5e-03 is not a meaningful separation — Amendment 13 expected zero-init to be *orders*
tighter, and after 13b we know why it is not: at maxiter=150 both fits have effectively
converged and what remains is accumulated rounding either way. The real zero-vs-random
difference is in the **contact set** (exactly identical vs 97% identical), not in the score
norm. Report that, not the 2×.

**What C13 licenses, and what it does not.** It licenses: *a pseudolikelihood Potts fit driven
by a real numerical optimiser produces an identical contact set and identical precision under
alphabet relabeling.* It does **not** license claiming score-level invariance to machine
precision for that estimator — that remains a property of the closed-form inverse-covariance
estimator alone (Proposition 2, ≤ 2.3e-14). Both must be stated. The gap between them is the
honest content of C13: **the theorem is about the computation, and a numerical solver inherits
it only up to its own conditioning.** Note also that at maxiter=150 the initialisation barely
matters (1.88e-03 vs 1.85e-03) — both fits have effectively converged, so the zero-vs-random
contrast the protocol was designed around turns out to be less important than the
iteration-count contrast it did not anticipate.

### F27 — C12: the dose-response is WEAK, which qualifies §4 (2026-08-05, ubiquitin only)

Ubiquitin, 48 full derangements at realized m=20 (4 anchors excluded per Amendment 12a), cost
range −1.43 to 2.99, overlap range 0.000 to 0.605 `[results:c12_run.log]`:

- Spearman ρ(sequence-weighted BLOSUM62 cost, overlap) = **−0.328**
- ρ(unweighted cost, overlap) = −0.325 (agrees)

Amendment 12 predicted ρ ≤ −0.6 and registered |ρ| < 0.3 as refuting. **Neither fires.** The
result lands in the weak band: the prediction is not met, and the hypothesis is not refuted.

This matters for how §4 is written. The *categorical* In-class-vs-random contrast is large
(0.733 vs 0.010 at m=20 on the main sweep), but among full derangements at constant m=20,
biochemical cost explains only about a tenth of the variance in overlap. So the effect is much
better described as **categorical than as a graded dose-response in BLOSUM cost** — preserving
biochemical *class* is what buys retention, and finer gradations of cost within the
class-destroying regime buy comparatively little. Any sentence in the paper implying a smooth
dose-response is not supported. Protein G is still running and could move this.

### F28 — C14: perceived distribution shift explains about half, and saturates (2026-08-05)

204 cells, ESM-2 650M, six proteins `[csv:h2_c14_perceived_shift]`. Δ_LL = LL(x) − LL(π(x)) in
nats/residue; positive means the relabeled sequence is less protein-like *to this model*.

| regime | Δ_LL m=2 | m=6 | m=10 | m=20 |  | overlap m=2 | m=6 | m=10 | m=20 |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.187 | 0.288 | 0.275 | 0.278 | | 0.711 | 0.342 | 0.079 | 0.010 |
| In-class | 0.027 | 0.083 | 0.159 | 0.226 | | 0.913 | 0.863 | 0.732 | 0.733 |
| Cross-class | 0.211 | 0.277 | 0.253 | 0.273 | | 0.544 | 0.226 | 0.006 | 0.022 |

**The registered mediation test, on the 144 random+In-class cells:**

- marginal Spearman(overlap, Δ_LL) = **−0.700**
- R² overlap ~ Δ_LL = 0.491 → adding regime = 0.621, so **ΔR² = 0.131**
- partial Spearman(overlap, regime | Δ_LL) = **+0.506**

**Verdict: BRANCH C — AMBIGUOUS, and we claim neither branch, as registered.** The partial
correlation (0.506) clears the not-mediated threshold of 0.30, but ΔR² (0.131) falls short of
0.15. The registered response to exactly this case is to report the numbers and claim nothing,
so that is what the paper does. The substance is *partial mediation*: perceived shift explains
about half the rank variance and regime adds a substantial independent effect on top.

**EXPLORATORY (not the registered DV; discovered while reading the output).** The most
informative pattern is one the mediation statistic cannot see: **Δ_LL saturates by m=6 while
overlap keeps collapsing.** In the random arm Δ_LL goes 0.187 → 0.288 → 0.275 → 0.278 — flat
after m=6 — while overlap over the same range goes 0.342 → 0.079 → 0.010, a 34× fall. The
model's sense of how unfamiliar the input is stops changing, and its contact map degrades all
the way to chance anyway. This is a *within-arm* comparison, so no regime confound can produce
it.

The same point in a single pair: **In-class m=20 (Δ_LL 0.226, overlap 0.733) versus random
m=20 (Δ_LL 0.278, overlap 0.010)** — the model finds these two inputs almost equally
unfamiliar, and retains 73× more of its contact set on one than the other.

Both observations must be labelled exploratory in the paper. They are descriptive readings of a
table, they were not registered, and the registered test came out ambiguous.

**The control that holds regardless**, and it is what stops Δ_LL being read as difficulty:
Potts on the same relabeled input is exactly invariant. A large Δ_LL never means the
coevolutionary computation is impossible on that input — only that *this model* finds the input
unfamiliar. Never write Δ_LL as "the task got harder".

*Within-(regime, m) seed-level correlations, which vary only the permutation seed:* random
−0.426 / +0.297 / +0.219 / +0.116; In-class +0.060 / −0.118 / −0.560 / −0.847 at
m=2/6/10/20. Mixed in the random arm, strongly negative in the In-class arm at high m.

### F29 — C15: the Δ_LL saturation replicates at every scale, out-of-sample (2026-08-05)

C14's saturation observation was exploratory — one model, one protein set, spotted after the
fact — and it was carrying more argumentative weight in §4 than an unregistered observation
should. C15 tested it against the committed **C8 scale** cells (4 models × 4 proteins × 2
regimes × 4 m × 2 seeds), a protein set sharing **only PF00018** with C14's
`[csv:h2_c15_shift_saturation]`.

**Random arm (registered):**

| model | Δ_LL m=2 | m=6 | m=10 | m=20 | | overlap m=2 | m=6 | m=10 | m=20 |
|---|---|---|---|---|---|---|---|---|---|
| 8M | 0.067 | 0.158 | 0.150 | 0.205 | | 0.375 | 0.172 | 0.143 | 0.042 |
| 35M | 0.173 | 0.286 | 0.259 | 0.320 | | 0.538 | 0.118 | 0.075 | 0.027 |
| 150M | 0.184 | 0.346 | 0.303 | 0.358 | | 0.712 | 0.276 | 0.160 | 0.028 |
| 650M | 0.208 | 0.440 | 0.365 | 0.384 | | 0.743 | 0.272 | 0.165 | 0.017 |

**Registered outcome, stated exactly.** P1 (Δ_LL rise from m=6 to m=20 ≤ 0.05) **passes at all
four scales**: +0.047, +0.034, +0.012, **−0.056**. P2 (overlap fall ≥ 0.10) passes at three;
it fails at 35M with 0.0915. The registered verdict is therefore **PARTIAL — 3 of 4 scales
replicate the conjunction — and that is what the paper says.**

The 35M failure is worth reading correctly rather than explaining away: that model had already
fallen to 0.118 overlap by m=6, so there was less than 0.10 of absolute room left to fall. In
relative terms it still dropped 4.4× (0.118 → 0.027). The threshold was absolute and it failed
on that. We do not restate it in relative terms to make it pass.

**The claim that survives, and it is the one that mattered:** the *saturation itself* — the
thing C14 could only assert exploratorily — replicates at **4/4 scales across an 80× parameter
range on a nearly disjoint protein set**. At 650M the rise is actually negative: perceived shift
*falls* from 0.440 to 0.384 between m=6 and m=20 while overlap falls 0.272 → 0.017.

**A double dissociation, from the descriptive In-class arm.** The two arms move Δ_LL and
overlap in *opposite* relations at every scale. Random: Δ_LL flat, overlap collapses to chance.
In-class: Δ_LL climbs ~3× (650M: 0.091 → 0.297) while overlap stays high (0.829 → 0.501).
So a rising perceived shift is compatible with a preserved contact map, and a flat perceived
shift is compatible with a destroyed one. **Whatever governs the contact map, it is not how
unfamiliar the model finds the input.** This arm is descriptive only — the In-class
builder's realized m saturates at 15/20, so its own Δ_LL plateau would not have been evidence —
but the *rise* is real and the dissociation does not depend on the plateau.

---

## Open Questions

- ~~Are the AMPLIFY per-year checkpoints actually public?~~ **RESOLVED (F19):** yes, and all
  yearly checkpoints 2011–2024 are on disk at `data/amplify` (8.4 GB) and verified loadable.
  H1's blocker is gone; the immediate use is C11 (third architecture), not H1.
- Where does multi-mutant (k≥2) data live in ProteinGym? H3 depends on it.
- ~~Does ESM-2's degradation under permutation differ between `In-class` and `Cross-class`
  at matched m?~~ **RESOLVED (F30, C6):** they differ sharply, and in a way that is *not* the
  "keyed on token identity alone" story. Against distance-matched substitution controls,
  In-class relabeling buys +0.44 to +0.49 of overlap at m ≥ 10; random buys ≤ +0.08 and
  Cross-class buys nothing (≤ 0). So ESM-2 *is* graded by biochemistry — it retains structure
  exactly to the extent the relabeling keeps residues biochemically recognisable, which is
  precisely the dependence a column-statistic estimator provably cannot have.
- ESM-3 weights are gated; is H5 (decoding-order dependence of its self-described
  "chain of thought") reachable before the deadline?

---

### F30 — C6 COMPLETE: at matched edit distance, only a *In-class* relabeling buys anything (2026-08-05)

**Status: CONFIRMATORY (registered as protocol Amendment 4, committed before the run).**
`[csv:h2_c6_matched_hamming]` — 306 rows = 3 proteins (ubiquitin, cytochrome c, protein G B1)
× 3 regimes × m ∈ {2,6,10,20} × 3 seeds × 3 arms = 102 matched cells. The within-cell Hamming
invariant was verified post hoc: **102 of 102 cells have identical Hamming distance across the
permuted / uniform / composition-matched arms.** The comparison is matched by construction.

Mean overlap, and the gap to the better of the two matched controls:

| regime | m | d | permuted | uniform | composition | gap |
|---|---|---|---|---|---|---|
| In-class | 2 | 8.3 | 0.804 | 0.750 | 0.704 | +0.054 |
| In-class | 6 | 25.4 | 0.698 | 0.493 | 0.446 | +0.205 |
| In-class | 10 | 46.2 | 0.573 | 0.087 | 0.025 | **+0.486** |
| In-class | 20 | 62.0 | 0.482 | 0.044 | 0.029 | **+0.438** |
| random | 2 | 10.0 | 0.689 | 0.709 | 0.681 | −0.021 |
| random | 6 | 23.8 | 0.410 | 0.349 | 0.365 | +0.046 |
| random | 10 | 43.6 | 0.156 | 0.076 | 0.023 | +0.079 |
| random | 20 | 78.7 | 0.019 | 0.035 | 0.030 | −0.016 |
| Cross-class | 2 | 8.8 | 0.631 | 0.730 | 0.679 | −0.099 |
| Cross-class | 6 | 26.6 | 0.274 | 0.319 | 0.443 | −0.169 |
| Cross-class | 10 | 43.9 | 0.061 | 0.045 | 0.083 | −0.022 |
| Cross-class | 20 | 66.3 | 0.042 | 0.045 | 0.024 | −0.003 |

**What was registered and held.** Amendment 4 predicted the permuted arm would beat both
matched controls, i.e. that edit distance is not the controlling variable. It holds, decisively,
in the In-class regime: at d ≈ 46 a relabeling retains 0.573 while a substitution of
*identical magnitude* retains 0.087 and a composition-matched one 0.025. The composition arm
kills the "it just preserved the composition" reading.

**What was NOT anticipated and is the sharper finding.** The margin exists *only* under
In-class relabeling. Under random or Cross-class relabeling the gap is ≤ 0 at every m ≥ 10 —
ESM-2's permuted contact map is statistically indistinguishable from mutating the same number
of positions at random. It has retained nothing beyond what the raw edit distance explains.

**Both readings are failures, because of the reference point.** A Potts model scores exactly
1.000 in every one of the 102 permuted cells, independent of regime, m and protein (Prop. 2,
verified ≤ 2.3e-14). Best case, ESM-2 discards half of a contact set that provably cannot
change; random/Cross-class case, it discards all of it.

**Where it does not hold.** Protein G B1 (L = 56) at In-class m = 20: permuted 0.071 vs
uniform 0.083, no gap, both within ~3× chance. The aggregate is carried by the two larger
proteins (ubiquitin 0.675 vs 0.035; cytochrome c 0.699 vs 0.013). The effect needs enough
long-range contacts to be visible; it is not a universal law.

**This supersedes the §4.1 anecdote.** The draft previously rested on a single cytochrome c
cell. It now carries the full 102-cell aggregate, which is both stronger and shows the regime
dependence the anecdote hid. See [[c6-within-cell-design]].

---

### F31 — C16: the anchoring is in the MODEL, not in our instrument (2026-08-05)

**Status: CONFIRMATORY. All three registered predictions pass (Amendment 16, locked before the
script was written).** `[csv:h2_c16_native_head]` `[results:c16_run.log]`

Every contact number in this project up to now came from the *categorical Jacobian* — an
instrument we built. The obvious reviewer objection is that alphabet anchoring might be a
property of that readout rather than of ESM-2. C16 reruns the identical cells through ESM-2's
**own shipped contact predictor**: the supervised attention-map logistic regression of Rao et
al. (2020), `predict_contacts()`, loaded from the `-contact-regression.pt` sibling checkpoints.
It is a different function of the same network — attention maps, not logit differences under
single-site mutation; supervised, not unsupervised.

Mean native-head overlap, ESM-2 650M, 9 proteins × 3 seeds (identity = 1.000 exactly, enforced
as a runtime gate; mean chance floor 0.018):

| regime | m=2 | m=6 | m=10 | m=20 |
|---|---|---|---|---|
| random | 0.842 | 0.554 | 0.229 | **0.028** |
| In-class | 0.936 | 0.882 | 0.801 | **0.793** |
| Cross-class | 0.752 | 0.433 | 0.069 | **0.009** |

| test | value | threshold | |
|---|---|---|---|
| P1 collapse | random m=20 = 0.028 | ≤ 0.15 | **PASS** |
| P2 biochemical grading | 0.793 − 0.028 = **0.765** | ≥ 0.20 | **PASS** |
| P3 instrument agreement | Spearman **+0.884** over 204 matched cells | ≥ +0.60 | **PASS** |

**What this licenses.** The pre-committed branch: *the anchoring is a property of the model,
not of our readout, and it reproduces in ESM-2's own published contact predictor.* The
categorical Jacobian stays the primary instrument because it is unsupervised; the native head
is the robustness check.

**The grading is *stronger* in the native head, not weaker.** The In-class-minus-random gap
is 0.765 here against roughly 0.48 in the Jacobian's matched-Hamming cells (F30). The predictor
that downstream users actually call is the more alphabet-anchored of the two.

**Robust to the DV knob.** Amendment 16 parenthesised |i−j| ≥ 24 while the sweep it calls
identical uses 12 (Amendment 3). Both were computed. At min_sep = 24: P1 = 0.025, P2 = 0.749 —
same verdict. The primary DV stays at 12 because P3 joins against Jacobian overlaps measured
there, and a mismatch would confound the agreement test.

**Two honest limits, recorded here.**
1. The P3 join dropped 120 of 324 cells and said so: 108 because ubiquitin, cytochrome c and
   T4 lysozyme are not yet in the Jacobian CSV (the main sweep is still writing them), and 12
   because those `Cross-class m=20` seeds were never run on the Jacobian side. **Re-run
   `--analyse-only` once the main sweep finishes** to recover the full join; ρ may move.
2. C16 cannot compare the two instruments' *absolute* overlap levels as a measure of quality —
   different predictors, different top-L/2 sets, one supervised and one not. Only the
   within-instrument pattern and the between-instrument rank correlation are interpreted.

**Registered in advance, so it cannot be invented now: the supervision objection.** The head is
trained on solved structures written in the standard alphabet, so a reviewer may say it has no
reason to be equivariant and the test is unfair. The answer fixed in Amendment 16: the
necessary condition applies to any function *claimed to predict contacts from sequence*, and
this head makes the stronger claim because it is the shipped predictor. We do **not** argue the
head "ought" to be invariant by construction. See [[h2-two-confounds]].

---

### F32 — C12 COMPLETE: there is no dose–response. The effect is categorical. (2026-08-05)

**Status: CONFIRMATORY, prediction NOT SUPPORTED.** `[csv:h2_c12_dose_response]`
`[results:c12_run.log]` Amendment 12 predicted ρ ≤ −0.6 on both proteins — damage rising
monotonically with the BLOSUM62-weighted biochemical cost of a full derangement.

| protein | n derangements | cost range | **overlap range** | chance | ρ weighted | ρ unweighted |
|---|---|---|---|---|---|---|
| ubiquitin | 48 | −1.43 … 2.99 | 0.000 … **0.605** | 0.012 | **−0.328** | −0.325 |
| protein G B1 | 53 | −1.41 … 2.93 | 0.000 … **0.143** | 0.028 | **+0.079** | +0.100 |

The BLOSUM sign assert passed (`sev(K→R) = −2 < sev(K→W) = +3`), so the positive sign on
protein G is not the broken-pipeline case Amendment 12 warned about.

**Neither registered branch is cleanly met, and we say so rather than picking the nearest one.**
Ubiquitin's −0.328 sits in the gap between the −0.6 confirmation and the |ρ| < 0.3 refutation.
Protein G's +0.079 is inside the refutation band. No protein reaches −0.6.

**Protein G could not have detected the effect.** Its entire DV range is 0.000–0.143 against a
chance floor of 0.028: essentially every full derangement destroyed the contact map, leaving
almost no variance to correlate cost against. This is range restriction — the same failure mode
that killed H4's registered design, and the reason `diag_range_restriction.py` exists in that
experiment. We flag it **as a property of the measured data**, not as grounds for discarding the
cell: the protocol pre-committed no such escape hatch, so protein G is reported as registered.
It simply means ubiquitin is the only cell with enough DV variance to test the hypothesis, and
ubiquitin lands weak.

**The conclusion is the same on every reading, which is why we can state it plainly: the paper
must not claim a dose–response.**

**What this does NOT contradict, and the resolution.** C6 and C16 show an enormous regime
effect: In-class relabeling beats random by 0.44–0.49 (Jacobian, matched Hamming) and 0.765
(native head). C12 says something different and compatible. It varies biochemical cost *within*
the set of full derangements — all of which move every residue type — and asks whether a
costlier one damages more. It does not. So:

> **Whether the relabeling preserves biochemical groups matters enormously; how far it moves
> residues along a scalar cost metric does not predict damage at all.**

The effect is categorical in the *structure* of the relabeling, not graded in its magnitude.
That is a sharper claim than a dose–response would have been, and it constrains the mechanism:
ESM-2 is not running a smooth biochemical-similarity function over residues. Amendment 12's
refutation branch anticipated exactly this — "something coarser … whether specific tokens move
at all" — and required §4 be **rewritten, not softened**. It has been.

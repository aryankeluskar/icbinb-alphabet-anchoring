# Research Log

Decision timeline. Newest entries at the bottom.

---

## 2026-08-05 — Bootstrap

**Question as posed by the user:** apply the "Chain-of-Thought Reasoning is a Mirage"
(Zhao et al., DataAlchemy) framing to ESM / bio-AI models, for ICBINB-BIO @ NeurIPS 2026.

**First decision — reject the naive framing.** ESM emits no reasoning trace, so
"CoT is a mirage ⇒ ESM is a mirage" is a category error and a reviewer will say so in the
first paragraph of their review. Recorded as `framing_risk` in research-state.yaml.
Three things *do* port: the dialable Δ knob, the trace/answer decomposition, and
train-from-scratch leakage control.

**Second decision — find the asymmetry that favours biology.** Zhao et al.'s Limitation (ii)
is that Δ cannot be measured for real LLMs because training corpora are opaque; their
workaround is to *construct* a known-unseen distribution rather than *estimate* an unknown
one. Protein corpora (UniRef) are public, versioned and timestamped, and sequence identity
is a real distance metric. So Δ becomes measurable. This is the novelty hook and we can
cite them conceding the limitation in their own words, which is far stronger than asserting
it ourselves. (Verbatim quotes confirmed from arXiv:2508.01191v6 §8.2 and Limitations (ii).)

**Five hypotheses registered** in research-state.yaml, ranked. H1 (measured Δ) is the
headline but is blocked on whether AMPLIFY's per-year checkpoints are public. H2 (alphabet
permutation) needs no training and no downloads beyond a pLM, so it goes first.

## 2026-08-05 — H2 protocol locked

Locked `experiments/H2-alphabet-permutation/protocol.md` and committed it (`d296f37`)
before running anything. The design turns Zhang & Ovchinnikov's *descriptive* claim
(pLM contact prediction is driven by local motif lookups) into a *falsifiable* prediction:
a consistent bijective relabeling of the amino-acid alphabet leaves the MRF exactly
invariant, so any ESM-2 degradation isolates reliance on literal token identity.

Chose to verify the mathematical premise on synthetic data *before* spending GPU time.
Rationale: if the MRF is not exactly invariant in our implementation the pipeline is buggy,
and there is no statistical route to non-invariance — so this is a hard correctness assert,
not an experiment.

## 2026-08-05 — H2 controls pass; one design error caught

All controls pass (see findings.md F1). Two corrections were forced along the way:

1. **precision@L/k is ceilinged** at n_true/n_pred. The first run reported 0.20 and I read
   it as weak; it was in fact the ceiling, i.e. perfect recovery. Added `recall_at_frac`
   and `precision_ceiling` and now report recall as the primary control metric.
2. **The original C3 was not a null.** A per-sequence permutation relabels both columns of
   a row identically, and mutual information is invariant under a within-row bijection, so
   it only blurs coevolution rather than destroying it (recall 1.000 → 0.600, versus 0.000
   for a true null). Replaced C3 with a per-column row shuffle and promoted the per-sequence
   case to C3b, a third experimental condition. Catching this mattered: had it stood, the
   pLM comparison would have been calibrated against a condition mislabelled "destroyed".

**Infrastructure note:** the control script must be run with `python3 -u` when redirected
to a file. Block buffering plus a heavily loaded box made a working script look hung.

## 2026-08-05 — Parallel recon in flight

Twenty background agents dispatched across: environment build (fair-esm on the A100 —
note the PyPI package `esm` is an unrelated project), literature with a novelty ledger,
data/checkpoint recon, and targeted novelty checks on each hypothesis. Two results already
back and both derisk H2:

- Zhang & Ovchinnikov contains **no permutation, shuffling or relabeling control anywhere**
  in the paper or SI. H2 is unoccupied ground.
- Their own text states the categorical Jacobian applied to an MRF/multivariate-Gaussian
  *exactly returns* its coupling tensor W. So the invariance we measured at 3.6e-15 is a
  theorem derivable from their equations, not an empirical accident. We can assert it
  analytically in the paper and use the numeric run as verification.

## 2026-08-05 — H4 run and abandoned; spine narrows to H2

Ran H4 (edit distance vs biochemical severity) on ProteinGym's precomputed zero-shot
scores — CPU-only, so it did not contend with the two GPU jobs. Locked the protocol first.

**Both designs failed the same registered prediction.** The primary per-variant analysis was
a flat null (all CIs include zero, z = 1.10). Diagnosis: 58 of 69 multi-mutant assays have
k_max = 2, so `k` had almost no variance, and per-variant rank error is dominated by DMS
measurement noise. Registered Amendment 1 (within-cell Spearman) *before* running it, and
recorded that the primary null could not be overwritten by whatever it returned.

Amendment 1 fixed the power problem — effect sizes went from ≈0.005 to ≈0.3 — and then
refuted H4 outright: `D > 0` for pLMs *and* alignment models alike (+0.270 vs +0.242,
z = 1.85 n.s.), when the whole hypothesis was that the two families would differ in sign.
GEMME, DeepSequence and EVE all came out on the pLM side of the prediction.

Per the falsification clause written into Amendment 1, **H4 is abandoned rather than re-cut a
third time**. Two differently-powered designs failing the same prediction is evidence about
the hypothesis, not about the method.

The most likely explanation for a universal `D > 0` is a property of the data rather than of
any model: higher-k variants are more often dead, which compresses within-cell DMS dynamic
range and mechanically depresses within-cell Spearman identically for every scorer.

**Methodological lesson, now in findings.md:** a collinearity control is not a variance
control. C1 confirmed k and mean severity were near-orthogonal but never checked that `k`
*had* variance — which `data/MULTI_MUTANT_INVENTORY.md` had already told me.

**Consequence for the paper.** The novelty ledger had H4 as the second-most-open angle and
the intended second experiment. It is gone. The spine is now H2 alone, which raises the bar
on finishing C6 and C7 properly rather than adding breadth.

**C6 partially rehabilitates H2.** Its first cell suggested the collapse was mere
edit-distance sensitivity; with more cells the reading reverses at m ≥ 6, where a bijective
relabeling costs ESM-2 *less* than the same number of arbitrary edits at the same positions.
That is the "partial competence" branch of the registered interpretation table.

## 2026-08-05 — Range restriction confirmed; spine reframed; depth over breadth

**The H4 artifact is measured, not conjectured.** corr(k, within-k DMS std) over all 69
multi-mutant assays: median −1.000, mean −0.521, negative in 77%. DMS dynamic range shrinks
as mutation depth grows. That reproduces the universal `D > 0` with no model-specific story,
and it means the aggregated design was measuring ProteinGym, not protein language models.

**Consequence I did not want but have to accept: H3 does not get promoted.** The obvious move
after H4 died was to advance the next-ranked hypothesis. H3 now carries *two* measured
confounds — this range restriction, and ProteinGym's additive-by-construction scoring
(findings.md F5) — stacked on the same DV. Promoting it would buy a third shallow experiment.
H3 stays at priority 4.

**Structural problem in the framing, found by reading our own spine against our own hook.**
The novelty pitch is "unlike frontier LLMs, Δ is *measurable* for pLMs." H2 does not measure
Δ; it *constructs* a shift, which is exactly Zhao et al.'s own move. Hook and experiment are
disconnected. Resolution chosen: reframe. H2 is sold on its own terms — pLM contact
prediction is not an alphabet-abstract computation, and a 20-line inverse-covariance baseline
is exactly invariant to a symmetry that large neural models violate. Δ-measurability becomes
a Related Work paragraph, not a promise. Instantiating H1 with the AMPLIFY yearly checkpoints
would keep the promise but costs most of the 24 remaining days on ground that is already
largely occupied.

**Direction: DEEPEN, not broaden.** Registered gaps to close, in order — (1) scale sweep
8M→3B, which is the first objection any reviewer raises and is squarely on ICBINB's genre;
(2) true contact precision against PDB for the unpermuted condition, so the self-consistency
proxy rests on maps that were good to begin with; (3) finish C6/C7; (4) figure, then draft.

**C7 first family (preliminary, 1 of 4).** MSA Transformer degrades 0.850 → 0.317 across
m on PF00018 while the in-run Potts assert holds at exactly 1.000 on the same alignment.
That is the strong branch of Amendment 5 — but it is one family and MSA-T does not reach
chance, so the claim to prepare for is "severe degradation," not "collapse."

## 2026-08-05 — C7 lands the strong branch; both remaining gaps closed in-flight

**C7 is the result the paper turns on.** Amendment 5 called the Potts-refits asymmetry "the
most serious threat to H2 and it is of our own making, not a reviewer's invention." MSA
Transformer removes it — alignment-conditioned at inference exactly like Potts, retrained on
nothing — and H2 survives: MSA-T falls 0.775 → 0.147 under random relabeling across 4
families while Potts on those same permuted alignments is exactly 1.000000 in all 136 cells.
So the claim does not narrow to single-sequence models. Two qualifications go in the paper
rather than being smoothed over: MSA-T does not reach chance the way ESM-2 650M does, so the
verb is *severe degradation*; and the In-class-vs-random dissociation replicates, so the
biochemistry reading is not a single-sequence artifact.

**Two gaps closed rather than deferred.** Ground truth landed (`src/structures.py`, all four
targets at 100% coverage), and with it the number that makes the collapse meaningful: ESM-2
650M gets **P@L/2 = 0.763 on ubiquitin, 14× chance**, before permutation. The degradation is
from competence. C9 then restores the locked protocol's *original* DV — precision, not
self-consistency — on the two targets that are 100% identical to their deposited chain.
Separately, a second readout was registered and launched: variant effect on 15 ProteinGym
assays, single mutants only, which sidesteps both known ProteinGym confounds at once.

**Two self-inflicted errors caught this session, both by auditing rather than by results.**
The In-class regime cannot relabel more than 15 of 20 residues, because A, C, G, H and P
have no biochemically similar partner — so the "m=20 In-class" point was really m=15 and
the flat 10→20 segment was the builder saturating, not the model coping (Amendment 7, F10).
And the structure module's first Needleman–Wunsch produced mappings that were 100% identical
at every inspected position while anchoring residues onto the wrong copies of a repeated
letter. That is now twice on this project that a check passed while being wrong. The pattern
worth remembering: **agreement on the values you inspect does not certify the indices you
did not.**

**Rao et al. verified from the PMLR PDF.** The 52.9/15.9/27.9 triple is real and quotable
with its population caveat; no alphabet relabeling appears anywhere in the paper or any
appendix; and one of my own glosses was wrong — their position shuffle is a single *global*
column reordering, not a per-sequence one. The find that changes the framing is Appendix A.4,
which concludes the model "stops focusing on the amino acid identities" in later layers. That
is an observational claim predicting the opposite of what we measure. The paper should lead
with it as the tension our intervention resolves.

## 2026-08-05 — The story reorganises around one number, and the theorem gets proved

**Direction change, prompted by a shift in target venue.** The question became not "will this
clear an ICBINB workshop" but "what is the most compelling version of this story", with ICLR
2026 accepted papers as the calibration set. Three agents are out: two ingesting ICLR 2026
exemplars (a protein-LM slice and a diagnostic-paper slice), one mining public OpenReview
reviews for *both* accepted and rejected papers in the diagnostic genre — the rejections
being the more informative half, since they name what actually kills this kind of paper.

**The claim, restated.** Coevolutionary contact prediction is a function of alignment column
statistics, and any such function is provably invariant to renaming the alphabet. pLMs are
not. The value of this framing is that it is a **necessary-condition** argument: we never
have to say what the model represents, only that it cannot be doing the thing the field
assumes. That is what separates it from a probing paper.

**So the theorem got proved rather than asserted** (`theory.md`). Proposition 2 is the one
that matters, because it covers the estimator we actually run: relabeling acts by orthogonal
conjugation, and every step of the scoring pipeline commutes with it. Testing the *conditions*
the proof names then paid for itself twice — the isotropic-ridge requirement is real (a
1.0–1.5× per-residue ridge breaks exactness by 11.8%), and the second condition caught a bug
in my own control, which had been swapping two characters the encoder folds into the same
slot and reporting invariance as though that proved something.

**C9 then produced the number the paper should open with.** True long-range precision against
1UBQ: **0.763 → 0.053, i.e. 14.4× chance → exactly chance**, with the identity gate passing to
three decimals. Self-consistency is thereby demoted from headline to control, which is what it
was always for — it answers "the permuted input is just harder", an objection precision alone
cannot address. Keeping both DVs and saying which answers which is stronger than either.

**What to cut, decided.** H4 leaves entirely — not an appendix. C6/C7/C8/C9 stop being four
experiments and become four objections to one experiment. Roughly nine iterations collapse
into three figures.

**What is still missing, and it is not more measurement.** A consequence. With precision at
the base rate, "the model works anyway" is harder to say about this regime, but "nobody
relabels alphabets in nature" is the objection the intervention is genuinely exposed to. The
candidate answer is to turn the diagnostic into a per-protein predictor — does alphabet
sensitivity predict where ESM-2 underperforms a Potts model that has a good MSA? That uses
machinery we already have. Whether it is optional or load-bearing depends on the review
evidence now being gathered.

---

## 2026-08-05 — The venue answers the question I was going to guess at

I had left one decision explicitly open: whether a well-controlled negative needs a
constructive half to clear a top venue. My working answer was yes, and the cheapest candidate
was turning the symmetry violation into a per-protein failure predictor. That would have cost
a week and exposed the paper to the one objection it cannot answer — nobody relabels alphabets
in nature.

The evidence says no. Not from intuition about reviewers, from the four analysis-driven pLM
papers actually accepted to ICLR 2026 main conference. The relevant one is thinner than what
we already have on disk: two experiments, two results figures in the whole main body, zero
tables, zero error bars, no limitations section, and a single actionable recommendation that
was tested, failed, and got disclosed in a subordinate clause. It was accepted because it
shipped an *instrument* — a way to ask a question that could not previously be posed.

So the failure predictor is dropped, not deferred, and the reframe is: the alphabet-
equivariance test is the contribution; the ESM-2 collapse is its first application. That is a
better fit for what we actually built anyway. We have a proof, a guard, a positive control and
a negative control for the test itself — that is an instrument, not an experiment.

The other thing worth writing down is that our baseline position is inverted relative to every
paper in that sample. In each of them the simple baseline is missing, or wins and is absorbed
by an adverb, or wins and is neutralised by inventing a new axis to score on. Ours satisfies
the property exactly, by theorem. Reverse Distillation had to prove a theorem to buy that
asset; we get it for free.

Registered two protocols before results this cycle. Amendment 9 (readout 2) because three
identity rows landed and two of them had ESM-2 Spearman near zero — the benchmark's own
published value, not a bug — which makes an absolute mean Δρ close to meaningless. That is
structurally the same defect that killed H4, so the aggregation rule got fixed before the
remaining twelve assays land rather than after. Amendment 10 (C10) because C9 says ESM-2 falls
to chance and cannot say whether chance is bad relative to anything a practitioner would use.
Potts has one number that holds under every relabeling; if it sits above ESM-2's permuted
number the claim upgrades from "degrades to chance" to "falls below the classical model it
replaced". The Potts arm has landed at 0.3158, 6.9× chance, bit-identical across relabelings
to 2e-14. The withdrawal branch is off the table; the crossing is pending the ESM-2 arm.

---

## 2026-08-05 (later still) — The crossing replicates, and I record why the sample was one

C10 now has two targets. Ubiquitin: ESM-2 goes from 2.33× Potts to 0.17× Potts. Protein G B1:
2.80× to 0.60×. Both cross. Potts does not move on either — max|ΔS| of 1e-14, asserted in-run.

The protein G version is weaker and I am not going to let it be quoted as if it matched.
Its Potts sits at 2.2× chance against ubiquitin's 6.9×, on an alignment five times shallower,
and its permuted ESM-2 lands at 0.60× Potts rather than 0.17×. The honest generalisation is
that the crossing holds on both targets with a margin that tracks how good the classical
baseline is — which is what you would expect if the effect is real.

Worth writing down how the second target happened. Amendment 10 restricted C10 to ubiquitin
and gave a reason: only one target had both a deep MSA and a 100%-identity PDB chain. That
reason was false. Protein G B1 is 100% identical to 1PGB and ProteinGym ships an alignment for
it at depth 3,109. I had not looked. The sample size of the headline result was one because of
an unchecked assertion in my own protocol, and the fix took under an hour. Amendment 10a says
so in the protocol rather than quietly adding a target.

Next real gap is the generalisation ask — a third architecture, neither ESM-2 nor MSA
Transformer. The AMPLIFY 120M yearly checkpoints are already on disk, 8.4 GB, and would give a
different tokenizer, different positional scheme and different training corpus. A CPU load
probe timed out at seven minutes with no output at all, so loadability is unverified and that
is the thing to debug before any protocol gets written for it.

---

## 2026-08-05 (evening) — Readout 2 shows the same shape, and AMPLIFY needed three fixes

Wrote the Amendment 9 analysis and ran it on what has landed. Three assays in, the
variant-effect readout reproduces the structure the contact readout has: In-class −0.122
at m=20 against random −0.333, with Cross-class tracking random rather than In-class. That
last detail matters more than it looks — Cross-class swaps also violate biochemistry, so a story
that said "any big relabeling hurts" would predict In-class and Cross-class together, and a
story that said "biochemistry is what is encoded" predicts Cross-class with random. It is the
second one.

PSSM Δρ is exactly zero at every permuted cell, which is Proposition 1 with J = 0, so it is a
pipeline assert rather than a finding.

The Amendment 9 risk is biting: only one of the first three assays clears the ρ ≥ 0.20 gate,
because two of them have ESM-2 identity Spearman near zero — the values ProteinGym itself
publishes. The script suppresses the gated DVs, as registered. The fourth assay came in at
0.737, so two of four; if that rate holds we get seven or eight and the gated analysis
survives.

The other half of the evening went to AMPLIFY, which is the third architecture for the
generalisation ask. It took three fixes rather than one. `trust_remote_code` hangs
indefinitely — twice, seven minutes each, no output and no traceback, with the hub forced
offline. The checkpoint ships everything needed, so the fix is to skip transformers' Auto*
machinery entirely; but then `amplify.py` hard-imports xformers, which has no wheel for
torch 2.11+cu128, and uses relative imports that require it to be loaded as a package
submodule. The xformers shim implements exactly the two symbols used, and the attention one is
written to be the same computation as amplify.py's own CPU branch rather than an approximation
of it — the CUDA branch is the only caller. The wrapper subclasses ESM2Jacobian and overrides
only tokenization and the forward, so the contact-map math is the identical code that produced
every ESM-2 number. A cross-architecture claim is worth nothing if the readout differs.

Loader is committed but not yet verified end to end. The box is at load average 42 with seven
GPU jobs, so the probe is crawling rather than failing.

## 2026-08-05 — AMPLIFY verified; C11 registered and launched

The AMPLIFY loader is verified end to end (findings F19): 118.3M params, correct 27-token
vocab mapping, wildtype recovery 1.000 at unmasked positions. Three sequential blockers,
all recorded in F19 — transformers' `Auto*` machinery hangs indefinitely against the local
checkpoints, `amplify.py` hard-imports an unavailable `xformers`, and its relative imports
need a synthetic package. The previous commit message (`f8f023e`) said "not yet verified end
to end; probe running"; that is now superseded.

Registered Amendment 11 (C11) **before** launching, per protocol. C11 is the generalisation
arm: ESM-2 and MSA Transformer are both Meta models on UniRef with the same vocabulary, and
a reviewer can fairly read that as one lineage rather than as evidence about pLMs. AMPLIFY is
a genuinely different lineage and is matched to ESM-2 150M on capacity (118M vs 148M), so
architecture and corpus vary while size roughly does not.

Two things the amendment fixes in advance. A **validity gate** — AMPLIFY's unpermuted P@L/2
must clear 3x chance on at least one of the two 100%-identity targets before any permuted cell
is reported. This is the lesson from readout 2 applied prospectively: a degradation statistic
on a unit with no signal is not a weak result, it is not a result. The gate costs nothing,
because it reuses the baseline Jacobian the sweep needs anyway. And the **refuting branch** —
if AMPLIFY turns out invariant, prediction 1 is refuted, that is the most interesting outcome
available, and the instruction on record is to make it the headline rather than bury it.

Box state at launch: four of our jobs running (C8 at 650M, readout 2, C6 protein 2, the main
sweep) alongside three from other users; GPU at 75/82 GB. C11 runs at batch 16 because of it.

## 2026-08-05 — C11 complete: the generalisation objection is closed

Both registered predictions confirmed; validity gate passed on both targets (findings F20).
AMPLIFY 120M — different vocabulary, different residue order, different corpus, different
feed-forward design — collapses under random alphabet relabeling exactly as ESM-2 does, and
shows the same In-class-vs-random dissociation widening with m.

Three things worth recording about how this went rather than what it found.

The gate earned its place. AMPLIFY's unpermuted P@L/2 is 0.237 on ubiquitin, which *looks*
like a failure next to ESM-2 650M's 0.763 and is in fact 4.4x chance. Without a
pre-registered ratio threshold the temptation would have been to eyeball 0.237, call the model
too weak to test, and drop a result that turned out to be one of the strongest in the project.
The gate also cost nothing, because it reused the baseline Jacobian the sweep needed anyway —
a design worth repeating.

The refuting branch did not fire, and it is worth saying so explicitly: Amendment 11 committed
in advance to making an invariant AMPLIFY the headline. It was not invariant. The prediction
was registered before the numbers existed and the numbers went the registered way.

One over-claim avoided. ESM-2 150M's In-class/random ratio at m=20 is 55.9x against
AMPLIFY's 7.3x, which reads as ESM-2 showing a far stronger dissociation. It does not — ESM-2's
denominator is 0.010, half of chance, so the ratio is an artifact of dividing by noise. F20
states the commensurable version instead: both architectures show the same monotone widening
and both retain an order of magnitude more structure under In-class relabeling. Not
quotable as a between-architecture ranking, and F20 says so in as many words.

C11 ran in 32 minutes on a GPU already 8-deep, 461s of which was module import off a saturated
filesystem, not compute.

## 2026-08-05 — C8 lands: the named consequence is now measured

C8 completed 8M/35M/150M/650M (findings F21). The result is sharper than "scale doesn't fix
it", and the sharper version is what the paper should say.

Under random relabeling the overlap is 1.8x / 1.2x / 1.2x / 0.8x chance as size goes 8M to
650M — flat, and the largest model is the most completely destroyed. Under In-class
relabeling it climbs 0.211 -> 0.367 -> 0.623 -> 0.501 and the In-class-minus-random gap
grows +0.170 -> +0.340 -> +0.596 -> +0.484. Scale buys sensitivity to biochemical similarity
and buys nothing on the axis a coevolutionary computation would be invariant along. Those are
different axes, and "bigger models understand proteins better" conflates them.

Two disciplines applied to our own result. The 150M -> 650M dip in the In-class arm is not
written up as a finding — n=8 per cell, two seeds, and the honest reading is a plateau after
150M. And 3B is stated as missing rather than quietly dropped: the registered budget guard
caught an OOM with 49.81 MiB free while other users held ~38 GB, wrote the complete 8M-650M
table, and named the gap. A polling retry at batch 4 is queued and gives up after 10 hours
rather than hammering a shared box; if it never lands the claim is quoted over 80x, not 375x.

## 2026-08-05 — the venue publishes its own rubric; the paper spine changes

Prompted to focus squarely on accept chance, I went to the source instead of continuing to
proxy for it, and found we had been calibrating against the wrong document. ICBINB-BIO
publishes both a REQUIRED four-section structure (Problem / Proposed approach / Observed
outcome / Reason for failure) and a five-point scoring rubric. Findings F22 records both
verbatim; paper/OUTLINE.md is rewritten around them.

Three things change.

The spine. Our narrative runs theorem -> measurement -> controls. The required structure does
not, and reorganising is not optional.

What counts as the contribution. Criterion 3 is "does the paper go beyond reporting a bad
number to characterize causes". C6, C8 and C11 are written throughout findings.md as CONTROLS —
instruments for closing reviewer objections. Against this rubric they are the contribution,
because together they are a causal account: C6 rules out input difficulty at matched Hamming
distance, C8 shows scale sharpens the biochemistry axis and never touches the abstraction axis,
C11 shows it is not one lab's tokenizer. Same experiments, different job.

Where C10 goes. Criterion 4 says "fails OR MISLEADS". ESM-2 falling from 2.33x Potts to 0.17x
Potts on the identical candidate set is the misleads case in one number. It moves into the body
with its own figure panel.

And one anxiety retires. F15 and F16 worried at length about shipping diagnosis with no fix.
This venue's guidelines explicitly penalise the opposite — leading with SOTA and skipping
failure analysis. The ICLR numbers (41.1% / 47.4%) were measuring a harder bar than we face.

## 2026-08-05 — review mining round 2 overturns the instrument strategy

The second mining pass (findings F23) came back with results that change decisions rather than
wording, and two of them contradict things this log previously recorded as settled.

F15 concluded that diagnosis-only clears the bar "only if the diagnostic is shipped as an
instrument". That is refuted. Artifact language moves acceptance by +2.0pp at p=0.73; papers
whose reviewers explicitly credited the artifact accept at exactly the baseline 36.4%; and in
ten of ten threads where both appeared, the same reviewer praised the tool and called the paper
not-a-contribution in a single review. Probes accept BELOW benchmarks corpus-wide. The audit
stays — it is cheap, true, and serves the reproducibility criterion — but it is not the pitch,
and building the paper around it would have been building on a measured non-effect.

More useful: we have been defending the wrong flank. "A necessary condition does not say the
model is bad" appears twice across 192 invariance-violation threads. Meanwhile "vacuous / true
by construction" accepts at 15.0% (n=20), the worst of any objection measured, and nothing in
the project answered it. The attack is easy to write: you defined a property only a
column-statistic model has, then showed a neural model lacks it.

The counter that works in the corpus is an attainability existence proof, and we have had one
all along without labelling it. Potts satisfies the condition exactly while being genuinely
accurate on the same input — so the property is discriminating, not tautological, and the proof
is a working system rather than an argument. C13, launched today, strengthens exactly this:
it extends the existence proof from our closed-form estimator to a pseudolikelihood fit with a
real optimiser, which is the estimator class the objection would otherwise retreat to.

One wording trap recorded because it is a knife-edge: reviewers credit cheapness when it is
framed as ACCESS to an otherwise-impossible measurement, and reject it when it reads as
convenience — an equally cheap label-free probe was rejected for being a "relatively simple
experiment". Identical fact, opposite reception.

## 2026-08-05 (late) — the gate earns its keep; C14 registered; first draft exists

**C13's registered correctness gate failed and that was the point.** plm P@L/2 = 0.2727 against
mean-field 0.3333, and zero-init invariance at 1.29e-02 relative where theory predicts
floating-point noise. Amendment 13 had pre-committed the response — *an implementation bug
until proven otherwise; debug before reporting* — so the branch was taken instead of argued
about. The bug: the L2 penalty was added to the summed NLL and the total then divided by `n`,
giving an effective coefficient of `lam/n` ≈ 3.3e-6 at n=3000. Unregularised, with 29,568
parameters per column against 3000 sequences. One cause, both symptoms: overfitting sank
precision, ill-conditioning made L-BFGS stop at different points so invariance looked loose.

Without the gate we would have published *"pseudolikelihood Potts is only approximately
invariant"* — a false negative against our own theory, in the direction of understating the
result. Recorded as Amendment 13a **before** any corrected number existed, because the
implementation changed after seeing a result and that has to be visible.

λ re-selected on the gate, on the **unpermuted** fit only (selecting on the DV would make C13
circular). All three cleared: 0.001 → 0.5152, 0.01 → 0.4848, 0.1 → 0.3636. Chose 0.001 by the
registered rule; it is also the least regularised and therefore worst conditioned, so it is the
hardest case for invariance rather than the easiest.

**Amendment 14 / C14 registered before writing the measurement code.** The strongest surviving
alternative explanation is that a relabeled sequence simply is not a protein to the model, so
the collapse is unremarkable. C6 answers the *edit-distance* form of that; Hamming distance is
not the model's own notion of distance. C14 measures Δ_LL = LL(x) − LL(π(x)) from the forward
pass the Jacobian already runs and asks whether regime still predicts overlap at *matched*
Δ_LL. Registered prediction: not mediated. Both branches are written down and the one I did
*not* predict is the one that flatters the paper — which is exactly why it had to be locked
first.

**First full draft.** `paper/DRAFT.md`, organised on the venue's four required sections rather
than on the order we discovered things in. Every number carries a `[csv:...]` provenance tag and
the pending experiments are marked as pending rather than assumed. Names the phenomenon
*alphabet anchoring* so the coinage can run through the section titles. A subagent is auditing
every number in it against the CSVs, because several were carried over from the outline rather
than recomputed — that is exactly the kind of drift that survives into a submission.

**New result, from aggregating the main sweep for the draft:** the Cross-class arm falls *faster*
than random at m=6 and m=10 (0.226 vs 0.342; 0.006 vs 0.079), converging at m=20. That answers
a standing Open Question and it is the sign "biochemistry is encoded" predicts — an arbitrary
derangement preserves biochemical class by accident more often than a deliberately cross-class
one does.

**Operational.** The box is at load ~33 on 8 cores with the GPU at 79/82 GB, so C14 is getting
about half a core and will take ~5 hours; that is acceptable with 24 days of runway and it was
left alone rather than restarted. C8's 3B cell is still polling for headroom and may never
land — the draft states its absence instead of hiding it.

## 2026-08-05 (night) — C13/C14/C15 land; an exploratory observation becomes a registered one

**C13 complete, 4/4 gates pass.** A pseudolikelihood Potts fit under a real optimiser returns an
*identical* contact set from an equivariant init and 0.970 overlap from a random one, with
permuted precision identical. The registered `<1e-8` score tolerance was missed at 1.9e-3, so
rather than relax it I registered Amendment 13b and ran a maxiter ladder: 1.1e-15 at two
iterations, 1.4e-14 at ten, 1.2e-09 at forty, 1.9e-03 at 150, with top-L/2 overlap exactly 1.000
at every rung. The solver grows the divergence; the formulation is symmetric. Claim bounded
accordingly — prediction-level invariance for both estimators, machine-precision score
invariance for the closed-form one only.

**C14 came out ambiguous and is reported as ambiguous.** ΔR² 0.131 against a 0.15 bar, partial
ρ +0.506 against a 0.30 bar. Branch C was registered for exactly this and says claim neither.

**The interesting thing in C14 was not its registered statistic**, and that was a problem: the
saturation of Δ_LL while overlap keeps collapsing was spotted after the fact, on one model and
one protein set, and it was carrying real weight in §4. So C15 registered it and tested it
out-of-sample against the C8 scale cells — a protein set sharing only PF00018 with C14's.

**It replicates. P1 passes at 4/4 scales** (+0.047, +0.034, +0.012, −0.056), across an 80×
parameter range. The registered *conjunction* holds at 3/4: 35M fails P2 because it had already
fallen to 0.118 overlap by m=6 and had under 0.10 of absolute room left. That is reported as a
failure; it is not restated in relative terms to make it pass.

**What C15 buys that C14 could not: a double dissociation.** At every scale the random arm has
Δ_LL flat while overlap collapses to chance, and the In-class arm has Δ_LL climbing ~3×
while overlap stays high. A rising perceived shift is compatible with a preserved contact map,
and a flat one with a destroyed map. Whatever governs the contact map, it is not how unfamiliar
the model finds the input — which is the cleanest available answer to "the relabeled sequence is
just OOD", and it is now measured rather than argued.

**Method note worth keeping.** Three separate times now the thing that caught an error was a
control with a threshold and a response fixed *before* the run: the gate that found the
regularisation bug, the audit that found §4.1 describing an experiment C6 never ran, and this
registration that turned a flattering after-the-fact observation into a testable one. In all
three the unregistered version would have been reported and would have been wrong or
overstated.

## 2026-08-05 (late) — C6 lands complete; readout 2 is 10/15 through and reproducing exactly

**C6 matched-Hamming, COMPLETE.** 306 rows, 102 within-cell matched triples, all three
proteins × three regimes × four m × three seeds. The Hamming-match invariant verified 102/102.
The registered prediction (Amendment 4) holds in the In-class regime and fails to be
needed in the others: In-class relabeling beats a distance-matched substitution by +0.49 at
m = 10 and +0.44 at m = 20, while random and Cross-class relabelings buy nothing at all. That
second half was not anticipated and is the sharper result — it makes the failure regime-graded
rather than uniform, and both halves are failures against the Potts reference of exactly 1.000.

Rewrote paper §4.1 around the 102-cell aggregate. The previous version rested on one cytochrome
c cell; the aggregate is stronger and exposes a regime dependence the anecdote concealed.

**Readout 2 (variant effect) is the generalisation the paper needed.** Registered separately
in `protocol-readout2-variant-effect.md` (locked, plus Amendment 9), so no protocol debt. 10 of
15 ProteinGym assays done. Every identity-condition ρ reproduces the published ESM2_650M number
to four decimals — the validity gate passes 10/10. The PSSM baseline, scored end-to-end on the
*relabeled* MSA with relabeled mutations, is invariant to exact floating-point zero in every
cell. ESM-2 at random m = 20 loses 0.55–0.74 of ρ on the six signal-bearing assays; In-class
loses 0.21–0.32. Same qualitative shape as the contact-map readout, on a different task with a
different metric and an external ground truth. Left to run: RL40A, TPK1, TRPC, VKOR1 ×2.

Also noted: a LaTeX build exists at `final_paper/main.tex` and compiles to **9 pages** against
an 8-page limit. One page must come out before submission.

**C16 registered, run and banked the same evening.** Grepping for `predict_contacts` returned
nothing outside `src/make_fair_esm_weights.py` — i.e. the project had never once used ESM-2's
own contact predictor, only the categorical Jacobian we built. That is the cheapest serious
reviewer objection available ("is the anchoring in the model or in your readout?"), and the
native head is ~1500× cheaper to evaluate than the Jacobian, so the whole 333-cell sweep runs
in minutes. Amendment 16 was locked first, with a refutation branch that would have narrowed
every contact claim in the abstract had P1 failed.

All three passed: random m=20 collapses to 0.028, the In-class gap is 0.765, and the two
instruments agree at Spearman +0.884 across 204 matched cells. The grading is *stronger* in the
shipped predictor than in ours. Two things the run did right and I want to keep doing: it
printed the 120 dropped join cells with a reason for each rather than silently correlating what
was left, and it computed both min_sep values when it found the amendment's parenthetical
contradicted Amendment 3, reporting the primary DV and the sensitivity separately.

Outstanding: re-run `run_c16_native_contact_head.py --analyse-only` after the main sweep
finishes, to recover the 108 cells dropped because ubiquitin / cytochrome c / T4 lysozyme are
not yet in the Jacobian CSV.

**C12 closed, and it did not go the way the protocol hoped.** Protein G came in at ρ = +0.079,
ubiquitin at −0.328, against a registered −0.6. Neither registered branch is cleanly met and I
have recorded it that way rather than rounding to the nearest one. Protein G's overlap range is
0.000–0.143 against a chance floor of 0.028 — a floor effect, the identical failure mode that
killed H4's registered design. I flagged the range restriction as a fact about the measured
data but did **not** use it to discard the cell, because Amendment 12 pre-committed no such
escape hatch. The conclusion is unchanged on every reading, which is what makes it safe to
state: no dose–response, and the paper must not imply one.

The reconciliation with C6/C16 is the interesting part, and it is now §2.4's closing paragraph:
whether a relabeling preserves biochemical *groups* matters enormously (gaps of 0.44–0.77),
while how far it moves residues along a scalar BLOSUM-weighted cost predicts nothing. The
effect is categorical in the structure of the relabeling, not graded in its magnitude. That is
a stronger constraint on the mechanism than a dose–response would have been — it says ESM-2 is
not computing a smooth biochemical-similarity function. Amendment 12's refutation branch had
literally anticipated "something coarser … whether specific tokens move at all" and demanded a
rewrite rather than a softening; that is what happened.

---

## 2026-08-12 — All experiments complete; paper updated and compiled

**All in-flight experiments have landed.** The GPU is free and no icbinb jobs are running.

- **C8 3B cell**: LANDED. random m=20 = 0.035 (1.1× chance), In-class = 0.526, gap = +0.491.
  Scale table now has 5 rows spanning a 375× parameter range. The 650M→3B step is a plateau,
  not an improvement — consistent with the 150M→650M dip already reported.
- **Readout 2 (variant effect)**: COMPLETE, 15/15 assays, 15/15 gates PASS. 11 signal-bearing
  (ρ_identity ≥ 0.20). random m=20 Δρ = −0.52 (signal), −0.43 (all). In-class = −0.19/−0.16.
  Cross-class = −0.54/−0.44. PSSM invariant to exact 0 in every cell. The same categorical
  dissociation replicates on a different task, a different metric, and against external ground truth.
- **Main sweep**: COMPLETE, 9 proteins, 315 rows. Table 1 updated with 9-protein aggregates.
  Numbers shifted slightly from the 6-protein version (e.g. random m=20: 0.010→0.015,
  In-class m=20: 0.733→0.655) but the story is unchanged.

**LaTeX paper (final_paper/main.tex)**: 8 pages, compiles clean, no undefined references.
All numbers verified against committed CSVs. Three-panel figure (instrument/scale/crossing)
generated and included. Readout 2 moved to appendix to fit page limit. Abstract updated to
reflect 375× range. §4.4 corrected: C15 covers 4 scales (not 5 — 3B was not in the C15 run),
overlap numbers in §4.4 use C14's 6-protein data (matching the Δ_LL source).

**Remaining for submission:**
1. Citation pass — fill .bib stub entries (user's job).
2. Anonymised supplementary code package.
3. Final proofread for LLM-heavy prose patterns.
4. Update or retire paper/DRAFT.md.

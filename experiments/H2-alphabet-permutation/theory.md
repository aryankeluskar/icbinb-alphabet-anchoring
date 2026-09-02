# Alphabet equivariance as a necessary condition for coevolutionary inference

The empirical result ("Potts is invariant, pLMs are not") is only as strong as the reason
Potts is invariant. If it is an empirical observation, a reviewer may reasonably suspect a
lucky estimator. It is not an observation — it is a theorem, and the numerical runs are
verification of an implementation rather than evidence for a claim. This file states it
precisely, including the conditions under which it fails, because those conditions are what
make it a real proposition rather than a slogan.

---

## Setup

An alignment is a matrix `X ∈ A^{N×L}` over the 20-letter amino-acid alphabet `A`
(N sequences, L columns). A relabeling is a bijection `π : A → A`, acting entrywise:

    (πX)_{ni} = π(X_{ni}).

The action is **global**: the same π is applied to every cell of the alignment. This is the
only version for which anything below holds, and it is the version we run. Two nearby
operations that are *not* this, and do not inherit the theorem:

- a per-sequence bijection (a different π per row) — destroys nothing in the mutual
  information sense but blurs coevolution across rows; measured at recall 1.000 → 0.600 in
  findings.md F2, and it is **not** a null;
- a per-column bijection — genuinely destroys the pairing between columns.

Write `P_π ∈ {0,1}^{20×20}` for the permutation matrix of π, so `P_π` is orthogonal:
`P_π P_πᵀ = I`.

## What relabeling does to alignment statistics

Single-column and pairwise empirical frequencies satisfy

    f^π_i(π a)      = f_i(a),
    f^π_{ij}(πa, πb) = f_{ij}(a, b).

So the statistics are **permuted, not altered**. No information is added or removed; the
histogram is the same histogram with its bins renamed. This is the entire content of the
transformation, and it is why "the task did not get harder" is a statement of fact rather
than an assumption we are asking a reader to grant.

---

## Proposition 1 (maximum-entropy Potts)

Let `(h, J)` maximise the regularised pseudolikelihood of `X` under the Potts Hamiltonian

    H(σ) = − Σ_i h_i(σ_i) − Σ_{i<j} J_{ij}(σ_i, σ_j),

with an `L₂` penalty `λ Σ_{i<j} ‖J_{ij}‖_F²`. Define the relabelled parameters

    h^π_i(πa) = h_i(a),        J^π_{ij}(πa, πb) = J_{ij}(a, b),

equivalently `J^π_{ij} = P_π J_{ij} P_πᵀ`. Then `(h^π, J^π)` is optimal for `πX`, and the
attained objective is identical.

**Proof.** By construction `H^π(πσ) = H(σ)` for every configuration σ, so the induced
Gibbs measure satisfies `P^π(πσ) = P(σ)`: the relabelled model is the pushforward of the
original along π. Since π is a bijection on `A^L`, the likelihood of `πX` under `(h^π, J^π)`
equals that of `X` under `(h, J)` term by term. The penalty is preserved because
`‖P_π J P_πᵀ‖_F = ‖J‖_F` — conjugation by an orthogonal matrix is a Frobenius isometry, and
here it is merely a permutation of the entries. The map `(h,J) ↦ (h^π, J^π)` is a bijection
on parameter space that preserves the objective, so it carries maximisers to maximisers. ∎

## Proposition 2 (the estimator we actually run: regularised inverse covariance)

Our implementation is not maximum-entropy Potts; it is the mean-field / Gaussian estimator,
and the proposition must be proved for the estimator that produced the numbers.

One-hot encode the alignment as `Y ∈ R^{N×20L}`, blocked by position. Relabeling permutes
the 20 coordinates *within each block identically*:

    Y^π = Y (I_L ⊗ P_π),   and write Q = I_L ⊗ P_π,  which is orthogonal.

The empirical covariance transforms by congruence, `C^π = Qᵀ C Q`. The couplings are the
blocks of the regularised precision matrix `W = (C + λI)^{-1}`. Then

    W^π = (Qᵀ C Q + λI)^{-1} = (Qᵀ (C + λI) Q)^{-1} = Qᵀ (C + λI)^{-1} Q = Qᵀ W Q,

using `QᵀQ = I` for the middle step. Blockwise this reads `W^π_{ij} = P_πᵀ W_{ij} P_π`.

The contact score applies, in order: mean-centring over each alphabet axis, symmetrisation,
a Frobenius norm per 20×20 block, and APC. Each step commutes with the conjugation:

- mean-centring — the row and column means are *permuted*, not changed, so subtracting them
  commutes with `P_π`;
- symmetrisation — `(Qᵀ W Q)ᵀ = Qᵀ Wᵀ Q`;
- Frobenius per block — `‖P_πᵀ W_{ij} P_π‖_F = ‖W_{ij}‖_F`, orthogonal invariance;
- APC — a function of the resulting `L×L` score matrix, which by the previous step is
  already unchanged.

Hence `S^π_{ij} = S_{ij}` **exactly**, and every downstream quantity (ranking, top-L/2 set,
precision) is identical, not merely close. ∎

### Why Proposition 2 carries the paper and Proposition 1 does not

The two propositions are not interchangeable, and the difference is the kind of objection each
one leaves available.

**Proposition 1 is an optimality statement.** It says the relabelled parameters *are a
maximiser* of the relabelled objective. That leaves a live objection: nobody solves the Potts
maximum-likelihood problem exactly. Real pipelines run pseudolikelihood with early stopping,
or a particular optimiser, or a particular initialisation — and an argument about maximisers
says nothing about which maximiser a given solver returns, or how close it gets. Answering
that objection means arguing about solvers, which is a fight rather than a proof.

**Proposition 2 is an identity on the computed output.** `S^π = S` is not a claim about the
optimum of anything; it is an equality between two matrices that a specific, closed-form
computation emits. There is no solver, no convergence, no initialisation, and therefore no
gap between "what the theory describes" and "what the code runs". The objection is not
answered — it is *structurally unavailable*. That is the property to protect when writing this
up, and it is why the empirical Potts numbers are correctness asserts rather than evidence.

**It is also universally quantified over the nuisance parameter.** The experiments sample a
handful of permutations per cell. The proposition holds for **every** π in the symmetric group
on the 20 residues — all 20! ≈ 2.4 × 10¹⁸ of them — simultaneously, because nothing in the
argument depends on which permutation matrix `P_π` is. A reviewer cannot ask what happens for
some permutation we did not try. That is the same asset a universally-quantified negative
result buys elsewhere: the sampled sweep is an illustration, not the claim.

Keep both propositions, because Proposition 1 is what connects the result to the *idea* of a
coevolutionary model rather than to our particular estimator. But when the two are in tension
— when a reader asks which one the result rests on — the answer is Proposition 2.

### Where it breaks — the conditions are load-bearing

The `λI` in Proposition 2 is not cosmetic. The step `Qᵀ(C+λI)Q = QᵀCQ + λI` requires the
regulariser to be **isotropic**. A per-residue or otherwise non-uniform ridge would give
`Qᵀ Λ Q ≠ Λ` and the invariance would fail. Likewise:

- **Gap and non-standard characters must be excluded from the permuted set.** Our
  `PROTECTED = {X,B,U,Z,O,-,.,*}` guard enforces this at the point of application. It is
  load-bearing, not defensive: a positive control that bypasses the guard moves the contact
  map by 0.157 versus 1.8e-15 with it (findings.md F1).

  **Stated precisely, because a first attempt at testing this got it wrong.** `mrf.ALPHABET`
  is `AA20 + GAP` — 21 states — and `one_hot_msa` folds *every* non-standard character into
  the single gap state. Two consequences that must not be conflated:

  - A permutation **within** the protected set (say `-` ↔ `X`) is invisible to the
    estimator, because both characters already occupy the same slot. Invariance holds
    trivially, and observing it proves nothing.
  - A permutation **across** the boundary (say `-` → `K`) moves probability mass between the
    gap slot and a residue slot, changes `C`, and breaks invariance. *That* is what the guard
    forbids and what a positive control must exercise.

  A control that swaps two protected characters and reports "invariant" is measuring the
  encoder's blind spot, not the theorem. See `code/test_theory_conditions.py`.
- **Sequence-reweighting schemes must be relabeling-blind.** Weights derived from sequence
  identity are, since identity is preserved under a global bijection. A weighting that
  referenced specific residue types would not be.

Stating these is what makes the proposition falsifiable rather than decorative — and each
one is a real way an implementation could silently violate it.

---

## Corollary (the claim the paper actually makes)

> Let `g` be any predictor whose output depends on the alignment only through the empirical
> column and column-pair statistics `{f_i}, {f_ij}`, and which is equivariant to renaming
> the alphabet in the trivial sense that renaming the input renames the output. Then `g`'s
> contact scores are invariant under a global relabeling π.

Contrapositive, which is the useful direction:

> **A model whose contact predictions change under a global alphabet relabeling is not
> computing a function of alignment column statistics.**

This is a *necessary condition*, and that is precisely its value. It is not a probe whose
interpretation is arguable, and it does not require us to say what the model represents. It
lets us rule out a hypothesis about the computation being performed without ever having to
establish what the model is doing instead.

**Two things it does not license.**

1. It does not say the model is bad. A lookup table keyed on literal residue identity can be
   highly accurate on families resembling its training data — and ESM-2 650M is: P@L/2 =
   0.763 on ubiquitin, 14× chance, before permutation (F12). The claim is about *mechanism*,
   and the accuracy number is what makes the mechanism claim interesting rather than a
   description of a weak model.
2. It does not, on its own, say the mechanism matters. Establishing that requires showing a
   downstream consequence — a regime where the lookup mechanism and the statistical one come
   apart. That is the open question and the natural second half of a longer paper.

---

## Relationship to the numerical results

With Propositions 1 and 2 in hand, every Potts number in this project is a **correctness
assert on the implementation**, not evidence for the hypothesis:

| Check | Measured | Interpretation |
|---|---|---|
| synthetic, 15 planted couplings | max\|ΔS\| ≤ 5.3e-15 | float64 noise |
| real Pfam seed alignments | ≤ 2.5e-16, top-L/2 set *identical* | exact |
| adversarial (15% gaps, 6 non-standard chars) | ≤ 1.8e-15 | guard works |
| C7, in-run, 136 cells | overlap exactly 1.000000, min = max | alignment provably unchanged |
| readout 2, PSSM Δρ | exactly 0.0e+00 at every cell | Prop. 2 applies to site-independent models too |

And the two **negative** controls, which are what make the conditions falsifiable rather than
decorative (`code/test_theory_conditions.py`, N=1500, L=40, m=20 random relabeling):

| Violated condition | Measured | Interpretation |
|---|---|---|
| anisotropic ridge, spread only 1.0×–1.5× λ | max\|ΔS\| = 3.5e-01 (11.8% of scale); top-L/2 overlap 0.850 | isotropy is load-bearing |
| `gap → K`, MSA with 10% gaps + 1.5% `X` | max\|ΔS\| = 1.8e-01 | the `PROTECTED` guard is a hypothesis of the proposition |

Both directions pass; the script exits non-zero if either fails.

The last row is worth noting: the site-independent PSSM is the `J = 0` special case of
Proposition 1, so its exact invariance is the same theorem with fewer terms.

**Consequence for how the result is presented.** The comparison is not "our baseline
happened to beat the neural model." It is "the baseline satisfies a property by theorem, the
neural model violates it by measurement." Those are different kinds of claim, and only the
second one is immune to the objection that we tuned a baseline until it won.

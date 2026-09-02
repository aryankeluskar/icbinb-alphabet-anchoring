"""Amino-acid alphabet permutations for the H2 invariance experiment.

The central object here is a bijection pi over the 20 standard amino acids, applied
CONSISTENTLY to every sequence in a family. Under such a relabeling a Potts/MRF
coevolution model is exactly invariant (coevolution is a statistic over alignment
columns, not over residue identity), so any degradation observed in a pLM isolates
reliance on literal token identity.

Special tokens and non-standard residues (X B U Z O - .) are NEVER permuted.
"""
import numpy as np

# The 20 standard amino acids, in a fixed canonical order.
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = frozenset(AA20)

# Anything that must pass through a permutation untouched.
PROTECTED = frozenset("XBUZO-.*")

# BLOSUM62-similar groups, used to build "In-class" permutations that preserve
# biochemistry. Swapping within a group is a small biochemical perturbation.
IN_CLASS_GROUPS = [
    "IVLM",   # aliphatic / hydrophobic
    "KR",     # positively charged
    "DE",     # negatively charged
    "NQ",     # amide
    "ST",     # small hydroxyl
    "FYW",    # aromatic
]

# Cross-class pairings, used to build "Cross-class" permutations that maximally violate
# biochemistry at the same edit budget.
CROSS_CLASS_PAIRS = [
    ("K", "D"),  # + <-> -
    ("R", "E"),  # + <-> -
    ("I", "E"),  # hydrophobic <-> charged
    ("V", "K"),  # hydrophobic <-> charged
    ("F", "D"),  # aromatic <-> charged
    ("W", "N"),  # bulky aromatic <-> small polar
    ("L", "S"),  # hydrophobic <-> small polar
    ("M", "Q"),  # hydrophobic <-> amide
    ("Y", "G"),  # aromatic <-> tiny
    ("A", "P"),  # helix-favouring <-> helix-breaking
]


def identity_perm():
    """The m=0 control. Must reproduce the unpermuted baseline exactly."""
    return {a: a for a in AA20}


def _derangement(items, rng):
    """A permutation of `items` with no fixed points.

    Rejection sampling: for the small alphabets here (m <= 20) the probability that a
    random permutation is a derangement is ~1/e, so this terminates fast.
    """
    if len(items) < 2:
        raise ValueError("a derangement needs at least 2 items")
    for _ in range(10_000):
        shuffled = list(rng.permutation(items))
        if all(a != b for a, b in zip(items, shuffled)):
            return dict(zip(items, shuffled))
    raise RuntimeError("failed to sample a derangement")


def random_perm(m, rng):
    """Permute `m` randomly-chosen amino acid types among themselves; fix the rest."""
    if m == 0:
        return identity_perm()
    if not 2 <= m <= 20:
        raise ValueError(f"m must be 0 or in [2, 20], got {m}")
    chosen = list(rng.choice(list(AA20), size=m, replace=False))
    perm = identity_perm()
    perm.update(_derangement(chosen, rng))
    return perm


def in_class_perm(m, rng):
    """Swap only within BLOSUM62-similar groups, touching ~m residue types.

    Biochemistry is approximately preserved, so a model that has learned biochemical
    regularities (rather than token identities) should be relatively unharmed.
    """
    if m == 0:
        return identity_perm()
    perm = identity_perm()
    groups = [list(IN_CLASS_GROUPS[k]) for k in rng.permutation(len(IN_CLASS_GROUPS))]
    touched = 0
    for group in groups:
        if touched >= m:
            break
        take = min(len(group), m - touched)
        if take < 2:
            continue
        sub = list(rng.choice(group, size=take, replace=False))
        perm.update(_derangement(sub, rng))
        touched += take
    return perm


def cross_class_perm(m, rng):
    """Swap across biochemical classes, touching ~m residue types.

    Same edit budget as `in_class_perm` but maximally violates biochemistry.
    Comparing the two at matched m separates "knows biochemistry" from "knows tokens".
    """
    if m == 0:
        return identity_perm()
    perm = identity_perm()
    pairs = [CROSS_CLASS_PAIRS[k] for k in rng.permutation(len(CROSS_CLASS_PAIRS))]
    touched = 0
    used = set()
    for a, b in pairs:
        if touched + 2 > m:
            break
        if a in used or b in used:
            continue
        perm[a], perm[b] = b, a
        used.update((a, b))
        touched += 2
    return perm


PERM_BUILDERS = {
    "random": random_perm,
    "In-class": in_class_perm,
    "Cross-class": cross_class_perm,
}


def apply_perm(seq, perm):
    """Apply pi to a single sequence, passing protected symbols through untouched."""
    return "".join(perm.get(c, c) if c not in PROTECTED else c for c in seq)


def apply_perm_msa(msa, perm):
    """Apply the SAME pi to every sequence — this is what makes MRF invariant."""
    return [apply_perm(s, perm) for s in msa]


def apply_perm_msa_inconsistent(msa, m, rng):
    """Control C3: a DIFFERENT random pi per sequence.

    This destroys the column statistics, so both the MRF and the pLM must collapse.
    Confirms our metric detects genuine signal loss, making the MRF's invariance under
    a consistent permutation a non-trivial observation rather than a metric artefact.
    """
    return [apply_perm(s, random_perm(m, rng)) for s in msa]


def composition(seq):
    """Amino-acid counts. Control C5: a consistent permutation permutes the counts but
    preserves the multiset, so sorted(counts) must be invariant. Any change is a bug."""
    counts = {a: 0 for a in AA20}
    for c in seq:
        if c in AA20_SET:
            counts[c] += 1
    return counts


def check_composition_invariant(before, after):
    """Assert the multiset of residue counts survived the permutation."""
    return sorted(composition(before).values()) == sorted(composition(after).values())

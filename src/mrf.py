"""Potts / MRF coevolution baseline via inverse covariance.

Implements the multivariate-Gaussian formulation of Dauparas et al., which Zhang &
Ovchinnikov (PNAS 2024) use as the linear-model comparator for the categorical Jacobian:

    W = -inv( X_c^T X_c / N  +  lambda * I )

where X is the one-hot MSA, X_c is mean-centred, followed by the average product
correction (APC) to produce a contact map.

The property this module exists to establish: W is EXACTLY invariant (up to the
corresponding relabeling of its 20x20 blocks) under a consistent bijective relabeling
of the amino-acid alphabet applied across the whole MSA. Therefore the APC contact map
is exactly invariant. This is a mathematical fact, and verifying it numerically is the
correctness gate (control C2) for the whole H2 experiment.

Kept free of modern syntax so it runs on the cluster's stock Python 3.6 as well as the
purpose-built environment.
"""
from typing import List, Tuple

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
ALPHABET = AA20 + GAP          # 21 states: 20 residues + gap
A = len(ALPHABET)
AA_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def one_hot_msa(msa):
    """Encode an MSA as a float array of shape (N, L, A).

    Unknown / non-standard characters are mapped to the gap state, matching the usual
    convention in coevolution pipelines.
    """
    n, length = len(msa), len(msa[0])
    if any(len(s) != length for s in msa):
        raise ValueError("MSA is ragged; all sequences must be the same length")
    x = np.zeros((n, length, A), dtype=np.float64)
    gap_idx = AA_TO_IDX[GAP]
    for i, seq in enumerate(msa):
        for j, c in enumerate(seq):
            x[i, j, AA_TO_IDX.get(c, gap_idx)] = 1.0
    return x


def couplings(msa, lam=None):
    """Fit pairwise couplings W with shape (L, A, L, A) by inverse covariance.

    lam defaults to the shrinkage used by Dauparas et al., which scales with the number
    of effective sequences.
    """
    x = one_hot_msa(msa)
    n, length, _ = x.shape
    flat = x.reshape(n, length * A)
    centred = flat - flat.mean(axis=0, keepdims=True)
    cov = centred.T.dot(centred) / n
    if lam is None:
        lam = 4.5 / np.sqrt(n)
    w = -np.linalg.inv(cov + lam * np.eye(length * A))
    return w.reshape(length, A, length, A)


def apc(matrix):
    """Average product correction: M_ij - (row_i * col_j) / total."""
    row = matrix.mean(axis=1, keepdims=True)
    col = matrix.mean(axis=0, keepdims=True)
    total = matrix.mean()
    if total == 0:
        return matrix.copy()
    return matrix - (row * col) / total


def contact_map(msa, lam=None, apply_apc=True):
    """Reduce couplings to an (L, L) contact score via the Frobenius norm + APC.

    The 20 residue states are used for the norm and the gap state is excluded, which is
    standard: gap-gap couplings reflect alignment artefacts, not contacts.
    """
    w = couplings(msa, lam=lam)
    length = w.shape[0]
    aa = len(AA20)
    # Frobenius norm over the 20x20 residue block of each (i, j) pair.
    scores = np.sqrt((w[:, :aa, :, :aa] ** 2).sum(axis=(1, 3)))
    np.fill_diagonal(scores, 0.0)
    scores = apc(scores) if apply_apc else scores
    return 0.5 * (scores + scores.T)      # symmetrise


def top_contacts(scores, seq_len, min_sep=24, top_frac=0.5):
    """Indices of the top-scoring long-range pairs, |i - j| >= min_sep."""
    idx = []
    for i in range(seq_len):
        for j in range(i + min_sep, seq_len):
            idx.append((scores[i, j], i, j))
    idx.sort(reverse=True)
    k = max(1, int(seq_len * top_frac))
    return [(i, j) for _, i, j in idx[:k]]


def precision_at_frac(scores, true_contacts, seq_len, min_sep=24, top_frac=0.5):
    """Precision of the top L*top_frac long-range predicted contacts.

    NOTE: when the number of true contacts is smaller than the number of predictions,
    precision is CEILINGED at n_true / n_pred and cannot reach 1.0. Use recall_at_frac
    alongside it, or precision is easy to misread as a weak result.
    """
    pred = top_contacts(scores, seq_len, min_sep=min_sep, top_frac=top_frac)
    if not pred:
        return float("nan")
    truth = set(map(tuple, true_contacts)) | set((j, i) for i, j in true_contacts)
    hits = sum(1 for p in pred if p in truth)
    return hits / float(len(pred))


def recall_at_frac(scores, true_contacts, seq_len, min_sep=24, top_frac=0.5):
    """Fraction of the true long-range contacts recovered in the top L*top_frac."""
    pred = set(top_contacts(scores, seq_len, min_sep=min_sep, top_frac=top_frac))
    truth = [tuple(c) for c in true_contacts if abs(c[0] - c[1]) >= min_sep]
    if not truth:
        return float("nan")
    hits = sum(1 for (i, j) in truth if (i, j) in pred or (j, i) in pred)
    return hits / float(len(truth))


def precision_ceiling(true_contacts, seq_len, min_sep=24, top_frac=0.5):
    """The maximum precision attainable given how many true contacts exist."""
    n_pred = max(1, int(seq_len * top_frac))
    n_true = len([c for c in true_contacts if abs(c[0] - c[1]) >= min_sep])
    return min(1.0, n_true / float(n_pred))


def shuffle_columns(msa, rng):
    """The CORRECT coevolution null: independently permute the rows within each column.

    This preserves every column's marginal residue distribution exactly while destroying
    all inter-column dependency. Contrast with a per-sequence alphabet permutation, which
    relabels within a row and therefore PRESERVES the mutual information between columns.
    """
    arr = np.array([list(s) for s in msa])
    out = arr.copy()
    for j in range(arr.shape[1]):
        out[:, j] = rng.permutation(arr[:, j])
    return ["".join(row) for row in out]

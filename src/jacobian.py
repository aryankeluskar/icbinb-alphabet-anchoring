"""Categorical Jacobian for ESM-2, following Zhang, Wayment-Steele, Brixi & Ovchinnikov
(PNAS 2024).

The construction: for every position i and every amino acid a, substitute a at i, run a
forward pass, and subtract the unmutated logits. This gives

    J[i, a, j, b] = logits(mutate i -> a)[j, b] - logits(wild-type)[j, b]

of shape (L, A, L, A). Mean-centring over both alphabet axes and symmetrising, then taking
the Frobenius norm over each 20x20 block and applying APC, yields a contact map.

Two details from the paper that are easy to get wrong and both matter:

  - the perturbation must be CATEGORICAL, not infinitesimal. Substituting the actual 20
    amino acids is what works; substituting the mask token instead drops P@L from 0.80 to
    0.76 in their hands.
  - BOS/EOS must be replaced with mask tokens or contact recovery degrades.

Cost is L x 20 forward passes per protein, which is why everything here is batched.

Why this module exists for H2: it is the like-for-like readout against Potts couplings.
The MRF is exactly invariant under a consistent alphabet relabeling (verified to 1e-15);
the question is whether ESM-2's Jacobian is.
"""
import numpy as np
import torch

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _apc(matrix):
    """Average product correction."""
    row = matrix.mean(axis=1, keepdims=True)
    col = matrix.mean(axis=0, keepdims=True)
    total = matrix.mean()
    if total == 0:
        return matrix.copy()
    return matrix - (row * col) / total


class ESM2Jacobian(object):
    """Wraps a fair-esm model and computes categorical Jacobians."""

    def __init__(self, model, alphabet, device="cuda", batch_size=64,
                 mask_bos_eos=True):
        self.model = model.eval().to(device)
        self.alphabet = alphabet
        self.device = device
        self.batch_size = batch_size
        self.mask_bos_eos = mask_bos_eos
        # Token indices for the 20 standard residues, in canonical order.
        self.aa_idx = [alphabet.get_idx(a) for a in AA20]
        self.mask_idx = alphabet.mask_idx

    def _tokenize(self, seq):
        """Sequence -> token tensor of shape (L + 2,), including BOS/EOS."""
        toks = [self.alphabet.cls_idx]
        toks += [self.alphabet.get_idx(c) for c in seq]
        toks += [self.alphabet.eos_idx]
        return torch.tensor(toks, dtype=torch.long)

    @torch.no_grad()
    def _logits(self, batch_toks):
        """Forward pass -> logits over the 20 residues only, shape (B, L, 20)."""
        batch_toks = batch_toks.to(self.device)
        if self.mask_bos_eos:
            # The paper replaces BOS/EOS with mask; without this, contact recovery degrades.
            batch_toks = batch_toks.clone()
            batch_toks[:, 0] = self.mask_idx
            batch_toks[:, -1] = self.mask_idx
        out = self.model(batch_toks, repr_layers=[], return_contacts=False)
        logits = out["logits"]                      # (B, L+2, vocab)
        logits = logits[:, 1:-1, :]                 # drop BOS/EOS positions
        return logits[:, :, self.aa_idx]            # (B, L, 20)

    @torch.no_grad()
    def jacobian(self, seq):
        """Categorical Jacobian, shape (L, 20, L, 20), float32 numpy."""
        length = len(seq)
        base_toks = self._tokenize(seq)

        wt = self._logits(base_toks.unsqueeze(0))[0].float().cpu().numpy()  # (L, 20)

        jac = np.zeros((length, 20, length, 20), dtype=np.float32)

        # Build every single-point mutant, then run them in batches.
        variants = [(i, a) for i in range(length) for a in range(20)]
        for start in range(0, len(variants), self.batch_size):
            chunk = variants[start:start + self.batch_size]
            toks = base_toks.unsqueeze(0).repeat(len(chunk), 1).clone()
            for row, (i, a) in enumerate(chunk):
                toks[row, i + 1] = self.aa_idx[a]   # +1 for BOS
            out = self._logits(toks).float().cpu().numpy()   # (B, L, 20)
            for row, (i, a) in enumerate(chunk):
                jac[i, a] = out[row] - wt

        return jac

    def contact_map(self, seq, jac=None):
        """Reduce the Jacobian to an (L, L) contact score, as in the paper."""
        if jac is None:
            jac = self.jacobian(seq)
        # Mean-centre over both alphabet axes.
        jac = jac - jac.mean(axis=1, keepdims=True)
        jac = jac - jac.mean(axis=3, keepdims=True)
        # Symmetrise: J[i,a,j,b] with its transpose J[j,b,i,a].
        jac = 0.5 * (jac + np.transpose(jac, (2, 3, 0, 1)))
        # Frobenius norm over each 20x20 block.
        scores = np.sqrt((jac ** 2).sum(axis=(1, 3)))
        np.fill_diagonal(scores, 0.0)
        scores = _apc(scores)
        return 0.5 * (scores + scores.T)


def load_esm2(name="esm2_t33_650M_UR50D", device="cuda"):
    """Load a fair-esm model by name."""
    import esm
    model, alphabet = getattr(esm.pretrained, name)()
    return ESM2Jacobian(model, alphabet, device=device)


def top_contacts(scores, min_sep=24, top_frac=0.5):
    """Top-scoring long-range pairs, |i - j| >= min_sep."""
    length = scores.shape[0]
    cand = []
    for i in range(length):
        for j in range(i + min_sep, length):
            cand.append((scores[i, j], i, j))
    cand.sort(reverse=True)
    k = max(1, int(length * top_frac))
    return [(i, j) for _, i, j in cand[:k]]


def set_overlap(a, b):
    """Jaccard-style agreement between two contact sets: |A n B| / |A|."""
    sa, sb = set(a), set(b)
    if not sa:
        return float("nan")
    return len(sa & sb) / float(len(sa))

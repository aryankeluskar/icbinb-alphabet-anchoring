"""Test the CONDITIONS stated in theory.md, not the conclusion.

theory.md Proposition 2 proves exact invariance for the regularised inverse-covariance
estimator, and claims two conditions are load-bearing:

  (i)  the ridge must be ISOTROPIC. The proof step Q'(C + lam I)Q = Q'CQ + lam I needs
       Q'(Lambda)Q = Lambda, which holds for Lambda = lam*I and fails otherwise.
  (ii) non-standard characters must be excluded from the permuted set.

A condition nobody tested is a condition that might not be real. If a non-isotropic ridge
does NOT break invariance, then the proof is stating a stronger requirement than the truth
and the writeup would be wrong. So this asserts both directions:

  isotropic ridge   -> invariance holds to float64 noise      (POSITIVE control)
  anisotropic ridge -> invariance measurably BREAKS           (NEGATIVE control)

Run: python3 -u test_theory_conditions.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mrf          # noqa: E402
import permute      # noqa: E402

A = mrf.A          # 21: the 20 amino acids plus a catch-all slot


def couplings_ridge(msa, ridge_diag):
    """mrf.couplings with an ARBITRARY diagonal ridge instead of lam*I."""
    x = mrf.one_hot_msa(msa)
    n, length, _ = x.shape
    flat = x.reshape(n, length * A)
    centred = flat - flat.mean(axis=0, keepdims=True)
    cov = centred.T.dot(centred) / n
    w = -np.linalg.inv(cov + np.diag(ridge_diag))
    return w.reshape(length, A, length, A)


def score_from_w(w):
    """The same scoring pipeline mrf.contact_map uses: centre, symmetrise, Frobenius, APC."""
    w = w - w.mean(axis=1, keepdims=True)
    w = w - w.mean(axis=3, keepdims=True)
    w = 0.5 * (w + np.transpose(w, (2, 3, 0, 1)))
    s = np.sqrt((w ** 2).sum(axis=(1, 3)))
    np.fill_diagonal(s, 0.0)
    return mrf.apc(s)


def synth_msa(n=1500, length=40, n_couple=10, seed=0):
    rng = np.random.default_rng(seed)
    aa = list(permute.AA20)
    msa = [list(rng.choice(aa, size=length)) for _ in range(n)]
    pairs = []
    while len(pairs) < n_couple:
        i, j = sorted(rng.choice(length, 2, replace=False))
        if j - i >= 12 and (i, j) not in pairs:
            pairs.append((i, j))
    for i, j in pairs:
        a, b = rng.choice(aa, 2)
        for row in msa:
            if rng.random() < 0.75:
                row[i], row[j] = a, b
    return ["".join(r) for r in msa], pairs


def main():
    msa, _ = synth_msa()
    length = len(msa[0])
    n = len(msa)
    rng = np.random.default_rng(7)
    perm = permute.random_perm(20, rng)
    pmsa = permute.apply_perm_msa(msa, perm)
    print("theory.md condition tests   N=%d L=%d  m=20 random relabeling\n"
          % (n, length))

    lam = 4.5 / np.sqrt(n)

    # ---- (i-a) POSITIVE: isotropic ridge, invariance must hold ---------------
    iso = np.full(length * A, lam)
    s0 = score_from_w(couplings_ridge(msa, iso))
    s1 = score_from_w(couplings_ridge(pmsa, iso))
    d_iso = float(np.abs(s0 - s1).max())
    scale = float(np.abs(s0).max())
    print("(i-a) isotropic ridge lam=%.5f" % lam)
    print("      max|dS| = %.3e   (score scale %.3f)   -> %s"
          % (d_iso, scale, "INVARIANT" if d_iso < 1e-9 else "BROKEN"))

    # ---- (i-b) NEGATIVE: anisotropic ridge, invariance must BREAK -----------
    # Ridge varying across the 20 alphabet slots, identical across positions. This is the
    # minimal violation: still diagonal, still positive, but no longer commuting with P_pi.
    per_aa = lam * (1.0 + 0.5 * np.arange(A) / (A - 1.0))
    aniso = np.tile(per_aa, length)
    s0a = score_from_w(couplings_ridge(msa, aniso))
    s1a = score_from_w(couplings_ridge(pmsa, aniso))
    d_aniso = float(np.abs(s0a - s1a).max())
    rel = d_aniso / max(scale, 1e-12)
    print("\n(i-b) anisotropic ridge, per-amino-acid, spread 1.0x-1.5x lam")
    print("      max|dS| = %.3e   (%.2f%% of score scale)  -> %s"
          % (d_aniso, 100 * rel, "BROKEN, as the proof requires"
             if d_aniso > 1e6 * max(d_iso, 1e-16) else "NOT BROKEN -- PROOF OVERSTATES"))

    # Does it change the reported quantity, or only the raw scores?
    def top(s):
        cand = sorted(((s[i, j], i, j) for i in range(length)
                       for j in range(i + 12, length)), reverse=True)
        return set((i, j) for _, i, j in cand[:max(1, length // 2)])
    ov = len(top(s0a) & top(s1a)) / float(len(top(s0a)))
    print("      top-L/2 contact-set overlap under the anisotropic ridge: %.3f" % ov)
    print("      (isotropic gives 1.000 by Proposition 2)")

    # ---- (ii) NEGATIVE: permuting non-standard characters -------------------
    # Inject gaps and X, then relabel WITHOUT the PROTECTED guard.
    rng2 = np.random.default_rng(11)
    dirty = []
    for s in msa:
        row = list(s)
        for k in range(len(row)):
            r = rng2.random()
            if r < 0.10:
                row[k] = "-"
            elif r < 0.115:
                row[k] = "X"
        dirty.append("".join(row))
    guarded = permute.apply_perm_msa(dirty, perm)
    # NOTE: swapping '-' with 'X' is INVISIBLE here -- mrf.ALPHABET is AA20 + GAP and
    # one_hot_msa folds every non-standard character into the gap state, so a permutation
    # WITHIN the protected set cannot be seen by the encoder at all. The relabeling the
    # guard actually forbids is protected <-> standard, which moves probability mass
    # between the gap slot and a real residue slot.
    bad = dict(perm)
    bad["-"] = "K"                          # gap -> a real amino acid
    unguarded = ["".join(bad.get(c, c) for c in s) for s in dirty]

    sd = score_from_w(couplings_ridge(dirty, iso))
    sg = score_from_w(couplings_ridge(guarded, iso))
    su = score_from_w(couplings_ridge(unguarded, iso))
    print("\n(ii) non-standard chars: 10% gaps + 1.5% X; unguarded maps gap -> K")
    print("      guarded   max|dS| = %.3e  -> %s"
          % (float(np.abs(sd - sg).max()),
             "INVARIANT" if np.abs(sd - sg).max() < 1e-9 else "BROKEN"))
    print("      unguarded max|dS| = %.3e  -> %s"
          % (float(np.abs(sd - su).max()),
             "BROKEN, as the proof requires"
             if np.abs(sd - su).max() > 1e-6 else "NOT BROKEN -- GUARD IS DECORATIVE"))

    ok = (d_iso < 1e-9 and d_aniso > 1e6 * max(d_iso, 1e-16)
          and np.abs(sd - sg).max() < 1e-9
          and np.abs(sd - su).max() > 1e-6)
    print("\nRESULT: %s" % ("all stated conditions are LOAD-BEARING and verified"
                            if ok else "at least one condition is NOT as theory.md states"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

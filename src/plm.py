"""Asymmetric pseudolikelihood Potts (plmDCA-style), for C13.

WHY THIS EXISTS. Every Potts number elsewhere in this project comes from `mrf.py`'s mean-field
/ inverse-covariance estimator, which Proposition 2 covers *exactly* and in closed form. But
the classical models a reviewer actually pictures -- plmDCA, GREMLIN, CCMpred -- are
pseudolikelihood fits driven by a numerical optimiser, and Proposition 1 (which covers them) is
only an optimality statement: it says the relabelled parameters ARE a maximiser, not that a
given solver finds it. This module exists so that claim can be tested rather than asserted.

The parameterisation, gauge and scoring deliberately mirror `mrf.py` -- same 21-state alphabet
(20 residues + gap), same non-standard-character folding, same APC -- so that the only thing
differing between the two estimators is the fitting principle.

Asymmetric plmDCA: for each column i independently, fit a multinomial logistic regression
predicting sigma_i from the one-hot encoding of every other column. Then symmetrise the
couplings. This is the standard formulation (Ekeberg et al. 2013) and is embarrassingly
parallel across columns.
"""
import numpy as np
from scipy import optimize

import mrf

A = mrf.A                       # 21 states, from mrf so the two estimators cannot drift apart


def _neg_pll_and_grad(theta, x_flat, y, lam, length):
    """Negative regularised pseudo-log-likelihood for ONE column, and its gradient.

    theta packs [h (A,), J (length*A, A)] for the target column. The target column's own block
    is zeroed after every unpack, so a column never couples to itself.
    """
    n = x_flat.shape[0]
    h = theta[:A]
    j = theta[A:].reshape(length * A, A)
    logits = x_flat.dot(j) + h                      # (n, A)
    logits -= logits.max(axis=1, keepdims=True)     # stabilise
    expl = np.exp(logits)
    probs = expl / expl.sum(axis=1, keepdims=True)

    ll = -np.log(np.maximum(probs[np.arange(n), y], 1e-300)).sum()
    resid = probs.copy()
    resid[np.arange(n), y] -= 1.0                   # (n, A)

    g_h = resid.sum(axis=0)
    g_j = x_flat.T.dot(resid)                       # (length*A, A)

    # Isotropic L2 -- isotropy is load-bearing for the invariance (theory.md).
    #
    # BUG FIXED 2026-08-05, found by C13's registered correctness gate. The penalty was
    # previously added to the SUMMED negative log-likelihood and the whole thing then divided
    # by n, making the effective coefficient lam/n -- 3.3e-6 at n=3000, i.e. essentially no
    # regularisation at all. That produced both of C13's first-run symptoms from one cause:
    # an unregularised fit with 29,568 parameters per column overfits (precision fell below
    # mean-field, failing the gate) and is ill-conditioned (invariance was 1e-2 relative
    # instead of near-machine-precision). The synthetic test missed it because L=12 with a
    # strong planted signal is well-determined with or without a penalty.
    #
    # `lam` now multiplies the penalty against the MEAN log-likelihood, which is the usual
    # plmDCA convention and makes lam directly interpretable and n-independent.
    return (ll / n + lam * (j * j).sum(),
            np.concatenate([g_h / n, (g_j / n + 2.0 * lam * j).ravel()]))


def fit(msa, lam=0.01, init="zero", rng=None, maxiter=200, verbose=False):
    """Fit couplings by asymmetric pseudolikelihood.

    Returns W with shape (L, A, L, A), symmetrised and in the zero-sum gauge.

    `init` is the whole point of C13:
      "zero"   -- a relabeling-EQUIVARIANT starting point. The optimisation trajectory of a
                  relabelled alignment is then the exact image of the original trajectory under
                  conjugation, so invariance should hold to floating-point noise whether or not
                  the solver converged.
      "random" -- drawn from `rng`, independently for each fit. Trajectories are unrelated, so
                  invariance can only hold up to convergence. The honest worst case.
    """
    x = mrf.one_hot_msa(msa)                        # (n, L, A)
    n, length, _ = x.shape
    x_flat = x.reshape(n, length * A)
    labels = x.argmax(axis=2)                       # (n, L)

    w = np.zeros((length, A, length, A))
    for i in range(length):
        xi = x_flat.copy()
        xi[:, i * A:(i + 1) * A] = 0.0              # never let column i predict itself
        y = labels[:, i]
        size = A + length * A * A
        if init == "zero":
            theta0 = np.zeros(size)
        elif init == "random":
            if rng is None:
                raise ValueError("init='random' needs an rng")
            theta0 = rng.normal(scale=0.01, size=size)
        else:
            raise ValueError("init must be 'zero' or 'random'")

        res = optimize.minimize(_neg_pll_and_grad, theta0, jac=True, method="L-BFGS-B",
                                args=(xi, y, lam, length),
                                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10})
        j = res.x[A:].reshape(length, A, A)
        j[i] = 0.0
        w[:, :, i, :] = j                           # J[k, a, i, b]: effect of col k on col i
        if verbose and (i + 1) % 20 == 0:
            print("      plm column %d/%d  nit=%d" % (i + 1, length, res.nit), flush=True)

    # Symmetrise: the two asymmetric estimates of the same pair are averaged.
    w = 0.5 * (w + w.transpose(2, 3, 0, 1))
    return _zero_sum_gauge(w)


def _zero_sum_gauge(w):
    """Project each 21x21 block onto the zero-sum gauge.

    Potts couplings are only identified up to a gauge; the zero-sum choice is standard and is
    also what makes the Frobenius norm a meaningful coupling strength. Note this is exactly the
    mean-centring step whose commutation with relabeling Proposition 2 relies on -- centring
    permutes the means rather than changing them.
    """
    w = w - w.mean(axis=1, keepdims=True)
    w = w - w.mean(axis=3, keepdims=True)
    return w


def contact_map(msa, lam=0.01, init="zero", rng=None, apply_apc=True, maxiter=200,
                verbose=False):
    """Frobenius-norm contact scores from a pseudolikelihood fit, with APC.

    Mirrors mrf.contact_map so the two estimators are compared on identical downstream code.
    """
    w = fit(msa, lam=lam, init=init, rng=rng, maxiter=maxiter, verbose=verbose)
    # Drop the gap state from the norm, matching the usual convention.
    scores = np.sqrt((w[:, :20, :, :20] ** 2).sum(axis=(1, 3)))
    np.fill_diagonal(scores, 0.0)
    return mrf.apc(scores) if apply_apc else scores

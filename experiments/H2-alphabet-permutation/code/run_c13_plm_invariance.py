"""C13 -- is a PSEUDOLIKELIHOOD Potts model invariant under alphabet relabeling?

Registered as protocol Amendment 13 before running. CPU-only by design.

theory.md flags this gap against itself: Proposition 2 covers our mean-field / inverse-
covariance estimator *exactly and in closed form*, but Proposition 1 -- which covers the
classical models a reviewer actually pictures (plmDCA, GREMLIN, CCMpred) -- is only an
optimality statement. It says the relabelled parameters ARE a maximiser; it says nothing about
what a given solver returns. Every Potts number in this project so far comes from the
closed-form estimator, so "the classical comparator is invariant" currently holds for an
estimator nobody in the field runs.

THE INITIALISATION CONTRAST IS THE POINT.
  zero init   -- a relabeling-EQUIVARIANT starting point, so the whole optimisation trajectory
                 of the relabelled problem is the exact image of the original under conjugation
                 by Q = I_L (x) P_pi. Invariance should hold to floating-point noise whether or
                 not the solver converged.
  random init -- drawn independently for the two fits, so the trajectories are unrelated and
                 invariance can only hold up to convergence. The honest worst case for the
                 "real pipelines use gradient descent" objection.

Reports both separately and never averages them.

Writes results/h2_c13_plm_invariance.csv.
"""
import os

# MUST precede numpy: BLAS reads these at load time. On this shared box (load average 35-55)
# the default thread pool is ~33x SLOWER than single-threaded -- 0.79 vs 26.1 Gflop/s measured.
# This is the difference between C13 taking ~25 minutes and ~13 hours per fit.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv          # noqa: E402
import sys          # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mrf          # noqa: E402
import permute      # noqa: E402
import plm          # noqa: E402
import structures   # noqa: E402
from run_c10_potts_crossing import (MSA_DIR, TARGETS, locate_region,  # noqa: E402
                                    read_a2m)

MIN_SEP = 12
TOP_FRAC = 0.5
# Selected by the Amendment 13a sweep on ubiquitin, against the UNPERMUTED fit's precision
# only -- the permuted fit was never run during selection and invariance was never consulted,
# because selecting on the DV would make C13 circular. All three candidates cleared the gate
# (mean-field 0.3333):  lam=0.001 -> 0.5152 | lam=0.01 -> 0.4848 | lam=0.1 -> 0.3636.
# The rule picks the best precision, which is also the LEAST regularised and so the
# worst-conditioned option -- i.e. the hardest case for invariance. That direction is
# deliberate: it cannot be accused of buying invariance with conditioning.
LAM = 0.001
MAXITER = 150
MAX_N = 3000          # subsample deep alignments; plm cost is linear in N
M = 20
SEED = 0

# Amendment 13 correctness gate: plm's UNPERMUTED precision must be at least as good as the
# mean-field estimator's on the same alignment. A pseudolikelihood fit that is worse than
# mean-field is broken, and the invariance of noise proves nothing.
GATE_SLACK = 0.02


def prepare(name, target, pdb_id, chain, a2m):
    """Same alignment/structure pipeline as C10, so the numbers are comparable."""
    seqs = read_a2m(os.path.join(MSA_DIR, a2m))
    query = seqs[0]
    loc = locate_region(query, target)
    if loc is None:
        return None
    q0, q1, t0, ident = loc
    sub = [s[q0:q1] for s in seqs]
    # Drop rows that are mostly gaps in the region -- standard, and identical for both fits.
    keep = [s for s in sub if s.count("-") / float(len(s)) <= 0.5]
    if len(keep) > MAX_N:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(keep), size=MAX_N, replace=False)
        keep = [keep[i] for i in sorted(idx)]

    path = structures.fetch_pdb(pdb_id)
    pdb_seq, coords = structures.ca_cb_coords(path, chain)
    cmap = structures.contact_map(coords)
    region_target = target[t0:t0 + (q1 - q0)]
    mapping, stats = structures.align_to_query(pdb_seq, region_target, return_stats=True)
    print("  %-14s region %d..%d of MSA query, ident=%.2f, depth %d, L=%d, pdb cov=%.0f%%"
          % (name, q0, q1, ident, len(keep), q1 - q0, 100 * stats["coverage"]), flush=True)
    return keep, cmap, mapping, q1 - q0


def precision(scores, cmap, mapping):
    p, n_eval = structures.precision_at_topk(scores, cmap, mapping, min_sep=MIN_SEP,
                                             top_frac=TOP_FRAC)
    return p, n_eval


def topset(scores, length):
    cand = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
    cand.sort(key=lambda p: -scores[p[0], p[1]])
    return set(cand[:max(1, int(TOP_FRAC * length))])


def main():
    out_path = os.path.join(HERE, "..", "results", "h2_c13_plm_invariance.csv")
    print("C13 pseudolikelihood Potts invariance  (threads pinned to %s)"
          % os.environ.get("OMP_NUM_THREADS"), flush=True)

    rng_perm = np.random.default_rng(1000 * SEED + M)
    perm = permute.PERM_BUILDERS["random"](M, rng_perm)
    print("relabeling: random, m=%d, realized %d\n"
          % (M, sum(1 for a, b in perm.items() if a != b)), flush=True)

    rows = []
    for name, target, pdb_id, chain, a2m in TARGETS:
        prep = prepare(name, target, pdb_id, chain, a2m)
        if prep is None:
            print("  %s: could not locate region, skipped" % name, flush=True)
            continue
        msa, cmap, mapping, length = prep
        pmsa = permute.apply_perm_msa(msa, perm)

        # ---- mean-field reference on the identical alignment (the gate's comparator) -------
        t0 = time.time()
        mf = mrf.contact_map(msa)
        mf_p, mf_n = precision(mf, cmap, mapping)
        print("    mean-field  P@L/2=%.4f (n_eval=%d)  [%.1fs]"
              % (mf_p, mf_n, time.time() - t0), flush=True)

        for init in ("zero", "random"):
            t0 = time.time()
            r1 = np.random.default_rng(101) if init == "random" else None
            r2 = np.random.default_rng(202) if init == "random" else None
            s_id = plm.contact_map(msa, lam=LAM, init=init, rng=r1, maxiter=MAXITER)
            s_pm = plm.contact_map(pmsa, lam=LAM, init=init, rng=r2, maxiter=MAXITER)
            el = time.time() - t0

            d_abs = float(np.abs(s_id - s_pm).max())
            scale = float(np.abs(s_id).max())
            d_rel = d_abs / scale if scale > 0 else float("nan")
            ov = len(topset(s_id, length) & topset(s_pm, length)) / float(
                max(1, int(TOP_FRAC * length)))
            p_id, n_eval = precision(s_id, cmap, mapping)
            p_pm, _ = precision(s_pm, cmap, mapping)

            gate = "PASS" if p_id >= mf_p - GATE_SLACK else "FAIL"
            print("    plm %-6s P@L/2=%.4f (mf %.4f, gate %s)  max|dS|=%.3e  rel=%.2e  "
                  "top-L/2 overlap=%.3f  permuted P=%.4f  [%.0fs]"
                  % (init, p_id, mf_p, gate, d_abs, d_rel, ov, p_pm, el), flush=True)

            rows.append(dict(protein=name, length=length, depth=len(msa), init=init,
                             mf_precision=round(mf_p, 5), plm_precision=round(p_id, 5),
                             plm_precision_permuted=round(p_pm, 5), gate=gate,
                             max_abs_dS=d_abs, score_scale=scale, rel_dS=d_rel,
                             topk_overlap=round(ov, 5), n_eval=n_eval, seconds=round(el, 1)))
            with open(out_path, "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print("", flush=True)

    # ---- verdict, printed last so it cannot be missed -----------------------------------
    print("=" * 72, flush=True)
    if not rows:
        print("no targets completed", flush=True)
        return 0
    if any(r["gate"] == "FAIL" for r in rows):
        print("CORRECTNESS GATE FAILED -- plm is worse than mean-field on some target.",
              flush=True)
        print("Per Amendment 13 C13 is WITHDRAWN: the invariance of a broken fit proves "
              "nothing.", flush=True)
        return 0
    print("correctness gate: PASS on all %d fits (plm >= mean-field - %.2f)"
          % (len(rows), GATE_SLACK), flush=True)
    for init in ("zero", "random"):
        sel = [r for r in rows if r["init"] == init]
        if not sel:
            continue
        print("  %-6s init:  max rel|dS| = %.2e   min top-L/2 overlap = %.3f"
              % (init, max(r["rel_dS"] for r in sel),
                 min(r["topk_overlap"] for r in sel)), flush=True)
    zero = [r for r in rows if r["init"] == "zero"]
    rand = [r for r in rows if r["init"] == "random"]
    if zero and rand:
        print("\nzero-init is %.0fx tighter than random-init in rel|dS| -- the trajectory of "
              "the\nrelabelled problem is the exact image of the original when the start point "
              "is\nequivariant, so invariance does not depend on convergence."
              % (max(r["rel_dS"] for r in rand) / max(1e-30, max(r["rel_dS"] for r in zero))),
              flush=True)
    print("\nwrote %s (%d rows)" % (out_path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

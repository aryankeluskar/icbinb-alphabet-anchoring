"""Amendment 13b diagnostic: does zero-init divergence grow with optimiser iterations?

Registered before running. Chaotic amplification of rounding predicts monotone growth
with maxiter; a structural bug predicts ~1e-3 already at 2 iterations.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np

ROOT = "/scratch/author/icbinb"
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/experiments/H2-alphabet-permutation/code")
import permute, plm
from run_c10_potts_crossing import MSA_DIR, TARGETS, locate_region, read_a2m
from run_c13_plm_invariance import prepare, topset, MIN_SEP, TOP_FRAC, LAM, M, SEED

name, target, pdb_id, chain, a2m = TARGETS[0]
prep = prepare(name, target, pdb_id, chain, a2m)
msa, cmap, mapping, length = prep
perm = permute.PERM_BUILDERS["random"](M, np.random.default_rng(1000 * SEED + M))
pmsa = permute.apply_perm_msa(msa, perm)
print("Amendment 13b diagnostic: %s L=%d depth=%d lam=%g zero init"
      % (name, length, len(msa), LAM), flush=True)
print("  %-8s %12s %12s %10s" % ("maxiter", "max|dS|", "rel|dS|", "top-L/2 ov"), flush=True)

for mx in (2, 10, 40):
    t0 = time.time()
    a = plm.contact_map(msa, lam=LAM, init="zero", maxiter=mx)
    b = plm.contact_map(pmsa, lam=LAM, init="zero", maxiter=mx)
    d = float(np.abs(a - b).max()); sc = float(np.abs(a).max())
    ov = len(topset(a, length) & topset(b, length)) / float(max(1, int(TOP_FRAC * length)))
    print("  %-8d %12.3e %12.3e %10.3f   [%.0fs]"
          % (mx, d, d / sc, ov, time.time() - t0), flush=True)
print("  %-8s %12.3e %12.3e %10.3f   (from the committed C13 run)"
      % (150, 3.808e-03, 1.88e-03, 1.000), flush=True)

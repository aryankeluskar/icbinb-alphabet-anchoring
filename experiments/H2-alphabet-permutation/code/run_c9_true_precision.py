"""C9 -- true long-range contact precision against experimental structure.

Registered as protocol Amendment 8 before running. This is the locked protocol's ORIGINAL
dependent variable, restored now that src/structures.py exists.

Self-consistency answers "does the model agree with itself under a transformation that
provably preserves every alignment statistic". It cannot tell a model that degrades from one
that was never right. This answers the other half: does the degradation cost real predictive
performance, measured against CB-CB < 8 Angstrom contacts from the deposited structure.

Only ubiquitin and protein G B1 are used -- both are 100% identical to their PDB chain.
Cytochrome c is deliberately excluded: our sequence is human, 1HRC is horse heart.

Writes results/h2_c9_true_precision.csv incrementally.
"""
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jacobian as jac_mod      # noqa: E402
import permute                  # noqa: E402
import structures               # noqa: E402
from run_h2_esm2 import EXTRA   # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
M_VALUES = [2, 10, 20]
REGIMES = ["random", "In-class"]
SEEDS = [0, 1]

# 100%-identity targets only (Amendment 8).
TARGETS = [("ubiquitin", "1UBQ", "A"), ("protein_G_B1", "1PGB", "A")]

# Registered prediction 1: these must reproduce or nothing else is reported.
EXPECTED = {"ubiquitin": 0.763, "protein_G_B1": 0.500}
GATE_TOL = 0.05


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(HERE, "..", "results", "h2_c9_true_precision.csv")
    print("C9 true precision  device=%s" % device, flush=True)

    prepared = []
    for name, pdb_id, chain in TARGETS:
        seq = EXTRA[name]
        path = structures.fetch_pdb(pdb_id)
        pdb_seq, coords = structures.ca_cb_coords(path, chain)
        cmap = structures.contact_map(coords)
        mapping, stats = structures.align_to_query(pdb_seq, seq, return_stats=True)
        n_lr = structures.n_long_range(cmap, min_sep=MIN_SEP)
        length = len(seq)
        n_pairs = sum(max(0, length - MIN_SEP - i) for i in range(length))
        chance = n_lr / float(n_pairs)
        print("  %-14s %s_%s  L=%d  cov=%.0f%% ident=%.0f%%  LR contacts=%d  chance=%.3f"
              % (name, pdb_id, chain, length, 100 * stats["coverage"],
                 100 * stats["identity"], n_lr, chance), flush=True)
        prepared.append((name, seq, cmap, mapping, chance))

    wrap = jac_mod.load_esm2("esm2_t33_650M_UR50D", device=device)
    print("loaded ESM-2 650M\n", flush=True)

    rows = []
    for name, seq, cmap, mapping, chance in prepared:
        def prec(s):
            scores = wrap.contact_map(s)
            p, n_eval = structures.precision_at_topk(
                scores, cmap, mapping, min_sep=MIN_SEP, top_frac=TOP_FRAC)
            return float(p), int(n_eval)

        p0, n_eval = prec(seq)
        gate = "PASS" if abs(p0 - EXPECTED[name]) <= GATE_TOL else "FAIL"
        print("%s  identity P@L/2 = %.3f  (expected %.3f, gate=%s)  n_eval=%d  "
              "chance=%.3f" % (name, p0, EXPECTED[name], gate, n_eval, chance),
              flush=True)
        if gate == "FAIL":
            # Registered: a pipeline that does not reproduce reports nothing.
            print("  !! GATE FAILED -- not reporting permuted cells for %s" % name,
                  flush=True)
            continue
        rows.append(dict(protein=name, regime="identity", m=0, realized_m=0, seed=0,
                         precision=round(p0, 5), delta=0.0, chance=round(chance, 5),
                         n_eval=n_eval, gate=gate))

        for regime in REGIMES:
            for m in M_VALUES:
                ps, real = [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    rm = sum(1 for a, b in perm.items() if a != b)
                    p, ne = prec(permute.apply_perm(seq, perm))
                    ps.append(p)
                    real.append(rm)
                    rows.append(dict(protein=name, regime=regime, m=m, realized_m=rm,
                                     seed=seed, precision=round(p, 5),
                                     delta=round(p - p0, 5), chance=round(chance, 5),
                                     n_eval=ne, gate=""))
                print("   %-13s m=%-3d (realized %.1f)  P@L/2=%.3f+-%.3f   "
                      "delta=%+.3f   x chance=%.1f"
                      % (regime, m, np.mean(real), np.mean(ps), np.std(ps),
                         np.mean(ps) - p0, np.mean(ps) / chance), flush=True)

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

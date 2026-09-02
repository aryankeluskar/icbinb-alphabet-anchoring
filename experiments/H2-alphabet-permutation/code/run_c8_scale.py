"""C8 -- the scale sweep. Registered as protocol Amendment 6 before running.

Everything is held fixed except parameter count: same proteins, same permutations, same
seeds, same DV. Each model is compared ONLY TO ITSELF, so the fact that bigger models
predict contacts better does not confound the invariance measurement.

650M is re-measured here rather than spliced in from run_h2_esm2.py, so every cell in the
published table comes from one internally consistent sweep.

Writes results/h2_c8_scale.csv incrementally and flushes after every (model, protein), so
an OOM or preemption at 3B still leaves a complete 8M-650M table.
"""
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jacobian as jac_mod      # noqa: E402
import permute                  # noqa: E402
from run_h2_esm2 import EXTRA, load_pfam_queries, spearman   # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
M_VALUES = [2, 6, 10, 20]
REGIMES = ["random", "In-class"]     # the two that carry the claim (Amendment 6)
SEEDS = [0, 1]

# Ascending size, so the cheap end of the table is complete before the expensive end
# is attempted. Batch size shrinks for 3B to fit alongside the other jobs on the GPU.
MODELS = [
    ("esm2_t6_8M_UR50D", 64),
    ("esm2_t12_35M_UR50D", 64),
    ("esm2_t30_150M_UR50D", 64),
    ("esm2_t33_650M_UR50D", 32),
    ("esm2_t36_3B_UR50D", 8),
]

# A fixed 4-protein subset. Kept small because cost is
# (n_models x n_proteins x n_conditions x L x 20) forward passes.
SUBSET = ["ubiquitin", "protein_G_B1", "cytochrome_c"]


def pick_proteins():
    proteins = {k: v for k, v in EXTRA.items() if k in SUBSET}
    pfam = load_pfam_queries(os.path.join(ROOT, "data", "pfam"))
    # One Pfam query for family diversity: the shortest available, to bound cost.
    if pfam:
        acc = min(pfam, key=lambda k: len(pfam[k]))
        proteins[acc] = pfam[acc]
    return proteins


def sweep_model(model_name, batch_size, proteins, device, rows, out_path):
    t0 = time.time()
    import esm
    model, alphabet = getattr(esm.pretrained, model_name)()
    wrap = jac_mod.ESM2Jacobian(model, alphabet, device=device, batch_size=batch_size)
    print("\n=== %s  (loaded in %.0fs, batch=%d) ==="
          % (model_name, time.time() - t0, batch_size), flush=True)

    for pid, seq in sorted(proteins.items()):
        length = len(seq)
        t1 = time.time()
        base_scores = wrap.contact_map(seq)
        base_top = jac_mod.top_contacts(base_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        iu = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
        base_vec = np.array([base_scores[i, j] for i, j in iu])
        chance = len(base_top) / float(len(iu))
        print("  %s L=%d baseline %.1fs (chance %.3f)"
              % (pid, length, time.time() - t1, chance), flush=True)

        for regime in REGIMES:
            for m in M_VALUES:
                ovs, sps = [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    pscores = wrap.contact_map(permute.apply_perm(seq, perm))
                    ov = jac_mod.set_overlap(
                        base_top,
                        jac_mod.top_contacts(pscores, min_sep=MIN_SEP,
                                             top_frac=TOP_FRAC))
                    sp = spearman(base_vec,
                                  np.array([pscores[i, j] for i, j in iu]))
                    ovs.append(ov)
                    sps.append(sp)
                    rows.append(dict(model=model_name, protein=pid, length=length,
                                     regime=regime, m=m, seed=seed,
                                     overlap=round(ov, 5), spearman=round(sp, 5),
                                     chance=round(chance, 5)))
                print("    %-13s m=%-3d overlap=%.3f+-%.3f  spearman=%.3f"
                      % (regime, m, np.mean(ovs), np.std(ovs), np.mean(sps)), flush=True)

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    del wrap, model
    torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proteins = pick_proteins()
    out_path = os.path.join(HERE, "..", "results", "h2_c8_scale.csv")

    print("C8 scale sweep  device=%s  models=%d  proteins=%d"
          % (device, len(MODELS), len(proteins)), flush=True)
    for k, v in sorted(proteins.items()):
        print("   %-28s L=%d" % (k, len(v)), flush=True)

    rows = []
    done = []
    for model_name, bs in MODELS:
        try:
            sweep_model(model_name, bs, proteins, device, rows, out_path)
            done.append(model_name)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # Registered budget guard: a failure at the expensive end must not discard
            # the cheap end. Report the omission rather than leaving a silent gap.
            print("\n!! %s FAILED: %s" % (model_name, exc), flush=True)
            print("!! reporting sweep as %s only" % ", ".join(done), flush=True)
            torch.cuda.empty_cache()
            break

    print("\nwrote %s (%d rows) covering %s"
          % (out_path, len(rows), ", ".join(done)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

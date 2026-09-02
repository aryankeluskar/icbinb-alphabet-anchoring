"""C11 -- a third architecture. Registered as protocol Amendment 11 before running.

ESM-2 (C8, five sizes) and MSA Transformer (C7) are both Meta models trained on UniRef with
the same 33-token vocabulary. A reviewer can read that as one lineage rather than as evidence
about pLMs in general. AMPLIFY 120M is a different lineage: SwiGLU feed-forward, RoPE,
RMSNorm, a 27-token vocabulary in a different residue order, and a different corpus.

The comparison point is ESM-2 150M from C8 -- 118M vs 148M parameters is the closest matched
pair available, so architecture and corpus vary while capacity roughly does not.

The Jacobian reduction is not reimplemented. `AMPLIFYJacobian` subclasses `ESM2Jacobian` and
overrides only `_tokenize` and `_logits`, so mean-centring, symmetrisation, per-block Frobenius
and APC are the same code that produced every ESM-2 number. A cross-architecture comparison is
worth nothing if the readout differs.

VALIDITY GATE (Amendment 11, fixed before the run). Readout 2 established the expensive way
that a degradation statistic is meaningless on a unit with no signal to destroy. Before any
permuted cell is reported, AMPLIFY's *unpermuted* long-range P@L/2 against 1UBQ_A and 1PGB_A
is measured. If it is below 3x chance on BOTH targets the sweep is declared uninformative and
withdrawn. The gate is free: it reuses the same baseline contact map the sweep needs anyway.

Writes results/h2_c11_amplify.csv incrementally, flushed after every protein.
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

import amplify_jac               # noqa: E402
import jacobian as jac_mod       # noqa: E402
import permute                   # noqa: E402
import structures                # noqa: E402
from run_h2_esm2 import EXTRA, spearman   # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
M_VALUES = [2, 6, 10, 20]
REGIMES = ["random", "In-class"]
SEEDS = [0, 1]
BATCH = 16                       # small: the GPU is shared with the C8 sweep

SUBSET = ["ubiquitin", "protein_G_B1", "cytochrome_c"]

# The two 100%-identity targets from C9. Cytochrome c is excluded from the gate for the same
# reason it was excluded there: our sequence is human, 1HRC is horse heart.
GATE_TARGETS = {"ubiquitin": ("1UBQ", "A"), "protein_G_B1": ("1PGB", "A")}
GATE_RATIO = 3.0                 # Amendment 11, fixed before the run


def load_gate_structures():
    out = {}
    for name, (pdb_id, chain) in sorted(GATE_TARGETS.items()):
        path = structures.fetch_pdb(pdb_id)
        pdb_seq, coords = structures.ca_cb_coords(path, chain)
        cmap = structures.contact_map(coords)
        mapping, stats = structures.align_to_query(pdb_seq, EXTRA[name], return_stats=True)
        out[name] = (cmap, mapping)
        print("  gate target %-14s %s_%s cov=%.0f%% ident=%.0f%%"
              % (name, pdb_id, chain, 100 * stats["coverage"], 100 * stats["identity"]),
              flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(HERE, "..", "results", "h2_c11_amplify.csv")
    print("C11 AMPLIFY 120M  device=%s  batch=%d" % (device, BATCH), flush=True)

    gate_structs = load_gate_structures()

    t0 = time.time()
    wrap = amplify_jac.load_amplify(year=2024, device=device, batch_size=BATCH)
    print("loaded AMPLIFY_120M_2024 in %.0fs  params=%.1fM\n"
          % (time.time() - t0, wrap.n_params / 1e6), flush=True)

    rows = []
    gate_results = {}

    for pid in SUBSET:
        seq = EXTRA[pid]
        length = len(seq)
        t1 = time.time()
        base_scores = wrap.contact_map(seq)
        base_top = jac_mod.top_contacts(base_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        iu = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
        base_vec = np.array([base_scores[i, j] for i, j in iu])
        overlap_chance = len(base_top) / float(len(iu))

        # --- validity gate, sharing the baseline Jacobian just computed -------------------
        gate_str = ""
        if pid in gate_structs:
            cmap, mapping = gate_structs[pid]
            prec, n_eval = structures.precision_at_topk(
                base_scores, cmap, mapping, min_sep=MIN_SEP, top_frac=TOP_FRAC)
            n_lr = structures.n_long_range(cmap, min_sep=MIN_SEP)
            n_pairs = sum(max(0, length - MIN_SEP - i) for i in range(length))
            struct_chance = n_lr / float(n_pairs)
            ratio = prec / struct_chance if struct_chance > 0 else 0.0
            gate_results[pid] = (prec, struct_chance, ratio)
            gate_str = "  GATE P@L/2=%.3f vs chance %.3f = %.1fx" % (prec, struct_chance, ratio)
            rows.append(dict(model="AMPLIFY_120M_2024", protein=pid, length=length,
                             regime="identity", m=0, seed=-1,
                             overlap=1.0, spearman=1.0, chance=round(overlap_chance, 5),
                             true_precision=round(prec, 5),
                             precision_chance=round(struct_chance, 5)))

        print("  %-14s L=%d baseline %.1fs (overlap chance %.3f)%s"
              % (pid, length, time.time() - t1, overlap_chance, gate_str), flush=True)

        for regime in REGIMES:
            for m in M_VALUES:
                ovs, sps = [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    pscores = wrap.contact_map(permute.apply_perm(seq, perm))
                    ov = jac_mod.set_overlap(
                        base_top,
                        jac_mod.top_contacts(pscores, min_sep=MIN_SEP, top_frac=TOP_FRAC))
                    sp = spearman(base_vec, np.array([pscores[i, j] for i, j in iu]))
                    ovs.append(ov)
                    sps.append(sp)
                    rows.append(dict(model="AMPLIFY_120M_2024", protein=pid, length=length,
                                     regime=regime, m=m, seed=seed,
                                     overlap=round(ov, 5), spearman=round(sp, 5),
                                     chance=round(overlap_chance, 5),
                                     true_precision="", precision_chance=""))
                print("    %-13s m=%-3d overlap=%.3f+-%.3f  spearman=%.3f"
                      % (regime, m, np.mean(ovs), np.std(ovs), np.mean(sps)), flush=True)

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # --- report the gate verdict LAST, so it is impossible to miss ------------------------
    print("\n" + "=" * 72, flush=True)
    passed = [p for p, (_, _, r) in gate_results.items() if r >= GATE_RATIO]
    for p, (prec, ch, r) in sorted(gate_results.items()):
        print("GATE %-14s P@L/2=%.3f  chance=%.3f  ratio=%.1fx  %s"
              % (p, prec, ch, r, "PASS" if r >= GATE_RATIO else "fail"), flush=True)
    if passed:
        print("GATE VERDICT: PASS on %s -- the permuted cells above are reportable."
              % ", ".join(sorted(passed)), flush=True)
    else:
        print("GATE VERDICT: FAILED on both targets (< %.0fx chance)." % GATE_RATIO, flush=True)
        print("Per Amendment 11 the C11 overlap sweep is WITHDRAWN as uninformative:", flush=True)
        print("a model that cannot predict contacts has no contact prediction to destroy.",
              flush=True)
        print("Do NOT substitute a different checkpoint to obtain a usable gate.", flush=True)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

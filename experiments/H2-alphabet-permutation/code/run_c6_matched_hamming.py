"""C6 -- the control that can refute H2.

A bijective relabeling with m=2 still changes ~10% of residue positions. So the pilot
result (overlap 0.525 at m=2) is equally consistent with "ESM-2 is sensitive to sequence
edits" as with "ESM-2 depends on literal residue identity". Those are very different
claims and only one of them is interesting.

C6 holds the AMOUNT of change fixed and varies only its STRUCTURE:

    permuted   : bijective relabeling pi, applied consistently to every position
    matched    : random substitutions at EXACTLY the same number of positions

If the two degrade equally, H2's identity claim is not supported and we report that.

For each (regime, m, seed) we build the permuted sequence, count its Hamming distance d
from wild-type, then build a random-substitution control at the same d. To make the
comparison as tight as possible the control mutates the SAME POSITIONS -- so the two
sequences differ from wild-type in identical places, and differ from each other only in
what they put there.

Two flavours of random control, because they answer slightly different questions:
  uniform      -- replacement drawn uniformly from the other 19 residues
  composition  -- replacement drawn from the wild-type's own residue composition, which
                  also preserves the amino-acid frequency profile the way a permutation
                  approximately does

Outputs results/h2_c6_matched_hamming.csv.
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import jacobian as jac_mod      # noqa: E402
import permute                  # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
M_VALUES = [2, 6, 10, 20]
REGIMES = ["random", "In-class", "Cross-class"]
SEEDS = [0, 1, 2]

# A subset of the main run's proteins -- enough for a paired comparison without
# doubling the whole grid's cost.
PROTEINS = {
    "ubiquitin": ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
                  "KESTLHLVLRLRGG"),
    "protein_G_B1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "cytochrome_c": ("GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGED"
                     "TLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE"),
}


def hamming_positions(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def random_substitute(seq, positions, rng, mode="uniform"):
    """Substitute at exactly `positions`, guaranteeing each one actually changes."""
    out = list(seq)
    if mode == "composition":
        pool = [c for c in seq if c in permute.AA20]
    else:
        pool = list(permute.AA20)
    for i in positions:
        choices = [c for c in pool if c != seq[i]]
        if not choices:
            choices = [c for c in permute.AA20 if c != seq[i]]
        out[i] = choices[rng.integers(len(choices))]
    return "".join(out)


def spearman(a, b):
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")
    out_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "results", "h2_c6_matched_hamming.csv"))

    print("C6 matched-Hamming control  model=%s device=%s" % (model_name, device))
    wrap = jac_mod.load_esm2(model_name, device=device)

    rows = []
    for pid, seq in sorted(PROTEINS.items()):
        length = len(seq)
        base_scores = wrap.contact_map(seq)
        base_top = jac_mod.top_contacts(base_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        iu = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
        base_vec = np.array([base_scores[i, j] for i, j in iu])
        print("\n%s L=%d" % (pid, length))

        def evaluate(s):
            sc = wrap.contact_map(s)
            top = jac_mod.top_contacts(sc, min_sep=MIN_SEP, top_frac=TOP_FRAC)
            return (jac_mod.set_overlap(base_top, top),
                    spearman(base_vec, np.array([sc[i, j] for i, j in iu])))

        for regime in REGIMES:
            for m in M_VALUES:
                seeds = SEEDS if not (regime == "Cross-class" and m == 20) else SEEDS[:1]
                agg = {"permuted": [], "uniform": [], "composition": []}
                for seed in seeds:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    pseq = permute.apply_perm(seq, perm)
                    pos = hamming_positions(seq, pseq)
                    d = len(pos)
                    if d == 0:
                        continue

                    ov_p, sp_p = evaluate(pseq)
                    agg["permuted"].append(ov_p)
                    rows.append(dict(protein=pid, length=length, regime=regime, m=m,
                                     seed=seed, arm="permuted", hamming=d,
                                     hamming_frac=d / float(length),
                                     overlap=ov_p, spearman=sp_p))

                    for mode in ("uniform", "composition"):
                        rseq = random_substitute(
                            seq, pos, np.random.default_rng(7000 + 1000 * seed + m), mode)
                        # Same positions changed, same count -- only the content differs.
                        assert len(hamming_positions(seq, rseq)) == d
                        ov_r, sp_r = evaluate(rseq)
                        agg[mode].append(ov_r)
                        rows.append(dict(protein=pid, length=length, regime=regime, m=m,
                                         seed=seed, arm=mode, hamming=d,
                                         hamming_frac=d / float(length),
                                         overlap=ov_r, spearman=sp_r))

                if agg["permuted"]:
                    print("   %-13s m=%-3d d=%-4d perm=%.3f  unif=%.3f  comp=%.3f"
                          % (regime, m, rows[-1]["hamming"],
                             np.mean(agg["permuted"]), np.mean(agg["uniform"]),
                             np.mean(agg["composition"])))

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

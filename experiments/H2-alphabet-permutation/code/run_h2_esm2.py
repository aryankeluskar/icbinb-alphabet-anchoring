"""H2, the actual measurement: is ESM-2's categorical Jacobian invariant to a consistent
alphabet relabeling?

The Potts side is settled: exactly invariant, verified to 1e-15 on synthetic, real Pfam,
and adversarially corrupted alignments (see analysis.md). This script asks the same
question of ESM-2.

DESIGN NOTE -- why this needs no structures.

The protocol's original DV was contact precision against experimental structures. That
requires PDB files we do not have yet. But invariance is a SELF-CONSISTENCY property, so it
can be measured without any ground truth at all:

    permute the sequence, recompute the Jacobian contact map, and ask how much the model's
    OWN prediction moved.

Positions are untouched by an alphabet relabeling, so the two maps live in the same
coordinate space and are directly comparable. The Potts reference value is exactly 1.000
overlap by construction. This is a cleaner measurement than precision, because it is immune
to the objection that the permuted sequence is "harder" -- we are not asking whether ESM-2
is right, only whether it is CONSISTENT with itself under a transformation that provably
preserves all alignment statistics.

Accuracy-against-structure remains worth running later; it answers a different question
(does the degradation cost real predictive performance) and is reported separately.

Outputs results/h2_esm2_invariance.csv.
"""
import csv
import os
import sys
import time

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


def load_pfam_queries(pfam_dir, max_len=140, min_len=50):
    """Ungapped query sequences from the Pfam seed alignments already on disk."""
    seqs = {}
    if not os.path.isdir(pfam_dir):
        return seqs
    for fn in sorted(os.listdir(pfam_dir)):
        if not fn.endswith(".sto"):
            continue
        acc = fn[:-4]
        rows = {}
        order = []
        with open(os.path.join(pfam_dir, fn)) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                name, chunk = parts
                if name not in rows:
                    rows[name] = []
                    order.append(name)
                rows[name].append(chunk.strip())
        for name in order:
            raw = "".join(rows[name])
            seq = "".join(c for c in raw.upper()
                          if c in permute.AA20)          # drop gaps/inserts/non-standard
            if min_len <= len(seq) <= max_len:
                seqs["%s|%s" % (acc, name.split("/")[0])] = seq
                break
    return seqs


EXTRA = {
    "ubiquitin": ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
                  "KESTLHLVLRLRGG"),
    "protein_G_B1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "lysozyme_T4": ("MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDE"
                    "AEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQK"
                    "RWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"),
    "cytochrome_c": ("GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGED"
                     "TLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE"),
}


def spearman(a, b):
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")
    here = os.path.dirname(__file__)
    pfam_dir = os.path.abspath(os.path.join(here, "..", "..", "..", "data", "pfam"))
    out_path = os.path.abspath(os.path.join(here, "..", "results",
                                            "h2_esm2_invariance.csv"))

    proteins = dict(EXTRA)
    proteins.update(load_pfam_queries(pfam_dir))

    print("model=%s device=%s  proteins=%d" % (model_name, device, len(proteins)))
    for k, v in sorted(proteins.items()):
        print("   %-28s L=%d" % (k, len(v)))

    wrap = jac_mod.load_esm2(model_name, device=device)

    rows = []
    for pid, seq in sorted(proteins.items()):
        length = len(seq)
        t0 = time.time()
        base_scores = wrap.contact_map(seq)
        base_top = jac_mod.top_contacts(base_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        iu = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
        base_vec = np.array([base_scores[i, j] for i, j in iu])
        n_pairs = len(iu)
        chance = len(base_top) / float(n_pairs)

        print("\n%s  L=%d  baseline %.1fs  (chance overlap %.3f)"
              % (pid, length, time.time() - t0, chance))

        # C1: identity permutation must reproduce the baseline exactly.
        ident = permute.apply_perm(seq, permute.identity_perm())
        assert ident == seq, "identity permutation altered the sequence"
        rows.append(dict(protein=pid, length=length, model=model_name, regime="identity",
                         m=0, seed=0, overlap=1.0, spearman=1.0, chance=chance,
                         n_pairs=n_pairs))

        for regime in REGIMES:
            for m in M_VALUES:
                ovs, sps = [], []
                # m=20 Cross-class is deterministic; extra seeds would be wasted compute.
                seeds = SEEDS if not (regime == "Cross-class" and m == 20) else SEEDS[:1]
                for seed in seeds:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    pseq = permute.apply_perm(seq, perm)
                    pscores = wrap.contact_map(pseq)
                    ptop = jac_mod.top_contacts(pscores, min_sep=MIN_SEP,
                                                top_frac=TOP_FRAC)
                    ov = jac_mod.set_overlap(base_top, ptop)
                    sp = spearman(base_vec,
                                  np.array([pscores[i, j] for i, j in iu]))
                    ovs.append(ov)
                    sps.append(sp)
                    rows.append(dict(protein=pid, length=length, model=model_name,
                                     regime=regime, m=m, seed=seed, overlap=ov,
                                     spearman=sp, chance=chance, n_pairs=n_pairs))
                print("   %-13s m=%-3d overlap=%.3f+-%.3f  spearman=%.3f"
                      % (regime, m, np.mean(ovs), np.std(ovs), np.mean(sps)))

        # Write incrementally so a crash or preemption does not lose everything.
        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

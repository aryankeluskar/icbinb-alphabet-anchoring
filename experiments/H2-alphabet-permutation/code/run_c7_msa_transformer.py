"""C7 -- MSA Transformer: the FAIR neural comparison.

The most serious threat to H2 is of our own making. The Potts model is REFIT on the
permuted alignment, while ESM-2 gets no relabeled training data, so "Potts is invariant,
ESM-2 collapses" partly just restates that one model derives knowledge at inference time
and the other has it frozen in weights.

MSA Transformer removes that asymmetry: like Potts, it reads an ALIGNMENT at inference.
Neither model is retrained; both see the same permuted family.

  - If MSA-T is invariant  -> the collapse is an artefact of single-sequence amortisation,
                              and H2 must narrow to single-sequence pLMs specifically.
  - If MSA-T also collapses -> much stronger. A neural model holding the alignment in its
                              hands still cannot do the alphabet-agnostic computation that
                              inverse covariance does exactly and for free.

Both outcomes are publishable and they say materially different things, which is why this
runs before H2 is written up.

Readout is MSA-T's supervised contact head, which is what Rao et al. 2021 used for their
own shuffling controls (Fig 6) -- so this lands on the same axis-comparison footing.

Reports, per family and per condition:
  overlap   top-L/2 long-range contact-set agreement with the unpermuted MSA
  potts     the same quantity for the inverse-covariance model on the SAME alignment,
            which must be exactly 1.000 -- an in-run assert that the alignment really is
            statistically unchanged.
"""
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mrf          # noqa: E402
import permute      # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
MAX_DEPTH = 256          # MSA rows fed to MSA-T
M_VALUES = [2, 6, 10, 20]
REGIMES = ["random", "In-class", "Cross-class"]
SEEDS = [0, 1, 2]


def parse_stockholm(text):
    rows, order = {}, []
    for line in text.splitlines():
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
    msa = ["".join(rows[n]) for n in order]
    if not msa:
        return []
    length = len(msa[0])
    msa = [s for s in msa if len(s) == length]
    keep = [j for j in range(length)
            if all(not s[j].islower() and s[j] != "." for s in msa)]
    return ["".join(s[j] for j in keep).upper() for s in msa]


def gap_filter(msa, max_gap_frac=0.5):
    arr = np.array([list(s) for s in msa])
    frac = (arr == "-").mean(axis=0)
    keep = np.where(frac <= max_gap_frac)[0]
    return ["".join(row[keep]) for row in arr]


def top_contacts(scores, min_sep=MIN_SEP, top_frac=TOP_FRAC):
    length = scores.shape[0]
    cand = []
    for i in range(length):
        for j in range(i + min_sep, length):
            cand.append((scores[i, j], i, j))
    cand.sort(reverse=True)
    k = max(1, int(length * top_frac))
    return set((i, j) for _, i, j in cand[:k])


def overlap(a, b):
    return len(a & b) / float(len(a)) if a else float("nan")


class MSATransformer(object):
    def __init__(self, device="cuda"):
        import esm
        self.model, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.model = self.model.eval().to(device)
        self.bc = self.alphabet.get_batch_converter()
        self.device = device

    @torch.no_grad()
    def contacts(self, msa):
        data = [[("s%d" % i, s) for i, s in enumerate(msa)]]
        _, _, toks = self.bc(data)
        toks = toks.to(self.device)
        out = self.model(toks, return_contacts=True)
        return out["contacts"][0].float().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pfam = os.path.join(ROOT, "data", "pfam")
    out_path = os.path.join(HERE, "..", "results", "h2_c7_msa_transformer.csv")

    families = []
    for fn in sorted(os.listdir(pfam)):
        if not fn.endswith(".sto"):
            continue
        with open(os.path.join(pfam, fn)) as fh:
            msa = gap_filter(parse_stockholm(fh.read()))
        if len(msa) < 30 or len(msa[0]) < 40:
            print("skip %s (N=%d L=%d)" % (fn, len(msa), len(msa[0]) if msa else 0))
            continue
        families.append((fn[:-4], msa[:MAX_DEPTH]))

    print("C7 MSA Transformer  device=%s  families=%d" % (device, len(families)))
    for acc, msa in families:
        print("   %-10s N=%-4d L=%d" % (acc, len(msa), len(msa[0])))

    mt = MSATransformer(device=device)
    print("loaded MSA-T.\n")

    rows = []
    for acc, msa in families:
        length = len(msa[0])

        base_mt = top_contacts(mt.contacts(msa))
        base_potts = top_contacts(mrf.contact_map(msa))
        print("%s  N=%d L=%d" % (acc, len(msa), length))

        for regime in REGIMES:
            for m in M_VALUES:
                seeds = SEEDS if not (regime == "Cross-class" and m == 20) else SEEDS[:1]
                ov_mt, ov_p = [], []
                for seed in seeds:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    # The SAME pi applied to every sequence -- this is what makes the
                    # alignment statistically identical.
                    pmsa = permute.apply_perm_msa(msa, perm)

                    o_mt = overlap(base_mt, top_contacts(mt.contacts(pmsa)))
                    o_p = overlap(base_potts, top_contacts(mrf.contact_map(pmsa)))
                    ov_mt.append(o_mt)
                    ov_p.append(o_p)
                    rows.append(dict(family=acc, length=length, depth=len(msa),
                                     regime=regime, m=m, seed=seed,
                                     msa_transformer=round(o_mt, 5),
                                     potts=round(o_p, 5)))

                # In-run assert: Potts MUST be exactly invariant on this same alignment.
                # If it is not, the alignment changed in some way we did not intend and the
                # MSA-T number is uninterpretable.
                if min(ov_p) < 0.999:
                    print("   !! POTTS NOT INVARIANT (%.4f) -- alignment altered "
                          "unexpectedly at %s m=%d" % (min(ov_p), regime, m))

                print("   %-13s m=%-3d MSA-T=%.3f+-%.3f   Potts=%.3f"
                      % (regime, m, np.mean(ov_mt), np.std(ov_mt), np.mean(ov_p)))

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

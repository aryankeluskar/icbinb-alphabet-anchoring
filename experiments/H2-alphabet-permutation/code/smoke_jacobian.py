"""Smoke test: is our categorical Jacobian implemented correctly?

We have no structures downloaded yet, so we cannot score against true contacts. But we do
not need them for a correctness check. ESM-2 ships a SUPERVISED contact head that is known
to work, and Zhang & Ovchinnikov's central claim is that the unsupervised categorical
Jacobian recovers substantially the same contacts. So:

    if our Jacobian is implemented correctly, its top-L/2 long-range contacts should
    overlap heavily with the supervised head's.

If the overlap is near chance, our Jacobian is wrong and nothing downstream is worth
running. This is a cheap gate on an expensive experiment -- the same role the MRF
invariance test played for the Potts side.

Also times a single Jacobian so we can budget the full run: cost is L x 20 forward passes.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import jacobian as jac_mod      # noqa: E402

# Ubiquitin (76 aa) and the B1 domain of protein G (56 aa): small, well-studied, and short
# enough that L x 20 forward passes is quick.
SEQS = {
    "ubiquitin": ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
                  "KESTLHLVLRLRGG"),
    "protein_G_B1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
}

MIN_SEP = 12          # these domains are short; |i-j|>=24 leaves too few pairs
TOP_FRAC = 0.5


def supervised_contacts(model_wrap, seq):
    """ESM-2's own supervised contact head -- the reference we validate against."""
    toks = model_wrap._tokenize(seq).unsqueeze(0).to(model_wrap.device)
    with torch.no_grad():
        out = model_wrap.model(toks, repr_layers=[], return_contacts=True)
    return out["contacts"][0].float().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")
    print("=" * 74)
    print("Smoke test: categorical Jacobian vs ESM-2 supervised contact head")
    print("=" * 74)
    print("model=%s  device=%s" % (name, device))

    wrap = jac_mod.load_esm2(name, device=device)
    print("loaded.\n")

    all_ok = True
    for label, seq in SEQS.items():
        length = len(seq)
        t0 = time.time()
        jac = wrap.jacobian(seq)
        jac_scores = wrap.contact_map(seq, jac=jac)
        dt = time.time() - t0

        sup_scores = supervised_contacts(wrap, seq)

        jac_top = jac_mod.top_contacts(jac_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        sup_top = jac_mod.top_contacts(sup_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        overlap = jac_mod.set_overlap(jac_top, sup_top)

        # Chance overlap: |top| / (number of eligible long-range pairs).
        n_pairs = sum(1 for i in range(length) for j in range(i + MIN_SEP, length))
        chance = len(jac_top) / float(n_pairs)

        # Rank correlation over all eligible pairs, as a second view.
        iu = [(i, j) for i in range(length) for j in range(i + MIN_SEP, length)]
        a = np.array([jac_scores[i, j] for i, j in iu])
        b = np.array([sup_scores[i, j] for i, j in iu])
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(ra, rb)[0, 1])

        ok = overlap > 4 * chance
        all_ok = all_ok and ok
        print("%-14s L=%-4d  jacobian %5.1fs  (%d fwd passes)"
              % (label, length, dt, length * 20))
        print("               top-L/2 overlap with supervised head = %.3f "
              "(chance %.3f, %.1fx)" % (overlap, chance, overlap / chance))
        print("               spearman over all long-range pairs   = %.3f" % spearman)
        print("               %s\n" % ("PASS" if ok else "FAIL -- Jacobian looks wrong"))

    print("=" * 74)
    print("VERDICT: %s" % ("Jacobian implementation validated" if all_ok
                           else "FAILURE -- do not run the H2 grid"))
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

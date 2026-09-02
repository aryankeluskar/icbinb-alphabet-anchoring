"""Control C2 + C3: does the MRF baseline behave as the H2 protocol claims?

H2 rests on a mathematical premise: a Potts/MRF coevolution model is EXACTLY invariant
under a consistent bijective relabeling of the amino-acid alphabet applied across a whole
MSA. If that premise is false -- or if our implementation breaks it -- the entire
experiment is void, because we would have no invariant reference against which to measure
ESM-2's degradation.

So before touching a GPU we verify, on synthetic alignments with PLANTED couplings:

  C0  recovery      MRF recovers the planted contacts well above chance.
  C2  invariance    A CONSISTENT permutation leaves the contact map exactly unchanged.
  C3  destruction   The true coevolution null (per-column row shuffle) destroys it.
  C3b observation   A per-SEQUENCE permutation only partially degrades it -- see below.

C3 matters as much as C2: without it, "invariance" could be an artefact of a metric that
cannot see anything at all.

C3b is a finding in its own right. Applying a different bijection to each SEQUENCE relabels
both columns of a row identically, and mutual information is invariant under a bijective
relabeling. So per-sequence permutation does NOT destroy coevolution -- it only blurs it by
mixing conjugated couplings across rows. The correct null is a per-COLUMN row shuffle, which
preserves marginals and destroys dependency. Getting this wrong would have made the pLM
comparison meaningless, since we would have been calling a partially-informative condition
'destroyed'.

Runs on stock Python 3.6 + numpy -- no GPU, no pLM, no downloads.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import mrf          # noqa: E402
import permute      # noqa: E402

AA20 = mrf.AA20


def synth_msa(n_seq, length, contacts, rng, coupling_strength=0.9):
    """Generate an MSA with planted pairwise couplings.

    Background columns are i.i.d. uniform over the 20 residues. For each planted contact
    (i, j) we couple column j to column i through a fixed random bijection, applied with
    probability `coupling_strength`; otherwise j is drawn independently. This produces
    genuine pairwise mutual information at exactly the planted positions and nowhere else,
    which is the cleanest possible test bed for a coevolution method.
    """
    aa = np.array(list(AA20))
    msa_arr = rng.choice(aa, size=(n_seq, length))
    for (i, j) in contacts:
        # A fixed bijection mapping the residue at i to the residue at j.
        partner = dict(zip(AA20, rng.permutation(list(AA20))))
        coupled = rng.random(n_seq) < coupling_strength
        for s in range(n_seq):
            if coupled[s]:
                msa_arr[s, j] = partner[msa_arr[s, i]]
    return ["".join(row) for row in msa_arr]


def main():
    rng = np.random.default_rng(0)

    n_seq, length = 2000, 60
    min_sep = 24
    # Planted long-range contacts, all with |i - j| >= min_sep so they are scored.
    # Plant enough contacts that precision is not ceilinged near zero: with top_frac=0.5
    # the precision ceiling is n_contacts / (L/2).
    contacts = [(2, 40), (5, 33), (10, 50), (14, 44), (0, 55), (7, 38),
                (3, 45), (8, 41), (12, 47), (1, 31), (16, 52), (6, 56),
                (4, 35), (11, 43), (9, 58)]
    for (i, j) in contacts:
        assert abs(i - j) >= min_sep, "planted contact %d,%d is not long-range" % (i, j)

    print("=" * 72)
    print("H2 control C0/C2/C3 -- MRF behaviour under alphabet permutation")
    print("=" * 72)
    print("MSA: N=%d  L=%d  planted long-range contacts=%d  min_sep=%d"
          % (n_seq, length, len(contacts), min_sep))

    msa = synth_msa(n_seq, length, contacts, rng)

    # ---- C0: does the MRF recover the planted couplings at all? ----------------
    scores = mrf.contact_map(msa)
    prec = mrf.precision_at_frac(scores, contacts, length, min_sep=min_sep)
    rec = mrf.recall_at_frac(scores, contacts, length, min_sep=min_sep)
    ceil = mrf.precision_ceiling(contacts, length, min_sep=min_sep)
    n_pred = max(1, int(length * 0.5))
    chance = len(contacts) / float(length * (length - 1) / 2)
    print("\n[C0] recovery")
    print("     precision@L/2 = %.4f  (ceiling %.4f, top %d pairs, chance ~%.4f)"
          % (prec, ceil, n_pred, chance))
    print("     recall@L/2    = %.4f" % rec)
    c0_pass = rec > 0.8
    print("     %s" % ("PASS" if c0_pass else "FAIL -- MRF cannot see planted signal"))

    # ---- C2: exact invariance under a CONSISTENT permutation ------------------
    print("\n[C2] exact invariance under consistent permutation")
    c2_pass = True
    for regime in ("random", "In-class", "Cross-class"):
        for m in (2, 6, 10, 20):
            if regime == "Cross-class" and m > 20:
                continue
            perm = permute.PERM_BUILDERS[regime](m, np.random.default_rng(m))
            msa_p = permute.apply_perm_msa(msa, perm)

            # C5: a consistent permutation permutes counts but preserves the multiset.
            comp_ok = permute.check_composition_invariant(msa[0], msa_p[0])

            scores_p = mrf.contact_map(msa_p)
            max_abs = np.abs(scores_p - scores).max()
            rec_p = mrf.recall_at_frac(scores_p, contacts, length, min_sep=min_sep)
            ok = np.allclose(scores_p, scores, atol=1e-8) and comp_ok
            c2_pass = c2_pass and ok
            print("     %-13s m=%-3d  max|dS|=%.3e  recall=%.4f  comp_ok=%s  %s"
                  % (regime, m, max_abs, rec_p, comp_ok, "PASS" if ok else "FAIL"))

    # ---- C3: the CORRECT coevolution null -------------------------------------
    print("\n[C3] destruction under per-COLUMN row shuffle (true null)")
    msa_null = mrf.shuffle_columns(msa, np.random.default_rng(7))
    rec_null = mrf.recall_at_frac(
        mrf.contact_map(msa_null), contacts, length, min_sep=min_sep)
    print("     recall@L/2 = %.4f  (was %.4f)" % (rec_null, rec))
    c3_pass = rec_null < 0.25 * rec
    print("     %s" % ("PASS -- dependency genuinely destroyed"
                       if c3_pass else "FAIL -- metric may be insensitive"))

    # ---- C3b: per-SEQUENCE permutation is NOT a null --------------------------
    print("\n[C3b] per-SEQUENCE permutation (relabels both columns of a row alike)")
    msa_seq = permute.apply_perm_msa_inconsistent(msa, 20, np.random.default_rng(7))
    rec_seq = mrf.recall_at_frac(
        mrf.contact_map(msa_seq), contacts, length, min_sep=min_sep)
    print("     recall@L/2 = %.4f  (was %.4f, true null %.4f)" % (rec_seq, rec, rec_null))
    print("     interpretation: MI between columns is invariant under a within-row")
    print("     bijection, so this only BLURS coevolution. It is not a destruction")
    print("     control -- it is a third experimental condition.")

    print("\n" + "=" * 72)
    verdict = c0_pass and c2_pass and c3_pass
    print("VERDICT: %s" % ("ALL CONTROLS PASS -- H2 premise verified, safe to proceed"
                           if verdict else "CONTROL FAILURE -- do not proceed to GPU"))
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

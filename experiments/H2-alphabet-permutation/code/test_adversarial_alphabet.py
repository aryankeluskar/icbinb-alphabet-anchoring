"""Adversarial C2: deliberately inject every character that could break invariance.

The real-Pfam run (test_real_msa_invariance.py) passed, but it did NOT actually test what
it was written to test: Pfam seed alignments are hand-curated, and our own >30%-gap column
filter stripped the messy columns, so the run reported gaps=0.0% and non-standard=none.
A pass on clean data is not evidence about dirty data.

So rather than hoping a download contains the awkward cases, we construct them. This
injects, at controlled rates, every character class that touches the catch-all branch of
`one_hot_msa` (which maps anything unrecognised to the gap state):

    -   gap
    X   unknown residue
    B   D/N ambiguity
    Z   E/Q ambiguity
    U   selenocysteine
    O   pyrrolysine
    *   translation stop

If a permutation ever moved a character into or out of that catch-all, invariance would
break and the contact map would shift. `permute.PROTECTED` is supposed to prevent exactly
this. This test is what makes that a verified claim rather than an intended one.

Also checks the boundary case that PROTECTED is doing real work, by running a deliberately
BROKEN permutation that does touch X/B/Z and confirming it DOES perturb the map -- otherwise
a pass here could just mean the injected characters are too rare to matter.

Stock Python 3.6 + numpy. No downloads, no GPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import mrf          # noqa: E402
import permute      # noqa: E402

NONSTANDARD = "XBZUO*"


def synth_msa(n_seq, length, contacts, rng, coupling_strength=0.9):
    """Same generator as the main control: planted couplings over a clean alphabet."""
    aa = np.array(list(mrf.AA20))
    arr = rng.choice(aa, size=(n_seq, length))
    for (i, j) in contacts:
        partner = dict(zip(mrf.AA20, rng.permutation(list(mrf.AA20))))
        coupled = rng.random(n_seq) < coupling_strength
        for s in range(n_seq):
            if coupled[s]:
                arr[s, j] = partner[arr[s, i]]
    return ["".join(row) for row in arr]


def inject(msa, rng, gap_rate=0.15, nonstd_rate=0.05):
    """Sprinkle gaps and non-standard residues through an otherwise clean MSA."""
    arr = np.array([list(s) for s in msa])
    n, length = arr.shape

    gap_mask = rng.random((n, length)) < gap_rate
    arr[gap_mask] = "-"

    nonstd_mask = (rng.random((n, length)) < nonstd_rate) & (~gap_mask)
    picks = rng.choice(list(NONSTANDARD), size=int(nonstd_mask.sum()))
    arr[nonstd_mask] = picks
    return ["".join(row) for row in arr]


def broken_perm(rng):
    """A DELIBERATELY WRONG permutation that also relabels non-standard characters."""
    perm = permute.random_perm(20, rng)
    # Map the non-standard characters onto real residues -- exactly what PROTECTED forbids.
    perm.update({"X": "A", "B": "D", "Z": "E", "U": "C", "O": "K", "*": "G"})
    return perm


def apply_unguarded(msa, perm):
    """Apply a permutation WITHOUT the PROTECTED guard.

    `permute.apply_perm` enforces PROTECTED at the application site, so it will pass X/B/Z
    through untouched no matter what the permutation dict says. That is the right defensive
    design, but it means a 'broken' permutation cannot express the breakage we need for a
    positive control. This helper bypasses the guard so we can show that relabeling the
    non-standard characters WOULD move the contact map -- which is what establishes that
    the guard is load-bearing and that the clean pass above has power.
    """
    return ["".join(perm.get(c, c) for c in s) for s in msa]


def census(msa):
    seen = {}
    for s in msa:
        for c in s:
            seen[c] = seen.get(c, 0) + 1
    return seen


def main():
    rng = np.random.default_rng(0)
    n_seq, length, min_sep = 1500, 60, 24
    contacts = [(2, 40), (5, 33), (10, 50), (14, 44), (0, 55), (7, 38),
                (3, 45), (8, 41), (12, 47), (1, 31), (16, 52), (6, 56),
                (4, 35), (11, 43), (9, 58)]

    print("=" * 76)
    print("Adversarial C2 -- invariance with gaps and non-standard residues injected")
    print("=" * 76)

    clean = synth_msa(n_seq, length, contacts, rng)
    msa = inject(clean, rng)

    cen = census(msa)
    total = float(n_seq * length)
    print("MSA: N=%d  L=%d" % (n_seq, length))
    print("     gaps        %5.1f%%" % (100 * cen.get("-", 0) / total))
    for c in NONSTANDARD:
        print("     %-11s %5.2f%%" % (c, 100 * cen.get(c, 0) / total))

    scores = mrf.contact_map(msa)
    base_top = set(mrf.top_contacts(scores, length, min_sep=min_sep, top_frac=0.5))
    rec = mrf.recall_at_frac(scores, contacts, length, min_sep=min_sep)
    print("\n[C0'] signal survives the corruption: recall@L/2 = %.4f" % rec)
    c0_ok = rec > 0.5
    print("      %s" % ("PASS" if c0_ok
                        else "FAIL -- corruption destroyed the signal, test has no power"))

    print("\n[C2'] invariance with dirty input")
    ok_all = True
    for regime in ("random", "In-class", "Cross-class"):
        for m in (6, 20):
            perm = permute.PERM_BUILDERS[regime](m, np.random.default_rng(m))
            msa_p = permute.apply_perm_msa(msa, perm)
            cen_p = census(msa_p)

            # Every protected character must survive in exactly the same count.
            prot_ok = all(cen_p.get(c, 0) == cen.get(c, 0) for c in NONSTANDARD + "-")

            scores_p = mrf.contact_map(msa_p)
            max_abs = np.abs(scores_p - scores).max()
            top_same = set(mrf.top_contacts(
                scores_p, length, min_sep=min_sep, top_frac=0.5)) == base_top

            ok = max_abs < 1e-8 and top_same and prot_ok
            ok_all = ok_all and ok
            print("      %-13s m=%-3d max|dS|=%.2e  top-L/2 identical=%-5s "
                  "protected=%-5s %s"
                  % (regime, m, max_abs, top_same, prot_ok, "PASS" if ok else "FAIL"))

    # Positive control: PROTECTED must be load-bearing.
    print("\n[C2''] positive control -- relabel X/B/Z/U/O/* too, bypassing the guard")
    bad = broken_perm(np.random.default_rng(3))

    # First confirm the guard actually blocks this when going through the normal path.
    guarded = permute.apply_perm_msa(msa, bad)
    cen_guarded = census(guarded)
    guard_holds = all(cen_guarded.get(c, 0) == cen.get(c, 0) for c in NONSTANDARD)
    print("      guard blocks it via apply_perm_msa: %s" % guard_holds)

    msa_bad = apply_unguarded(msa, bad)
    cen_bad = census(msa_bad)
    altered = [c for c in NONSTANDARD if cen_bad.get(c, 0) != cen.get(c, 0)]
    scores_bad = mrf.contact_map(msa_bad)
    delta_bad = np.abs(scores_bad - scores).max()
    top_bad_same = set(mrf.top_contacts(
        scores_bad, length, min_sep=min_sep, top_frac=0.5)) == base_top

    power_ok = guard_holds and delta_bad > 1e-6 and altered
    print("      unguarded: max|dS| = %.3e  top-L/2 identical=%s  chars altered: %s"
          % (delta_bad, top_bad_same, "".join(altered) if altered else "none"))
    print("      %s" % ("PASS -- PROTECTED is load-bearing; the clean pass has power"
                        if power_ok else
                        "FAIL -- could not demonstrate the guard matters"))

    print("\n" + "=" * 76)
    verdict = c0_ok and ok_all and power_ok
    print("VERDICT: %s" % ("invariance robust to gaps and non-standard residues"
                           if verdict else "FAILURE -- fix before any pLM run"))
    print("=" * 76)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

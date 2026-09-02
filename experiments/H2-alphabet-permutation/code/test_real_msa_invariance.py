"""C2 on REAL alignments: does exact invariance survive messy biological data?

The synthetic control (test_mrf_invariance.py) used a clean 20-letter alphabet with no
gaps. Real alignments are messier in ways that could silently break the pipeline:

  - gap columns and gap-rich sequences
  - non-standard residues: X (unknown), B (D/N), Z (E/Q), U (selenocysteine), O
  - lowercase letters and '.' marking INSERT columns in Stockholm format
  - wildly non-uniform column composition and strong phylogenetic correlation

Each of these touches `one_hot_msa`, which maps anything unrecognised to the gap state.
If a permutation ever moved a residue into or out of that catch-all, invariance would
break. It should not -- PROTECTED excludes X/B/U/Z/O/-/. and pi is closed over AA20 --
but "should not" is exactly the kind of claim this project exists to stop trusting.

We also check something stronger than max|dS|: whether the predicted top-L/2 long-range
contact SET is literally identical before and after. That is the quantity the paper will
report, so it is the one that has to be invariant.

Downloads Pfam seed alignments from InterPro. Stock Python 3.6 + numpy, no GPU.
"""
import gzip
import io
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import mrf          # noqa: E402
import permute      # noqa: E402

# A deliberately varied set: signalling, RNA-binding, protein-protein interaction,
# enzymatic. Different lengths, depths and gap patterns.
FAMILIES = [
    ("PF00072", "Response_reg"),
    ("PF00076", "RRM_1"),
    ("PF00018", "SH3_1"),
    ("PF00013", "KH_1"),
    ("PF00027", "cNMP_binding"),
]

URL = ("https://www.ebi.ac.uk/interpro/api/entry/pfam/{acc}/"
       "?annotation=alignment:seed&download")


def fetch_stockholm(acc, cache_dir):
    """Download a Pfam seed alignment, caching to disk."""
    path = os.path.join(cache_dir, acc + ".sto")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as fh:
            return fh.read()
    req = urllib.request.Request(
        URL.format(acc=acc),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    raw = urllib.request.urlopen(req, timeout=120).read()
    # InterPro serves these gzipped regardless of the Accept-Encoding we send.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    text = raw.decode("utf-8", errors="replace")
    with open(path, "w") as fh:
        fh.write(text)
    return text


def parse_stockholm(text):
    """Extract aligned sequences from Stockholm, keeping MATCH columns only.

    In Stockholm, lowercase letters and '.' mark insert columns relative to the model.
    Those columns are not part of the alignment proper and are dropped, which is what
    every coevolution pipeline does.
    """
    seqs = {}
    order = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, chunk = parts
        if name not in seqs:
            seqs[name] = []
            order.append(name)
        seqs[name].append(chunk.strip())
    msa = ["".join(seqs[n]) for n in order]
    if not msa:
        return []
    length = len(msa[0])
    msa = [s for s in msa if len(s) == length]
    # Keep a column only if it is a match column (never lowercase, never '.').
    keep = [j for j in range(length)
            if all(not s[j].islower() and s[j] != "." for s in msa)]
    return ["".join(s[j] for j in keep).upper() for s in msa]


def gap_filter(msa, max_gap_frac=0.3):
    """Drop columns that are mostly gaps -- standard preprocessing."""
    if not msa:
        return msa
    arr = np.array([list(s) for s in msa])
    frac = (arr == "-").mean(axis=0)
    keep = np.where(frac <= max_gap_frac)[0]
    return ["".join(row[keep]) for row in arr]


def alphabet_census(msa):
    """What characters actually appear? This is the part that could break invariance."""
    seen = {}
    for s in msa:
        for c in s:
            seen[c] = seen.get(c, 0) + 1
    return seen


def main():
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "pfam")
    cache = os.path.abspath(cache)
    if not os.path.isdir(cache):
        os.makedirs(cache)

    print("=" * 74)
    print("C2 on REAL Pfam alignments -- does exact invariance survive messy data?")
    print("=" * 74)

    all_ok = True
    for acc, name in FAMILIES:
        try:
            text = fetch_stockholm(acc, cache)
        except Exception as exc:                      # noqa: BLE001
            print("\n%s (%s): FETCH FAILED -- %s" % (acc, name, exc))
            all_ok = False
            continue

        msa = gap_filter(parse_stockholm(text))
        if not msa or len(msa) < 20 or len(msa[0]) < 30:
            print("\n%s (%s): unusable (N=%d, L=%d)"
                  % (acc, name, len(msa), len(msa[0]) if msa else 0))
            continue

        n, length = len(msa), len(msa[0])
        census = alphabet_census(msa)
        nonstd = {c: k for c, k in census.items() if c not in mrf.AA20 and c != "-"}
        gap_frac = census.get("-", 0) / float(n * length)

        print("\n%s (%s): N=%d  L=%d  gaps=%.1f%%  non-standard=%s"
              % (acc, name, n, length, 100 * gap_frac,
                 (nonstd if nonstd else "none")))

        scores = mrf.contact_map(msa)
        base_top = set(mrf.top_contacts(scores, length, min_sep=24, top_frac=0.5))

        fam_ok = True
        for regime in ("random", "In-class", "Cross-class"):
            for m in (6, 20):
                perm = permute.PERM_BUILDERS[regime](m, np.random.default_rng(m))
                msa_p = permute.apply_perm_msa(msa, perm)

                # The permutation must not create or destroy any non-AA20 character.
                cen_p = alphabet_census(msa_p)
                nonstd_p = {c: k for c, k in cen_p.items()
                            if c not in mrf.AA20 and c != "-"}
                protected_ok = (nonstd_p == nonstd
                                and cen_p.get("-", 0) == census.get("-", 0))

                scores_p = mrf.contact_map(msa_p)
                max_abs = np.abs(scores_p - scores).max()
                top_p = set(mrf.top_contacts(scores_p, length, min_sep=24, top_frac=0.5))
                set_same = (top_p == base_top)
                comp_ok = permute.check_composition_invariant(msa[0], msa_p[0])

                ok = (max_abs < 1e-8) and set_same and comp_ok and protected_ok
                fam_ok = fam_ok and ok
                print("    %-13s m=%-3d max|dS|=%.2e  top-L/2 identical=%-5s "
                      "protected=%-5s comp=%-5s %s"
                      % (regime, m, max_abs, set_same, protected_ok, comp_ok,
                         "PASS" if ok else "FAIL"))
        all_ok = all_ok and fam_ok

    print("\n" + "=" * 74)
    print("VERDICT: %s" % ("invariance holds on real alignments" if all_ok
                           else "FAILURE on real data -- investigate before any pLM run"))
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

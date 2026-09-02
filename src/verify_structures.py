"""Self-test for src/structures.py against four small, well-behaved crystal structures.

Run:  source src/activate_env.sh && python -u src/verify_structures.py

The point of this script is to fail LOUDLY. A silently wrong sequence->structure mapping
does not crash anything downstream; it just produces contact-precision numbers that are
plausible and meaningless. So two bars are checked explicitly for every target:

  BAR 1  coverage:  >= 90% of query positions map to an observed residue
  BAR 2  density:   the number of true long-range (|i-j| >= 12) CB-CB contacts is
                    between 0.5x and 2x the observed chain length, which is the range a
                    compact globular domain lives in. Outside it, either the chain is not
                    globular or the geometry parse is wrong.

A third, softer check reports sequence identity over the mapped positions. Below 100% the
deposited construct is not the query -- point mutant, ortholog, tag -- and the caller must
decide whether that is acceptable.
"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import structures  # noqa: E402

H2_SCRIPT = os.path.join(HERE, "..", "experiments", "H2-alphabet-permutation",
                         "code", "run_h2_esm2.py")

TARGETS = [
    ("ubiquitin",    "1UBQ", "A"),
    ("protein_G_B1", "1PGB", "A"),
    ("cytochrome_c", "1HRC", "A"),
    ("lysozyme_T4",  "2LZM", "A"),
]

MIN_SEP = 12
CUTOFF = 8.0
COVERAGE_BAR = 0.90
DENSITY_LO, DENSITY_HI = 0.5, 2.0


def load_extra(path):
    """Pull the EXTRA dict out of run_h2_esm2.py without importing torch."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "EXTRA":
                    return ast.literal_eval(node.value)
    raise RuntimeError("no EXTRA dict in %s" % path)


def check_precision_machinery(cmap, mapping, length):
    """Two sanity probes on precision_at_topk that need no model."""
    rng = np.random.default_rng(0)
    idx = np.array([-1 if m is None else m for m in mapping])

    # (a) an oracle that scores true contacts highest must reach precision 1.0
    oracle = rng.random((length, length)) * 0.01
    for i in range(length):
        for j in range(length):
            if idx[i] >= 0 and idx[j] >= 0 and cmap[idx[i], idx[j]]:
                oracle[i, j] += 1.0
    oracle = (oracle + oracle.T) / 2.0
    p_oracle, _ = structures.precision_at_topk(oracle, cmap, mapping, min_sep=MIN_SEP)

    # (b) random scores must land near the base rate of long-range contacts
    p_rand = []
    for s in range(20):
        r = rng.random((length, length))
        r = (r + r.T) / 2.0
        p, _ = structures.precision_at_topk(r, cmap, mapping, min_sep=MIN_SEP)
        p_rand.append(p)
    return p_oracle, float(np.mean(p_rand))


def unit_tests(query):
    """Exercise the code paths the four crystal targets do NOT reach.

    All four are fully ordered, 100%-observed chains, so a run over them alone never
    touches the gap machinery -- which is exactly the machinery that will be used on every
    real target with a disordered loop. These checks run first and are hard failures.
    """
    problems = []

    # 1. NW must reach the true linear-gap optimum (checked against exhaustive recursion).
    def brute(a, b, gap=-1.0, match=1.0, mismatch=0.0):
        from functools import lru_cache

        @lru_cache(None)
        def f(i, j):
            if i == len(a) and j == len(b):
                return 0.0
            best = -1e18
            if i < len(a) and j < len(b):
                best = max(best, (match if a[i] == b[j] else mismatch) + f(i + 1, j + 1))
            if i < len(a):
                best = max(best, gap + f(i + 1, j))
            if j < len(b):
                best = max(best, gap + f(i, j + 1))
            return best
        return f(0, 0)

    import random
    random.seed(7)
    n_bad = 0
    for _ in range(200):
        a = "".join(random.choice("ACDEFG") for _ in range(random.randint(0, 12)))
        b = "".join(random.choice("ACDEFG") for _ in range(random.randint(0, 12)))
        pairs = structures._needleman_wunsch(a, b)
        got = sum(-1.0 if (i is None or j is None)
                  else (1.0 if a[i] == b[j] else 0.0) for i, j in pairs)
        ra = "".join(a[i] for i, _ in pairs if i is not None)
        rb = "".join(b[j] for _, j in pairs if j is not None)
        if abs(got - brute(a, b)) > 1e-9 or ra != a or rb != b:
            n_bad += 1
    if n_bad:
        problems.append("NW is not optimal on %d/200 random pairs" % n_bad)

    # 2. Unobserved termini must map exactly, via the substring path.
    frag = query[5:-4]
    m = structures.align_to_query(frag, query)
    if not (m[:5] == [None] * 5 and m[-4:] == [None] * 4
            and all(m[5 + k] == k for k in range(len(frag)))):
        problems.append("unobserved-termini mapping is wrong")

    # 3. An internal disordered loop must give ONE contiguous gap at the right place.
    for lo, hi in [(30, 39), (50, 64)]:
        frag = query[:lo] + query[hi:]
        m = structures.align_to_query(frag, query)
        unmapped = [i for i, v in enumerate(m) if v is None]
        exact = (all(m[i] == i for i in range(lo))
                 and all(m[i] == i - (hi - lo) for i in range(hi, len(query))))
        if unmapped != list(range(lo, hi)) or not exact:
            problems.append("internal gap %d-%d misplaced: unmapped=%s"
                            % (lo, hi - 1, unmapped))

    # 4. precision_at_topk must drop unmapped pairs and still let an oracle reach 1.0.
    frag = query[5:-4]
    m = structures.align_to_query(frag, query)
    xyz = np.random.default_rng(1).normal(size=(len(frag), 3)) * 12
    cm = structures.contact_map(xyz)
    idx = np.array([-1 if v is None else v for v in m])
    oracle = np.random.default_rng(2).random((len(query), len(query))) * 1e-3
    for i in range(len(query)):
        for j in range(len(query)):
            if idx[i] >= 0 and idx[j] >= 0 and cm[idx[i], idx[j]]:
                oracle[i, j] += 1.0
    oracle = (oracle + oracle.T) / 2
    p, _n, det = structures.precision_at_topk(oracle, cm, m, min_sep=MIN_SEP,
                                              return_details=True)
    n_all = np.triu_indices(len(query), k=MIN_SEP)[0].size
    if p < 0.999:
        problems.append("gapped oracle precision %.3f != 1.000" % p)
    if det["n_candidates"] >= n_all:
        problems.append("unmapped pairs were not excluded from the candidate set")

    # 5. A fully unmapped target must yield nan, not a crash or a fake 0.0.
    p, n = structures.precision_at_topk(np.zeros((10, 10)), cm, [None] * 10, min_sep=2)
    if not (np.isnan(p) and n == 0):
        problems.append("all-unmapped case returned (%r, %r) instead of (nan, 0)" % (p, n))

    return problems


def main():
    extra = load_extra(H2_SCRIPT)
    cache = structures.DEFAULT_CACHE
    print("cache dir : %s" % cache)
    print("query src : %s\n" % os.path.normpath(H2_SCRIPT))

    rows = []
    failures = []

    unit_problems = unit_tests(extra["ubiquitin"])
    if unit_problems:
        print("UNIT TESTS FAILED:")
        for p in unit_problems:
            print("  * %s" % p)
        failures.extend("unit: " + p for p in unit_problems)
    else:
        print("unit tests (NW optimality, gapped mapping, scoring): pass\n")

    for name, pdb_id, chain in TARGETS:
        if name not in extra:
            failures.append("%s: not in EXTRA dict" % name)
            continue
        query = extra[name]
        try:
            path = structures.fetch_pdb(pdb_id, cache_dir=cache)
        except Exception as exc:                                    # noqa: BLE001
            failures.append("%s (%s): DOWNLOAD FAILED -- %s" % (name, pdb_id, exc))
            print("!! %s (%s) unavailable: %s\n" % (name, pdb_id, exc))
            continue

        seq, coords, resids, used_ca = structures.parse_chain(path, chain)
        mapping, stats = structures.align_to_query(seq, query, return_stats=True)
        cmap = structures.contact_map(coords, cutoff=CUTOFF)
        nlr = structures.n_long_range(cmap, min_sep=MIN_SEP)

        lq, lp = len(query), len(seq)
        cov = stats["coverage"]
        density = nlr / float(lp)
        pair_frac = nlr / float(max(1, (lp - MIN_SEP) * (lp - MIN_SEP + 1) // 2))

        ok_cov = cov >= COVERAGE_BAR
        ok_den = DENSITY_LO <= density <= DENSITY_HI
        verdict = "PASS" if (ok_cov and ok_den) else "FAIL"
        if not ok_cov:
            failures.append("%s: coverage %.1f%% < %.0f%%" % (name, 100 * cov,
                                                              100 * COVERAGE_BAR))
        if not ok_den:
            failures.append("%s: long-range contacts %d = %.2fx chain length "
                            "(outside %.1fx-%.1fx)" % (name, nlr, density,
                                                       DENSITY_LO, DENSITY_HI))

        p_oracle, p_rand = check_precision_machinery(cmap, mapping, lq)
        if not (p_oracle > 0.999):
            failures.append("%s: oracle precision %.3f != 1.000 -- scoring path is broken"
                            % (name, p_oracle))

        rows.append(dict(name=name, pdb="%s_%s" % (pdb_id, chain), lq=lq, lp=lp,
                         mapped=stats["n_mapped"], cov=cov, nlr=nlr, density=density,
                         pair_frac=pair_frac, ident=stats["identity"],
                         method=stats["method"], verdict=verdict,
                         ca_fallback=sum(used_ca), p_oracle=p_oracle, p_rand=p_rand,
                         mismatches=stats["mismatches"],
                         first_resid=resids[0], last_resid=resids[-1]))

    hdr = ("%-14s %-7s %5s %5s %7s %7s %8s %8s %8s  %-20s %s"
           % ("target", "pdb", "Lq", "Lpdb", "mapped", "cov", "LRcont", "dens",
              "ident", "align", "bar"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-14s %-7s %5d %5d %7d %6.1f%% %8d %8.2f %7.1f%%  %-20s %s"
              % (r["name"], r["pdb"], r["lq"], r["lp"], r["mapped"], 100 * r["cov"],
                 r["nlr"], r["density"], 100 * r["ident"], r["method"], r["verdict"]))

    print("\nlegend: cov = mapped/Lq (bar >= %.0f%%); dens = LRcont/Lpdb "
          "(bar %.1f-%.1f); LRcont counts |i-j| >= %d CB-CB pairs under %.1f A."
          % (100 * COVERAGE_BAR, DENSITY_LO, DENSITY_HI, MIN_SEP, CUTOFF))

    print("\nper-target detail")
    for r in rows:
        print("  %s (%s)  residues %s..%s  CA-fallback %d  "
              "long-range pair density %.3f of all |i-j|>=%d pairs"
              % (r["name"], r["pdb"], r["first_resid"], r["last_resid"],
                 r["ca_fallback"], r["pair_frac"], MIN_SEP))
        print("      scoring probe: oracle P@L/2 = %.3f (must be 1.000), "
              "random P@L/2 = %.3f (chance)" % (r["p_oracle"], r["p_rand"]))
        if r["mismatches"]:
            shown = ", ".join("%s%d%s" % (q, k + 1, p) for k, q, p in r["mismatches"][:12])
            more = "" if len(r["mismatches"]) <= 12 else " (+%d more)" % (
                len(r["mismatches"]) - 12)
            print("      %d residue mismatch(es) query->pdb: %s%s"
                  % (len(r["mismatches"]), shown, more))

    print("\n" + "=" * 78)
    if failures:
        print("SANITY BAR: FAILED for %d check(s)" % len(failures))
        for f in failures:
            print("  * %s" % f)
    else:
        print("SANITY BAR: all %d targets pass coverage and density." % len(rows))
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""H4 Amendment 1 -- aggregated DV.

The primary per-variant analysis returned a clean null, diagnosed as (i) 58/69 assays have
k_max=2 so k barely varies, and (ii) per-variant rank error is dominated by DMS measurement
noise so both coefficients sat at ~0.05-0.10.

This attacks (ii): partition each assay into (k x severity-tercile) cells, compute Spearman
inside each cell of >=50 variants, and regress cell Spearman on standardised k and severity.
Rank correlation over 50+ points is far more stable than a single variant's error.

It does NOT fix (i), so k-rich (k_max>=4) and k-poor assays are reported separately.

The primary null is not overwritten by whatever this returns -- see protocol Amendment 1.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCORES = os.path.join(ROOT, "data", "proteingym", "zero_shot_substitutions_scores")
OUT = os.path.join(HERE, "..", "results")

sys.path.insert(0, HERE)
from run_h4 import (NON_MODEL, PLM, ALIGNMENT, parse_mutant, severity,  # noqa: E402
                    check_c4, MAX_ROWS)

MIN_CELL = 50      # variants needed inside a cell to trust its Spearman
MIN_CELLS = 6      # cells needed per assay to fit 2 predictors


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def standardise(v):
    v = np.asarray(v, dtype=float)
    return None if v.std() < 1e-12 else (v - v.mean()) / v.std()


def fit(y, k, sev):
    zy, zk, zs = standardise(y), standardise(k), standardise(sev)
    if zy is None or zk is None or zs is None:
        return None
    X = np.column_stack([zk, zs])
    beta, _, rank, _ = np.linalg.lstsq(X, zy, rcond=None)
    return (float(beta[0]), float(beta[1])) if rank >= 2 else None


def main():
    check_c4()
    targets = set(PLM) | set(ALIGNMENT)
    files = sorted(f for f in os.listdir(SCORES) if f.endswith(".csv"))
    print("H4 Amendment 1 -- aggregated DV (within-cell Spearman)")
    print("assays on disk: %d   min cell=%d   min cells/assay=%d\n"
          % (len(files), MIN_CELL, MIN_CELLS))

    rows = []
    n_assay_ok = n_assay_drop = 0

    for fi, fn in enumerate(files):
        assay = fn[:-4]
        with open(os.path.join(SCORES, fn)) as fh:
            rdr = csv.DictReader(fh)
            model_cols = [c for c in rdr.fieldnames
                          if c not in NON_MODEL and c in targets]
            recs = list(rdr)
        if not model_cols:
            continue

        ks, sevs, dms, keep = [], [], [], []
        for i, r in enumerate(recs):
            p = parse_mutant(r["mutant"])
            if p is None:
                continue
            sv = [severity(w, m) for w, _, m in p]
            if any(s is None for s in sv):
                continue
            try:
                d = float(r["DMS_score"])
            except (ValueError, TypeError):
                continue
            ks.append(len(p))
            sevs.append(float(np.mean(sv)))
            dms.append(d)
            keep.append(i)

        if len(ks) > MAX_ROWS:
            sub = np.random.default_rng(12345).choice(len(ks), MAX_ROWS, replace=False)
            sub.sort()
            ks = [ks[i] for i in sub]
            sevs = [sevs[i] for i in sub]
            dms = [dms[i] for i in sub]
            keep = [keep[i] for i in sub]

        ks = np.array(ks, dtype=float)
        sevs = np.array(sevs, dtype=float)
        dms = np.array(dms, dtype=float)
        if len(ks) < MIN_CELL * MIN_CELLS or ks.max() < 2:
            continue
        k_max = int(ks.max())

        # Cells: k value x severity tercile (terciles computed within each k).
        cells = defaultdict(list)
        for kv in np.unique(ks):
            idx = np.where(ks == kv)[0]
            if len(idx) < MIN_CELL:
                continue
            s = sevs[idx]
            q1, q2 = np.percentile(s, [33.333, 66.667])
            for t, sel in enumerate((s <= q1, (s > q1) & (s <= q2), s > q2)):
                members = idx[sel]
                if len(members) >= MIN_CELL:
                    cells[(kv, t)] = members

        if len(cells) < MIN_CELLS:
            n_assay_drop += 1
            continue
        n_assay_ok += 1

        for mc in model_cols:
            vals = np.full(len(keep), np.nan)
            for j, i in enumerate(keep):
                try:
                    vals[j] = float(recs[i].get(mc, ""))
                except (ValueError, TypeError):
                    pass

            cy, ck, cs = [], [], []
            for (kv, _t), members in cells.items():
                m = members[~np.isnan(vals[members])]
                if len(m) < MIN_CELL:
                    continue
                rho = spearman(vals[m], dms[m])
                if rho is None:
                    continue
                cy.append(rho)
                ck.append(kv)
                cs.append(float(sevs[m].mean()))

            if len(cy) < MIN_CELLS:
                continue
            res = fit(cy, ck, cs)
            if res is None:
                continue
            bk, bs = res
            rows.append(dict(assay=assay, model=mc, beta_k=round(bk, 5),
                             beta_sev=round(bs, 5), D=round(abs(bk) - abs(bs), 5),
                             n_cells=len(cy), k_max=k_max))

        if (fi + 1) % 60 == 0:
            print("  ... %d/%d files, %d fits" % (fi + 1, len(files), len(rows)))

    print("\nassays usable: %d   dropped for <%d cells: %d"
          % (n_assay_ok, MIN_CELLS, n_assay_drop))
    if not rows:
        print("no fits")
        return 1

    with open(os.path.join(OUT, "h4_strata.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(0)

    def ci(v, it=5000):
        v = np.array(v, dtype=float)
        b = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(it)]
        return float(np.median(v)), float(np.percentile(b, 2.5)), \
            float(np.percentile(b, 97.5))

    for subset, label in ((lambda r: True, "ALL assays"),
                          (lambda r: r["k_max"] >= 4, "k-rich assays (k_max>=4)")):
        sel = [r for r in rows if subset(r)]
        if not sel:
            continue
        by = defaultdict(list)
        for r in sel:
            by[r["model"]].append(r["D"])
        print("\n## %s  (%d assays)" % (label, len({r["assay"] for r in sel})))
        print("| model | median D | 95% CI | n |")
        print("|---|---|---|---|")
        plm_v, aln_v = [], []
        for grp, lab, acc in ((PLM, "pLM", plm_v), (ALIGNMENT, "alignment", aln_v)):
            print("| **%s** | | | |" % lab)
            for m in grp:
                if m in by and len(by[m]) >= 3:
                    md, lo, hi = ci(by[m])
                    acc.extend(by[m])
                    flag = "  **CI excludes 0**" if (lo > 0 or hi < 0) else ""
                    print("| %s | %+.3f | [%+.3f, %+.3f] | %d |%s"
                          % (m, md, lo, hi, len(by[m]), flag))
        if plm_v and aln_v:
            a, b = np.array(plm_v), np.array(aln_v)
            allv = np.concatenate([a, b])
            r = allv.argsort().argsort().astype(float) + 1
            u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
            mu = len(a) * len(b) / 2.0
            sd = np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12.0)
            print("\npLM median D = %+.4f (n=%d);  alignment = %+.4f (n=%d);  "
                  "Mann-Whitney z = %.2f" % (np.median(a), len(a), np.median(b),
                                             len(b), (u - mu) / sd))
    return 0


if __name__ == "__main__":
    sys.exit(main())

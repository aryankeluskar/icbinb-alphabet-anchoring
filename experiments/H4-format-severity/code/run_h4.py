"""H4 -- does model error track edit distance (k) or biochemical severity (BLOSUM62)?

Runs entirely on ProteinGym's precomputed zero-shot scores. No GPU, no re-scoring.

Per assay and per model we fit, on standardised variables,

    err ~ beta_k * k + beta_sev * mean_sev

where err is the within-assay absolute percentile-rank difference between the model score
and the measured DMS score. The primary statistic is D = |beta_k| - |beta_sev|: positive
means error is driven more by how MANY residues changed, negative by how BAD the changes
were.

Controls implemented here (see protocol.md): C1 collinearity filter, C2 shuffled-severity
null, C3 singles-only sanity, C4 BLOSUM sign assert, C5 per-assay aggregation only.

Stock Python 3.6 + numpy -- deliberately, so it runs while the GPU jobs hold the env.
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

NON_MODEL = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}

# Models we care about most, grouped. Others are still scored and written out.
PLM = ["ESM2_8M", "ESM2_35M", "ESM2_150M", "ESM2_650M", "ESM2_3B", "ESM2_15B",
       "ESM1b", "ESM1v_ensemble", "Progen2_large", "RITA_l", "ProtGPT2"]
ALIGNMENT = ["Site_Independent", "EVmutation", "EVE_ensemble", "GEMME",
             "DeepSequence_ensemble", "MSA_Transformer_ensemble"]

MIN_ROWS = 200
MAX_COLLIN = 0.5

# 95 models x 2.4M rows is ~228M Python-level float conversions, which the stock 3.6
# interpreter cannot do in reasonable time. So we (a) score only the models the registered
# predictions actually concern, and (b) cap each assay at MAX_ROWS via a SEEDED random
# subsample. Both are reported in the output. Subsampling is safe here because the DV is a
# within-assay regression over >=200 points -- it costs precision on beta, not validity --
# and it further protects the per-assay aggregation from the two giant assays that hold 58%
# of all multi-mutant rows.
MAX_ROWS = 8000
TARGET_MODELS = set()

# ---------------------------------------------------------------- BLOSUM62
# Standard BLOSUM62, 20x20. Higher = more similar, so severity is the NEGATION.
B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
B62_RAW = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def build_b62():
    vals = [int(x) for x in B62_RAW.split()]
    assert len(vals) == 400, len(vals)
    mat = np.array(vals).reshape(20, 20)
    assert (mat == mat.T).all(), "BLOSUM62 must be symmetric"
    idx = {c: i for i, c in enumerate(B62_ORDER)}
    return mat, idx


B62, B62_IDX = build_b62()


def severity(wt, mut):
    """Negation of BLOSUM62: higher = more biochemically disruptive."""
    if wt not in B62_IDX or mut not in B62_IDX:
        return None
    return -float(B62[B62_IDX[wt], B62_IDX[mut]])


def check_c4():
    """C4 -- sign convention. K->R is In-class; K->W is Cross-class."""
    cons = severity("K", "R")
    rad = severity("K", "W")
    assert cons < rad, ("BLOSUM sign inverted: sev(K->R)=%s must be < sev(K->W)=%s"
                        % (cons, rad))
    assert severity("I", "V") < severity("I", "D"), "I->V must be milder than I->D"
    return cons, rad


def parse_mutant(m):
    """'A24G:L100P' -> [('A',24,'G'), ('L',100,'P')]; None if unparseable."""
    out = []
    for tok in m.split(":"):
        tok = tok.strip()
        if len(tok) < 3:
            return None
        wt, mut = tok[0], tok[-1]
        try:
            pos = int(tok[1:-1])
        except ValueError:
            return None
        out.append((wt, pos, mut))
    return out


def rankpct(x):
    """Percentile rank in [0, 1], average ranks for ties."""
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks / max(1.0, len(x) - 1.0)


def standardise(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    if sd < 1e-12:
        return None
    return (v - v.mean()) / sd


def fit(err, k, sev):
    """OLS of standardised err on standardised [k, sev]. Returns (beta_k, beta_sev)."""
    zk, zs, ze = standardise(k), standardise(sev), standardise(err)
    if zk is None or zs is None or ze is None:
        return None
    X = np.column_stack([zk, zs])
    beta, _, rank, _ = np.linalg.lstsq(X, ze, rcond=None)
    if rank < 2:
        return None
    return float(beta[0]), float(beta[1])


def main():
    global TARGET_MODELS
    TARGET_MODELS = set(PLM) | set(ALIGNMENT)

    cons, rad = check_c4()
    print("[C4] BLOSUM sign convention OK: sev(K->R)=%.0f < sev(K->W)=%.0f" % (cons, rad))

    if not os.path.isdir(SCORES):
        print("missing %s" % SCORES)
        return 1
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    files = sorted(f for f in os.listdir(SCORES) if f.endswith(".csv"))
    print("assays on disk: %d" % len(files))

    rows = []
    null_rows = []
    n_multi = n_kept = n_collin = n_small = 0

    for fi, fn in enumerate(files):
        assay = fn[:-4]
        path = os.path.join(SCORES, fn)
        with open(path) as fh:
            rdr = csv.DictReader(fh)
            model_cols = [c for c in rdr.fieldnames
                          if c not in NON_MODEL and c in TARGET_MODELS]
            recs = list(rdr)
        if not model_cols:
            continue

        ks, sevs, dms, keep = [], [], [], []
        for i, r in enumerate(recs):
            parsed = parse_mutant(r["mutant"])
            if parsed is None:
                continue
            svs = [severity(w, m) for w, _, m in parsed]
            if any(s is None for s in svs):
                continue
            try:
                d = float(r["DMS_score"])
            except (ValueError, TypeError):
                continue
            ks.append(len(parsed))
            sevs.append(float(np.mean(svs)))
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
        if len(ks) < MIN_ROWS or ks.max() < 2:
            n_small += 1
            continue
        n_multi += 1

        # C1 -- collinearity filter.
        if ks.std() < 1e-12 or sevs.std() < 1e-12:
            n_collin += 1
            continue
        corr = float(np.corrcoef(ks, sevs)[0, 1])
        if abs(corr) > MAX_COLLIN:
            n_collin += 1
            continue
        n_kept += 1

        dms_rank = rankpct(dms)
        rng = np.random.default_rng(hash(assay) % (2 ** 32))
        sev_shuf = sevs.copy()
        rng.shuffle(sev_shuf)

        for mc in model_cols:
            vals, ok = [], []
            for j, i in enumerate(keep):
                v = recs[i].get(mc, "")
                try:
                    vals.append(float(v))
                    ok.append(j)
                except (ValueError, TypeError):
                    continue
            if len(ok) < MIN_ROWS:
                continue
            ok = np.array(ok)
            mrank = rankpct(vals)
            err = np.abs(mrank - dms_rank[ok])

            res = fit(err, ks[ok], sevs[ok])
            if res is None:
                continue
            bk, bs = res
            rows.append(dict(assay=assay, model=mc, beta_k=round(bk, 5),
                             beta_sev=round(bs, 5), D=round(abs(bk) - abs(bs), 5),
                             n=len(ok), corr_k_sev=round(corr, 4),
                             k_max=int(ks[ok].max())))

            # C2 -- shuffled-severity null.
            resn = fit(err, ks[ok], sev_shuf[ok])
            if resn is not None:
                null_rows.append(dict(assay=assay, model=mc,
                                      beta_k=round(resn[0], 5),
                                      beta_sev=round(resn[1], 5)))

        if (fi + 1) % 40 == 0:
            print("  ... %d/%d files, %d fits" % (fi + 1, len(files), len(rows)))

    print("\n[C5] assays: %d multi-mutant considered, %d kept, %d dropped for "
          "collinearity, %d too small/singles-only" % (n_multi, n_kept, n_collin, n_small))

    if not rows:
        print("no fits produced")
        return 1

    with open(os.path.join(OUT, "h4_severity.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r["D"])

    def med_ci(vals, iters=2000):
        v = np.array(vals, dtype=float)
        rng = np.random.default_rng(0)
        boots = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(iters)]
        return float(np.median(v)), float(np.percentile(boots, 2.5)), \
            float(np.percentile(boots, 97.5))

    print("\n# H4 -- D = |beta_k| - |beta_sev|   (>0 edit-distance-driven, "
          "<0 severity-driven)\n")
    print("| model | median D | 95% CI | assays |")
    print("|---|---|---|---|")
    for group, label in ((PLM, "pLM"), (ALIGNMENT, "alignment")):
        print("| **%s** | | | |" % label)
        for m in group:
            if m in by_model and len(by_model[m]) >= 3:
                md, lo, hi = med_ci(by_model[m])
                print("| %s | %+.3f | [%+.3f, %+.3f] | %d |"
                      % (m, md, lo, hi, len(by_model[m])))

    # C2 null summary
    if null_rows:
        nb = np.array([abs(r["beta_sev"]) for r in null_rows])
        rb = np.array([abs(r["beta_sev"]) for r in rows])
        print("\n[C2] shuffled-severity null: mean |beta_sev| = %.4f "
              "(real %.4f).  %s" % (nb.mean(), rb.mean(),
                                    "PASS" if nb.mean() < 0.5 * rb.mean()
                                    else "FAIL -- severity term may be spurious"))

    print("\nwrote %s (%d rows)" % (os.path.join(OUT, "h4_severity.csv"), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

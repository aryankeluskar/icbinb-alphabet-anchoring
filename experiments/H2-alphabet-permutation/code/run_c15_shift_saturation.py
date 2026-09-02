"""C15 -- does the Delta_LL saturation replicate, and does it hold at every scale?

Registered as protocol Amendment 15 BEFORE this file was written.

C14 found, EXPLORATORILY, that in the random arm the model-perceived shift saturates by m=6
(0.187 -> 0.288 -> 0.275 -> 0.278) while contact overlap keeps collapsing (0.711 -> 0.010).
One model, one protein set, noticed after the fact. This tests it out-of-sample.

Cells come from the committed C8 SCALE csv, not the main sweep C14 used. Only PF00018 is shared
between the two protein sets, so even the 650M row is a near-independent replication, and the
smaller models ask what C14 could not: is saturation a property of one model or of the class?

REGISTERED, per scale, RANDOM ARM ONLY:
  P1 saturation        mean dLL(m=20) - mean dLL(m=6)   <= 0.05 nats/residue
  P2 continued collapse mean overlap(m=6) - mean overlap(m=20) >= 0.10
Both must hold for that scale to count as replicating. The claim is the conjunction.

REFUTING: if P1 fails at >=3 of 4 scales, C14's saturation was specific to 650M and/or that
protein set, and the exploratory paragraph is REMOVED from the paper's sec 4.4.

The In-class arm is reported but NOT registered: its realized m saturates at 15/20 by
construction, so a plateau there is partly the permutation builder and is not evidence.

Writes results/h2_c15_shift_saturation.csv.
"""
import os

# MUST be 1. The box has 8 cores at load ~33; a larger BLAS pool thrashes and measured ~33x
# slower on the same matmul. C14 was first launched at 8 threads and paced 13 hours; at 1
# thread with batching it finished in 11 minutes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

BATCH = 8

import collections  # noqa: E402
import csv          # noqa: E402
import sys          # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import jacobian as jac_mod   # noqa: E402
import permute               # noqa: E402
from run_h2_esm2 import EXTRA, load_pfam_queries          # noqa: E402
from run_c14_perceived_shift import mean_loglik, mean_loglik_batch   # noqa: E402

SCALE_CSV = os.path.join(HERE, "..", "results", "h2_c8_scale.csv")
OUT_CSV = os.path.join(HERE, "..", "results", "h2_c15_shift_saturation.csv")

# Smallest first, so a failure surfaces on a cheap model rather than after the 650M run.
MODEL_ORDER = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
               "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]

# Registered thresholds (Amendment 15). Not tunable after the fact.
P1_MAX_DLL_RISE = 0.05
P2_MIN_OVERLAP_FALL = 0.10
REFUTE_IF_P1_FAILS_AT_LEAST = 3


def main():
    device = "cuda" if torch.cuda.is_available() and os.environ.get("C15_CUDA") else "cpu"

    with open(SCALE_CSV) as fh:
        scale = list(csv.DictReader(fh))

    pfam_dir = os.path.join(ROOT, "data", "pfam")
    proteins = dict(EXTRA)
    proteins.update(load_pfam_queries(pfam_dir))

    missing = sorted({r["protein"] for r in scale} - set(proteins))
    if missing:
        print("FATAL: unresolvable proteins %s" % missing, flush=True)
        return 1

    print("C15 shift saturation across scale   device=%s  cells=%d  models=%d"
          % (device, len(scale), len(MODEL_ORDER)), flush=True)

    rows = []
    t_start = time.time()
    for model_name in MODEL_ORDER:
        cells = [r for r in scale if r["model"] == model_name]
        if not cells:
            print("  %s: no rows, skipped" % model_name, flush=True)
            continue
        print("\n  loading %s  (%d cells)" % (model_name, len(cells)), flush=True)
        wrap = jac_mod.load_esm2(model_name, device=device)

        by_protein = collections.OrderedDict()
        for r in cells:
            by_protein.setdefault(r["protein"], []).append(r)

        for pid, prot_cells in by_protein.items():
            seq = proteins[pid]
            ll_x = mean_loglik(wrap, seq)
            prepared = []
            for r in prot_cells:
                # C8's seeding convention, so the permutation is the one that made this overlap.
                rng = np.random.default_rng(1000 * int(r["seed"]) + int(r["m"]))
                perm = permute.PERM_BUILDERS[r["regime"]](int(r["m"]), rng)
                prepared.append((r, permute.apply_perm(seq, perm),
                                 sum(1 for a, b in perm.items() if a != b)))

            for start in range(0, len(prepared), BATCH):
                chunk = prepared[start:start + BATCH]
                lls = mean_loglik_batch(wrap, [p[1] for p in chunk])
                for (r, pseq, realized), ll_p in zip(chunk, lls):
                    rows.append(dict(
                        model=model_name, protein=pid, length=len(seq), regime=r["regime"],
                        m=int(r["m"]), seed=int(r["seed"]), m_realized=realized,
                        hamming=sum(1 for a, b in zip(seq, pseq) if a != b),
                        ll_identity=round(ll_x, 6), ll_permuted=round(ll_p, 6),
                        delta_ll=round(ll_x - ll_p, 6),
                        overlap=float(r["overlap"]), chance=float(r["chance"])))
            with open(OUT_CSV, "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("    %-22s L=%-4d LL(x)=%+.4f  (%d cells, %.0fs)"
                  % (pid, len(seq), ll_x, len(prot_cells), time.time() - t_start), flush=True)

        del wrap

    print("\nwrote %s (%d rows)" % (OUT_CSV, len(rows)), flush=True)
    analyse(rows)
    return 0


def _mean(rows, model, regime, m, key):
    sel = [r[key] for r in rows
           if r["model"] == model and r["regime"] == regime and r["m"] == m]
    return float(np.mean(sel)) if sel else float("nan")


def analyse(rows):
    print("\n" + "=" * 78, flush=True)
    print("C15 ANALYSIS  (Amendment 15; thresholds fixed before measurement)", flush=True)
    print("=" * 78, flush=True)

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    for regime in ("random", "In-class"):
        tag = "REGISTERED" if regime == "random" else "descriptive only, NOT registered"
        print("\n%s arm  (%s)" % (regime, tag), flush=True)
        print("  %-22s %8s %8s %8s %8s | %8s %8s %8s %8s"
              % ("model", "dLL m2", "m6", "m10", "m20",
                 "ov m2", "m6", "m10", "m20"), flush=True)
        for mdl in models:
            d = [_mean(rows, mdl, regime, m, "delta_ll") for m in (2, 6, 10, 20)]
            o = [_mean(rows, mdl, regime, m, "overlap") for m in (2, 6, 10, 20)]
            print("  %-22s %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f %8.3f"
                  % (mdl.replace("esm2_", "").replace("_UR50D", ""),
                     d[0], d[1], d[2], d[3], o[0], o[1], o[2], o[3]), flush=True)
        if regime == "In-class":
            print("  (realized m saturates at 15/20 for this builder, so a dLL plateau here is",
                  flush=True)
            print("   partly the permutation builder and is NOT evidence for saturation.)",
                  flush=True)

    print("\n" + "-" * 78, flush=True)
    print("REGISTERED TEST, random arm, per scale", flush=True)
    print("  P1 saturation:        dLL(m20) - dLL(m6)      <= %+.2f" % P1_MAX_DLL_RISE,
          flush=True)
    print("  P2 continued collapse: overlap(m6) - overlap(m20) >= %.2f" % P2_MIN_OVERLAP_FALL,
          flush=True)
    print("  %-22s %12s %8s %14s %8s %10s"
          % ("model", "dLL rise", "P1", "overlap fall", "P2", "replicates"), flush=True)

    p1_fail = 0
    n_replicate = 0
    for mdl in models:
        rise = _mean(rows, mdl, "random", 20, "delta_ll") - _mean(rows, mdl, "random", 6,
                                                                  "delta_ll")
        fall = _mean(rows, mdl, "random", 6, "overlap") - _mean(rows, mdl, "random", 20,
                                                                "overlap")
        p1 = rise <= P1_MAX_DLL_RISE
        p2 = fall >= P2_MIN_OVERLAP_FALL
        if not p1:
            p1_fail += 1
        if p1 and p2:
            n_replicate += 1
        print("  %-22s %12.4f %8s %14.4f %8s %10s"
              % (mdl.replace("esm2_", "").replace("_UR50D", ""),
                 rise, "PASS" if p1 else "FAIL", fall, "PASS" if p2 else "FAIL",
                 "YES" if (p1 and p2) else "no"), flush=True)

    print("\nVERDICT:", flush=True)
    if p1_fail >= REFUTE_IF_P1_FAILS_AT_LEAST:
        print("  REFUTED. P1 failed at %d of %d scales. C14's saturation was specific to the"
              % (p1_fail, len(models)), flush=True)
        print("  650M model and/or its protein set. Per Amendment 15 the exploratory paragraph",
              flush=True)
        print("  is REMOVED from the paper's section 4.4 -- not softened -- and C14 reverts to",
              flush=True)
        print("  reporting only its ambiguous registered result.", flush=True)
    elif n_replicate == len(models):
        print("  REPLICATED AT EVERY SCALE (%d/%d). Perceived unfamiliarity stops increasing"
              % (n_replicate, len(models)), flush=True)
        print("  while the prediction keeps degrading, on a protein set that shares only one",
              flush=True)
        print("  member with C14's and across an 80x parameter range. The observation is no",
              flush=True)
        print("  longer exploratory and can be stated as a registered finding.", flush=True)
    else:
        print("  PARTIAL: %d of %d scales replicate the conjunction; P1 failed at %d."
              % (n_replicate, len(models), p1_fail), flush=True)
        print("  Below the refutation bar, above nothing. Report per-scale, claim only the",
              flush=True)
        print("  scales that pass, and say plainly which did not.", flush=True)

    print("\nNOTE: dLL is NOT comparable across models (different supports, temperatures,",
          flush=True)
    print("corpora). Every comparison above is a model against ITSELF at two values of m.",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())

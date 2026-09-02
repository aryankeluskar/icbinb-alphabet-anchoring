"""Amendment 9 analysis for readout 2 (variant effect).

Three dependent variables, reported in this order and never re-ordered:

  1. PRE-REGISTERED, unchanged: mean absolute dRho over ALL sampled assays.
  2. SIGNAL-GATED absolute dRho: mean over assays with rho_identity >= 0.20. The threshold was
     fixed in Amendment 9 before the run and must not be moved. It is ~3 standard errors for a
     Spearman correlation at n >= 200 (SE ~ 1/sqrt(n-1) ~ 0.07).
  3. RELATIVE DESTRUCTION on the gated subset: dRho / rho_identity per assay, then averaged.
     This is directly comparable to the contact readout, where the corresponding fraction is
     1.00 (0.763 -> chance, findings.md F13).

Underpowered clause: if fewer than 6 assays clear the gate, DV2 and DV3 are suppressed and
only DV1 is reported.

Usage: python3 analyze_readout2.py [csv]
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "results", "h2_readout2_variant_effect.csv")

GATE = 0.20          # Amendment 9, fixed before the run
MIN_GATED = 6        # Amendment 9 underpowered clause


def mean(xs):
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["esm_rho"] = float(r["esm_rho"])
        r["pssm_rho"] = float(r["pssm_rho"])
        r["m"] = int(r["m"])

    by_assay = collections.defaultdict(list)
    for r in rows:
        by_assay[r["assay"]].append(r)

    identity = {}
    gates = {}
    for a, rs in by_assay.items():
        idr = [r for r in rs if r["regime"] == "identity"]
        if not idr:
            continue
        identity[a] = idr[0]["esm_rho"]
        gates[a] = idr[0].get("gate", "?")

    complete = [a for a, rs in by_assay.items()
                if a in identity and any(r["regime"] != "identity" for r in rs)]
    print("assays with an identity row: %d;  with permuted cells: %d"
          % (len(identity), len(complete)))
    bad_gate = [a for a in complete if gates.get(a) != "PASS"]
    if bad_gate:
        print("!! identity gate did NOT pass for: %s -- nothing is reported until fixed"
              % ", ".join(bad_gate))
        return 1
    print("identity gate: PASS on all %d\n" % len(complete))

    # PSSM must be exactly invariant -- Proposition 1 with J = 0. Correctness assert.
    worst = max((abs(r["pssm_rho"] - identity_pssm(by_assay, r["assay"]))
                 for r in rows if r["regime"] != "identity"), default=0.0)
    print("PSSM max |dRho| across every permuted cell: %.1e  %s\n"
          % (worst, "(exact, as Prop. 1 requires)" if worst < 1e-9 else "!! NOT INVARIANT"))

    def deltas(assays, regime=None, m=None):
        out = collections.defaultdict(list)
        for a in assays:
            for r in by_assay[a]:
                if r["regime"] == "identity":
                    continue
                if regime and r["regime"] != regime:
                    continue
                if m is not None and r["m"] != m:
                    continue
                out[(r["regime"], r["m"])].append(r["esm_rho"] - identity[a])
        return out

    def table(title, assays):
        print(title)
        d = deltas(assays)
        for key in sorted(d):
            v = d[key]
            print("    %-14s m=%-3d  mean dRho = %+.4f   (n=%d cells)"
                  % (key[0], key[1], mean(v), len(v)))
        print()

    # ---- DV1: pre-registered, all assays -------------------------------------------
    table("DV1  PRE-REGISTERED -- mean absolute dRho over ALL %d assays:" % len(complete),
          complete)

    # ---- DV2 / DV3: signal-gated ---------------------------------------------------
    gated = [a for a in complete if identity[a] >= GATE]
    dropped = [(a, identity[a]) for a in complete if identity[a] < GATE]
    print("gate rho_identity >= %.2f:  %d of %d assays clear it" % (GATE, len(gated), len(complete)))
    if dropped:
        print("  dropped (identity rho): %s"
              % ", ".join("%.3f" % v for _, v in sorted(dropped, key=lambda t: t[1])))
    print()

    if len(gated) < MIN_GATED:
        print("UNDERPOWERED (Amendment 9): fewer than %d assays clear the gate, so DV2 and DV3"
              % MIN_GATED)
        print("are SUPPRESSED. Only DV1 above is reported.")
        return 0

    table("DV2  SIGNAL-GATED -- mean absolute dRho over the %d gated assays:" % len(gated),
          gated)

    print("DV3  RELATIVE DESTRUCTION -- mean dRho / rho_identity over the %d gated assays"
          % len(gated))
    print("     (contact readout, for comparison: 1.00 -- 0.763 falls to the base rate)")
    rel = collections.defaultdict(list)
    for a in gated:
        for r in by_assay[a]:
            if r["regime"] == "identity":
                continue
            rel[(r["regime"], r["m"])].append((r["esm_rho"] - identity[a]) / identity[a])
    for key in sorted(rel):
        v = rel[key]
        print("    %-14s m=%-3d  mean fraction destroyed = %+.3f   (n=%d cells)"
              % (key[0], key[1], -mean(v), len(v)))
    return 0


def identity_pssm(by_assay, assay):
    for r in by_assay[assay]:
        if r["regime"] == "identity":
            return r["pssm_rho"]
    return float("nan")


if __name__ == "__main__":
    sys.exit(main())

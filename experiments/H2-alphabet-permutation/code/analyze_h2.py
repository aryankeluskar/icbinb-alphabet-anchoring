"""Analysis + figure for H2.

Produces the paper's core panel: contact-set self-consistency as a function of how many
amino-acid types are relabeled, with the Potts reference line at 1.000 (exact, verified to
1e-15) and the chance line at the bottom.

The scientifically interesting comparison is not ESM-2 vs Potts -- that is partly rigged,
since Potts refits on the permuted alignment while ESM-2 gets no relabeled training data
(see protocol Amendment 5). It is In-class vs RANDOM at matched m: same number of
residue types relabeled, but one preserves biochemistry and the other does not. That
contrast is internal to ESM-2 and carries no such asymmetry.

Reads results/h2_esm2_invariance.csv, and results/h2_c6_matched_hamming.csv if present.
Writes h2_invariance.png/.pdf and prints a markdown summary table.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def agg(rows, keyfn, valfn):
    """key -> (mean, sem, n) with the SEM taken over PROTEINS, not over rows.

    Averaging over rows would treat 3 seeds on one protein as 3 independent observations
    and understate the error. We collapse to a per-protein mean first.
    """
    per_protein = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_protein[keyfn(r)][r["protein"]].append(valfn(r))
    out = {}
    for key, prots in per_protein.items():
        means = np.array([np.mean(v) for v in prots.values()])
        n = len(means)
        sem = float(means.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[key] = (float(means.mean()), sem, n)
    return out


def main():
    main_rows = load(os.path.join(RESULTS, "h2_esm2_invariance.csv"))
    c6_rows = load(os.path.join(RESULTS, "h2_c6_matched_hamming.csv"))
    if not main_rows:
        print("no main results yet")
        return 1

    for r in main_rows:
        r["m"] = int(r["m"])
        r["overlap"] = float(r["overlap"])
        r["spearman"] = float(r["spearman"])
        r["chance"] = float(r["chance"])

    regimes = ["random", "In-class", "Cross-class"]
    ms = sorted({r["m"] for r in main_rows if r["regime"] != "identity"})
    chance = float(np.mean([r["chance"] for r in main_rows]))
    n_prot = len({r["protein"] for r in main_rows})

    stats = agg([r for r in main_rows if r["regime"] != "identity"],
                lambda r: (r["regime"], r["m"]), lambda r: r["overlap"])

    print("\n# H2 -- ESM-2 contact-map self-consistency under alphabet relabeling")
    print("\nProteins: %d.  Chance overlap: %.3f.  Potts reference: 1.000 (exact).\n"
          % (n_prot, chance))
    header = "| m | " + " | ".join(regimes) + " |"
    print(header)
    print("|---|" + "---|" * len(regimes))
    for m in ms:
        cells = []
        for reg in regimes:
            if (reg, m) in stats:
                mu, sem, n = stats[(reg, m)]
                cells.append("%.3f ± %.3f" % (mu, sem) if n > 1 else "%.3f" % mu)
            else:
                cells.append("—")
        print("| %d | %s |" % (m, " | ".join(cells)))

    # The headline contrast, computed explicitly.
    print("\n## In-class vs random at matched m (the contrast that is not confounded)")
    for m in ms:
        if ("random", m) in stats and ("In-class", m) in stats:
            r_mu = stats[("random", m)][0]
            c_mu = stats[("In-class", m)][0]
            print("  m=%-3d random %.3f   In-class %.3f   ratio %.1fx"
                  % (m, r_mu, c_mu, c_mu / r_mu if r_mu > 1e-9 else float("inf")))

    if c6_rows:
        for r in c6_rows:
            r["m"] = int(r["m"])
            r["overlap"] = float(r["overlap"])
        print("\n## C6 -- matched-Hamming control (does structure of change matter?)")
        c6 = agg(c6_rows, lambda r: (r["regime"], r["m"], r["arm"]),
                 lambda r: r["overlap"])
        print("| regime | m | permuted | uniform | composition | verdict |")
        print("|---|---|---|---|---|---|")
        for reg in regimes:
            for m in ms:
                got = [(reg, m, a) in c6 for a in ("permuted", "uniform", "composition")]
                if not all(got):
                    continue
                p = c6[(reg, m, "permuted")][0]
                u = c6[(reg, m, "uniform")][0]
                cp = c6[(reg, m, "composition")][0]
                if abs(p - u) < 0.05:
                    v = "edit-distance effect"
                elif p > u:
                    v = "permutation SPARED"
                else:
                    v = "permutation WORSE"
                print("| %s | %d | %.3f | %.3f | %.3f | %s |" % (reg, m, p, u, cp, v))
    else:
        print("\n## C6 -- not yet available. H2's m-sweep is NOT reportable without it.")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; skipped figure)")
        return 0

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = {"random": "#c0392b", "In-class": "#2471a3", "Cross-class": "#8e44ad"}
    for reg in regimes:
        xs, ys, es = [], [], []
        for m in ms:
            if (reg, m) in stats:
                mu, sem, _ = stats[(reg, m)]
                xs.append(m)
                ys.append(mu)
                es.append(0.0 if np.isnan(sem) else sem)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                        color=colors[reg], label=reg)

    ax.axhline(1.0, ls="--", lw=1.4, color="#1e8449",
               label="Potts / MRF (exact, verified 1e-15)")
    ax.axhline(chance, ls=":", lw=1.4, color="#7f8c8d", label="chance")
    ax.set_xlabel("number of amino-acid types relabeled (m)")
    ax.set_ylabel("top-L/2 contact-set overlap with wild-type")
    ax.set_title("ESM-2 650M contact prediction under a consistent\n"
                 "bijective relabeling of the amino-acid alphabet", fontsize=10)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks(ms)
    ax.legend(fontsize=7.5, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, "h2_invariance." + ext), dpi=200)
    print("\nwrote %s" % os.path.join(RESULTS, "h2_invariance.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

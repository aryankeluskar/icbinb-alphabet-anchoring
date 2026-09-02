"""Main figure, v2 -- three panels, one row, one page-width.

A  The instrument.  ESM-2 650M overlap vs m, three relabeling regimes.
B  Scale.           Overlap at m=20 across 8M/35M/150M/650M, random vs In-class.
C  The crossing.    ESM-2 above Potts at identity, below it after relabeling, on one
                    identical restricted candidate set.

Reads only committed CSVs in ../results/ and writes h2_main_figure_v2.{pdf,png} there.
Nothing else in the repo is touched.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv                                        # noqa: E402
import sys                                        # noqa: E402

import numpy as np                                # noqa: E402
import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.lines import Line2D               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

# Okabe-Ito, colorblind-safe. One color per regime, shared by panels A and B.
C_CONS = "#009E73"   # bluish green
C_RAND = "#0072B2"   # blue
C_RADI = "#D55E00"   # vermillion
C_ESM = "#CC79A7"    # reddish purple -- panel C only, a different semantic axis
C_POTTS = "#000000"
C_CHANCE = "0.45"

REGIME = {
    "In-class": (C_CONS, "o", "In-class"),
    "random": (C_RAND, "s", "random"),
    "Cross-class": (C_RADI, "^", "Cross-class"),
}
SIZES = [("esm2_t6_8M_UR50D", 8e6, "8M"),
         ("esm2_t12_35M_UR50D", 35e6, "35M"),
         ("esm2_t30_150M_UR50D", 150e6, "150M"),
         ("esm2_t33_650M_UR50D", 650e6, "650M")]

REPORT = []


def say(line=""):
    REPORT.append(line)
    print(line)


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return list(csv.DictReader(fh))


def seed_band(rows, valkey="overlap"):
    """(grand mean over rows, min, max, seeds, n_rows).

    The band is the min--max across permutation SEEDS of the seed's mean over
    proteins. Each seed is one draw of the relabeling; a seed that happens to move
    a well-tolerated set of residues moves every protein the same way, so the seed
    is the unit that the band is about.
    """
    seeds = sorted({r["seed"] for r in rows})
    per_seed = []
    for s in seeds:
        v = [float(r[valkey]) for r in rows if r["seed"] == s]
        per_seed.append(float(np.mean(v)))
    allv = [float(r[valkey]) for r in rows]
    return (float(np.mean(allv)), min(per_seed), max(per_seed), per_seed, len(allv))


def panel_label(ax, letter, text):
    ax.text(-0.30, 1.10, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")
    ax.text(-0.17, 1.10, text, transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left")


# --------------------------------------------------------------------------- A
def panel_a(ax):
    rows = [r for r in load("h2_esm2_invariance.csv")
            if r["model"] == "esm2_t33_650M_UR50D" and r["regime"] != "identity"]
    ms = [2, 6, 10, 20]
    say("PANEL A -- ESM-2 650M, 6 proteins, seeds 0/1/2, top-L/2 long-range overlap")
    say("  regime         m   n   mean    seed-min  seed-max   per-seed means")
    for regime in ("random", "In-class", "Cross-class"):
        col, mk, lab = REGIME[regime]
        xs, ys, lo, hi, single = [], [], [], [], []
        for m in ms:
            sub = [r for r in rows if r["regime"] == regime and int(r["m"]) == m]
            if not sub:
                continue
            mean, mn, mx, per, n = seed_band(sub)
            xs.append(m)
            ys.append(mean)
            lo.append(mn)
            hi.append(mx)
            single.append(len(per) == 1)
            say("  %-13s %2d  %2d  %.4f  %.4f    %.4f     %s"
                % (regime, m, n, mean, mn, mx, [round(p, 4) for p in per]))
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, ys, color=col, marker=mk, ms=3.4, lw=1.3, label=lab, zorder=4,
                clip_on=False)
        # A point backed by a single seed gets a hollow marker: no band is possible.
        sx = [x for x, s in zip(xs, single) if s]
        sy = [y for y, s in zip(ys, single) if s]
        if sx:
            ax.plot(sx, sy, marker=mk, ms=3.4, ls="none", mfc="white",
                    mec=col, mew=1.0, zorder=5, clip_on=False)

    chance = float(np.mean([float(r["chance"]) for r in rows]))
    say("  chance (mean over rows) = %.4f" % chance)
    ax.axhline(1.0, color=C_POTTS, ls="-", lw=1.1, zorder=3)
    ax.text(20.4, 0.965,
            "Potts: invariant by Proposition 2,\nnot by measurement",
            fontsize=5.6, va="top", ha="right", color=C_POTTS, linespacing=1.25)
    ax.axhline(chance, color=C_CHANCE, ls="--", lw=0.9, zorder=1)
    ax.text(2.0, chance + 0.012, "chance", fontsize=6, color=C_CHANCE,
            va="bottom", ha="left")

    ax.set_xticks(ms)
    ax.set_xlim(1.2, 20.8)
    ax.set_ylim(-0.03, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("$m$ residue types relabeled")
    ax.set_ylabel("top-$L/2$ long-range\ncontact-set overlap")
    ax.legend(fontsize=6, loc="center right", frameon=False, handlelength=1.6,
              borderpad=0.1, labelspacing=0.3, bbox_to_anchor=(1.03, 0.40))
    panel_label(ax, "A", "one bijective relabeling, ESM-2 650M")


# --------------------------------------------------------------------------- B
def panel_b(ax):
    rows = [r for r in load("h2_c8_scale.csv") if int(r["m"]) == 20]
    say("")
    say("PANEL B -- m=20, 4 proteins, seeds 0/1, top-L/2 long-range overlap")
    say("  regime         model   n   mean    seed-min  seed-max   per-seed means")
    for regime in ("In-class", "random"):
        col, mk, lab = REGIME[regime]
        xs, ys, lo, hi = [], [], [], []
        for model, params, short in SIZES:
            sub = [r for r in rows if r["model"] == model and r["regime"] == regime]
            if not sub:
                continue
            mean, mn, mx, per, n = seed_band(sub)
            xs.append(params)
            ys.append(mean)
            lo.append(mn)
            hi.append(mx)
            say("  %-13s %5s   %d  %.4f  %.4f    %.4f     %s"
                % (regime, short, n, mean, mn, mx, [round(p, 4) for p in per]))
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, ys, color=col, marker=mk, ms=3.4, lw=1.3, label=lab, zorder=4)

    chance = float(np.mean([float(r["chance"]) for r in rows]))
    say("  chance (mean over rows) = %.4f" % chance)
    ax.axhline(chance, color=C_CHANCE, ls="--", lw=0.9, zorder=1)
    ax.text(8.6e6, chance + 0.012, "chance", fontsize=6, color=C_CHANCE,
            va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xticks([p for _, p, _ in SIZES])
    ax.set_xticklabels([s for _, _, s in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(6e6, 9.5e8)
    ax.set_ylim(-0.025, 0.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("ESM-2 parameters")
    ax.set_ylabel("overlap at $m=20$")
    ax.legend(fontsize=6, loc="upper left", frameon=False, handlelength=1.6,
              borderpad=0.1, labelspacing=0.25)
    panel_label(ax, "B", "scale does not restore invariance")


# --------------------------------------------------------------------------- C
def panel_c(ax):
    rows = load("h2_c10_potts_crossing.csv")
    # `identity-fulllen` is the C9 cross-check on the unrestricted sequence, not a
    # point on the restricted candidate set this panel is about. Drop it.
    rows = [r for r in rows if r["regime"] in ("identity", "random")]
    targets = []
    for r in rows:
        if r["target"] not in targets:
            targets.append(r["target"])
    pretty = {"ubiquitin": "ubiquitin", "protein_G_B1": "protein G B1"}
    dash = {targets[0]: (), targets[1]: (2.6, 1.4)}

    say("")
    say("PANEL C -- restricted candidate set, long-range P@L/2 (n_eval per CSV)")
    say("  target        model  identity   permuted(m=20)  seeds_permuted   chance")
    for t in targets:
        for model, col in (("esm2", C_ESM), ("potts", C_POTTS)):
            ident = [float(r["precision"]) for r in rows
                     if r["target"] == t and r["model"] == model
                     and r["regime"] == "identity"]
            perm = [float(r["precision"]) for r in rows
                    if r["target"] == t and r["model"] == model
                    and r["regime"] == "random"]
            ch = float([r["chance"] for r in rows if r["target"] == t][0])
            y0, y1 = float(np.mean(ident)), float(np.mean(perm))
            say("  %-13s %-5s  %.4f     %.4f          %s   %.4f"
                % (t, model, y0, y1, [round(p, 4) for p in perm], ch))
            ax.plot([0, 1], [y0, y1], color=col, lw=1.4, dashes=dash[t],
                    marker="o" if model == "esm2" else "s", ms=3.6, zorder=4,
                    clip_on=False)
        # per-protein chance, drawn faintly -- the permuted ESM-2 point lands on it
        ch = float([r["chance"] for r in rows if r["target"] == t][0])
        ax.plot([-0.12, 1.12], [ch, ch], color=C_CHANCE, ls=":", lw=0.8, zorder=1)

    ax.text(-0.10, float([r["chance"] for r in rows
                          if r["target"] == targets[1]][0]) + 0.015,
            "chance", fontsize=6, color=C_CHANCE, va="bottom", ha="left")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["identity", "relabeled\n($m=20$, random)"])
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(0.0, 0.92)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("long-range P@$L/2$\n(identical candidate set)")
    handles = [Line2D([], [], color=C_ESM, marker="o", ms=3.6, lw=1.4,
                      label="ESM-2 650M"),
               Line2D([], [], color=C_POTTS, marker="s", ms=3.6, lw=1.4,
                      label="Potts (measured)"),
               Line2D([], [], color="0.35", lw=1.2, label=pretty[targets[0]]),
               Line2D([], [], color="0.35", lw=1.2, dashes=dash[targets[1]],
                      label=pretty[targets[1]])]
    ax.legend(handles=handles, fontsize=6, loc="upper right", frameon=False,
              handlelength=1.6, borderpad=0.1, labelspacing=0.3,
              bbox_to_anchor=(1.05, 1.04))
    panel_label(ax, "C", "the crossing")


def main():
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.4)
    out = []
    for ext in ("pdf", "png"):
        p = os.path.join(RESULTS, "h2_main_figure_v2.%s" % ext)
        fig.savefig(p, dpi=300)
        out.append(p)
    say("")
    for p in out:
        say("wrote %s (%d bytes)" % (p, os.path.getsize(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

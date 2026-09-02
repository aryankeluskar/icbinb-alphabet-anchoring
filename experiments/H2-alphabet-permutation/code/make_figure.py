"""The paper's core figure. Renders whatever has finished; skips panels with no data.

Three panels, chosen so that each one answers an objection the previous one invites:

  A  ESM-2 650M, single sequence. The headline collapse -- and the In-class-vs-random
     contrast that shows the failure is of the alphabet-abstract algorithm, not of
     biochemistry.
  B  MSA Transformer, alignment-conditioned. Answers "your baseline refits on the permuted
     data and your model does not" (protocol Amendment 5). MSA-T reads the same alignment
     at inference, so the asymmetry is gone, and the Potts curve here is MEASURED on that
     alignment rather than assumed.
  C  Scale. Answers "does it go away at 3B?" (Amendment 6).

Every panel plots each model against ITSELF -- self-consistency under a transformation that
provably preserves all alignment statistics -- so nothing here is confounded by bigger
models simply being better at contact prediction.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

import permute            # noqa: E402


def realized_m(regime, m, seed):
    """How many residue types the permutation ACTUALLY moved (protocol Amendment 7).

    Recomputed from the same seed formula the runs used, so this is not an estimate --
    it is the permutation that was applied. In-class saturates at 15 of 20 and
    Cross-class at ~15, because A, C, G, H and P have no biochemically similar partner: a
    fully In-class relabeling of the whole alphabet does not exist.
    """
    if m == 0:
        return 0
    rng = np.random.default_rng(1000 * int(seed) + int(m))
    perm = permute.PERM_BUILDERS[regime](int(m), rng)
    return sum(1 for a, b in perm.items() if a != b)

REGIME_STYLE = {
    "In-class": ("#1b7837", "o", "In-class"),
    "random":       ("#2166ac", "s", "random"),
    "Cross-class":      ("#b2182b", "^", "Cross-class"),
}
# Parameter counts for the x axis of panel C.
MODEL_PARAMS = {
    "esm2_t6_8M_UR50D": 8e6,
    "esm2_t12_35M_UR50D": 35e6,
    "esm2_t30_150M_UR50D": 150e6,
    "esm2_t33_650M_UR50D": 650e6,
    "esm2_t36_3B_UR50D": 3.0e9,
}
MODEL_SHORT = {k: k.split("_")[2] for k in MODEL_PARAMS}


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def by_unit(rows, keyfn, valfn, unitfn):
    """key -> (mean, sem, n_units), collapsing to one value per unit FIRST.

    The unit is the protein or family, not the row. Three seeds on one protein are not
    three independent observations, and averaging over rows would understate the error.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = valfn(r)
        if v is not None and np.isfinite(v):
            acc[keyfn(r)][unitfn(r)].append(v)
    out = {}
    for key, units in acc.items():
        means = np.array([np.mean(v) for v in units.values()])
        n = len(means)
        sem = float(means.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[key] = (float(means.mean()), sem, n)
    return out


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def draw_curves(ax, stats, m_values, ylabel, title, chance=None,
                potts=None, potts_label="Potts (exact, 1.000)", seeds=(0, 1, 2)):
    ticks = set()
    for regime, (colour, marker, label) in REGIME_STYLE.items():
        xs, ys, es = [], [], []
        for m in m_values:
            if (regime, m) in stats:
                mean, sem, _ = stats[(regime, m)]
                # Amendment 7: plot at the m that was DELIVERED, not the m requested.
                x = float(np.mean([realized_m(regime, m, s) for s in seeds]))
                xs.append(x)
                ys.append(mean)
                es.append(sem)
                ticks.add(round(x))
        if xs:
            order = np.argsort(xs)
            ax.errorbar(np.array(xs)[order], np.array(ys)[order],
                        yerr=np.array(es)[order], color=colour, marker=marker, ms=5,
                        lw=1.8, capsize=3, label=label, zorder=3)
    span = [min(m_values), 20]
    if potts is not None:
        ax.plot(span, [potts] * 2, color="k", ls="--", lw=1.4,
                label=potts_label, zorder=2)
    if chance is not None:
        ax.axhline(chance, color="0.55", ls=":", lw=1.2, zorder=1)
        ax.text(20, chance, " chance", va="bottom", ha="right",
                fontsize=7, color="0.4")
    ax.set_xlim(0, 21)
    m_values = sorted(ticks)
    ax.set_xlabel("residue types actually relabeled")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(m_values)
    ax.grid(alpha=0.25, lw=0.5)


def panel_a(ax):
    rows = [r for r in load("h2_esm2_invariance.csv") if r["regime"] != "identity"]
    if not rows:
        return False
    m_values = sorted({int(r["m"]) for r in rows})
    stats = by_unit(rows, lambda r: (r["regime"], int(r["m"])),
                    lambda r: fnum(r, "overlap"), lambda r: r["protein"])
    chance = float(np.mean([fnum(r, "chance") for r in rows]))
    n_prot = len({r["protein"] for r in rows})
    draw_curves(ax, stats, m_values, "contact-set overlap with own unpermuted map",
                "A  ESM-2 650M (single sequence)\n%d proteins" % n_prot,
                chance=chance, potts=1.0)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    return True


def panel_b(ax):
    rows = load("h2_c7_msa_transformer.csv")
    if not rows:
        return False
    m_values = sorted({int(r["m"]) for r in rows})
    stats = by_unit(rows, lambda r: (r["regime"], int(r["m"])),
                    lambda r: fnum(r, "msa_transformer"), lambda r: r["family"])
    potts = by_unit(rows, lambda r: (r["regime"], int(r["m"])),
                    lambda r: fnum(r, "potts"), lambda r: r["family"])
    # The Potts line in this panel is MEASURED, not assumed. Report the worst cell so a
    # reader can see it really is exact rather than merely close.
    worst = min(v[0] for v in potts.values())
    n_fam = len({r["family"] for r in rows})
    draw_curves(ax, stats, m_values, "contact-set overlap with own unpermuted map",
                "B  MSA Transformer (reads the alignment)\n%d families" % n_fam,
                potts=worst,
                potts_label="Potts, same alignment (measured, %.3f)" % worst)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    return True


def panel_c(ax):
    rows = load("h2_c8_scale.csv")
    if not rows:
        return False
    # m >= 10: the regime where the alphabet is substantially rewritten. Averaging the two
    # deepest cells is less noisy than reading a single m off the curve.
    deep = [r for r in rows if int(r["m"]) >= 10]
    if not deep:
        return False
    stats = by_unit(deep, lambda r: (r["regime"], r["model"]),
                    lambda r: fnum(r, "overlap"), lambda r: r["protein"])
    models = [m for m in MODEL_PARAMS if any(k[1] == m for k in stats)]
    models.sort(key=lambda m: MODEL_PARAMS[m])
    if len(models) < 2:
        return False

    for regime in ("In-class", "random"):
        colour, marker, label = REGIME_STYLE[regime]
        xs = [MODEL_PARAMS[m] for m in models if (regime, m) in stats]
        ys = [stats[(regime, m)][0] for m in models if (regime, m) in stats]
        es = [stats[(regime, m)][1] for m in models if (regime, m) in stats]
        ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, ms=5, lw=1.8,
                    capsize=3, label=label, zorder=3)

    chance = float(np.mean([fnum(r, "chance") for r in deep]))
    ax.axhline(1.0, color="k", ls="--", lw=1.4, label="Potts (exact)", zorder=2)
    ax.axhline(chance, color="0.55", ls=":", lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([MODEL_PARAMS[m] for m in models])
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=8)
    # A log axis defaults to decade minor ticks (2e7, 3e7...) that mean nothing here --
    # the only meaningful x positions are the checkpoints we actually ran.
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(MODEL_PARAMS[models[0]] / 1.8, MODEL_PARAMS[models[-1]] * 1.8)
    ax.set_xlabel("ESM-2 parameters")
    ax.set_ylabel("contact-set overlap, deep relabeling")
    ax.set_title("C  Scale does not buy alphabet abstraction\n%d sizes" % len(models),
                 fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    return True


def main():
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    drawn = []
    for ax, fn, name in zip(axes, (panel_a, panel_b, panel_c), ("A", "B", "C")):
        if fn(ax):
            drawn.append(name)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "panel %s\nno data yet" % name, ha="center",
                    va="center", fontsize=9, color="0.6", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(RESULTS, "h2_main_figure.%s" % ext)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    print("panels drawn: %s" % (", ".join(drawn) if drawn else "none"))
    print("wrote %s/h2_main_figure.{png,pdf}" % RESULTS)
    return 0 if drawn else 1


if __name__ == "__main__":
    sys.exit(main())

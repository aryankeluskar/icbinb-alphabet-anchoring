"""C12 -- dose-response between the biochemical cost of a relabeling and contact-map damage.

Registered as protocol Amendment 12 before running.

Our causal account says ESM-2's contact map is produced by residue-identity-conditioned
pattern matching rather than column-statistic inference. The biochemistry half of that account
currently rests on THREE ORDINAL LEVELS -- In-class < random < Cross-class. Three points is a
contrast, not a mechanism.

If the account is right, damage is a continuous function of how far the relabeling moves each
residue in biochemical space. C12 measures that function.

**Intervention size is held constant.** Every permutation here is a full derangement of all 20
residues, so every condition changes the same number of sequence positions. Only the
biochemical character varies. This is C6's matched-Hamming logic extended from two points to a
continuum, and it is what makes a correlation here interpretable as biochemistry rather than
as disruption magnitude.

BLOSUM62 and the `severity` sign convention are reused from H4 rather than re-entered, with
H4's own sign assert run before anything else.

Writes results/h2_c12_dose_response.csv incrementally after every permutation, so a
preemption leaves a usable partial curve.
"""
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "H4-format-severity", "code"))

import jacobian as jac_mod       # noqa: E402
import permute                   # noqa: E402
from run_h2_esm2 import EXTRA, spearman        # noqa: E402
from run_h4 import check_c4, severity          # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
MODEL = "esm2_t33_650M_UR50D"
BATCH = 8
N_PERMS = 60
N_BINS = 12                  # cost-stratified sampling: ~5 kept per bin
POOL = 4000                  # candidate derangements to sample before stratifying
TARGETS = ["ubiquitin", "protein_G_B1"]


def perm_cost(perm, seq):
    """Sequence-weighted mean severity, and the unweighted 20-letter mean.

    Weighted is primary (Amendment 12): damage should track the residues the model actually
    sees. Unweighted is the registered confound check -- a permutation could score low simply
    by sparing common residues while mangling rare ones.
    """
    w = [severity(c, perm[c]) for c in seq if c in permute.AA20_SET]
    u = [severity(a, perm[a]) for a in permute.AA20]
    return float(np.mean(w)), float(np.mean(u))


def _anneal(seq, rng, sign, steps=4000):
    """Search for an EXTREME-cost full derangement (Amendment 12a(b)).

    Random derangements span only the middle of the cost axis; the In-class anchor sits
    well below anything they reach. Without this search the low-cost end of the dose-response
    would be tested only by permutations whose realized m is 15, reintroducing the very
    confound the constant-m design removes.

    Metropolis on a permutation of the 20 images, rejecting any move that creates a fixed
    point, so every state visited is a full derangement. `sign` = +1 minimises cost,
    -1 maximises it.
    """
    letters = list(permute.AA20)
    cur = permute._derangement(letters, rng)
    cost = sign * perm_cost(cur, seq)[0]
    temp0 = 1.0
    for t in range(steps):
        a, b = rng.choice(len(letters), size=2, replace=False)
        la, lb = letters[a], letters[b]
        if cur[la] == lb or cur[lb] == la:      # the swap would create a fixed point
            continue
        cand = dict(cur)
        cand[la], cand[lb] = cur[lb], cur[la]
        c = sign * perm_cost(cand, seq)[0]
        temp = temp0 * (1.0 - t / float(steps)) + 1e-3
        if c < cost or rng.random() < np.exp(-(c - cost) / temp):
            cur, cost = cand, c
    assert all(a != b for a, b in cur.items()), "annealing broke the derangement constraint"
    return cur


def sample_stratified(seq, rng, n_anneal=25):
    """Full derangements of all 20 residues, spread across the achievable cost range.

    Random derangements cluster near the mode of the cost distribution. Binning and sampling
    within bins buys coverage of the tails, which is where the dose-response is actually
    tested -- an unstratified sample would concentrate the design points in the middle and
    leave the prediction barely constrained at the ends. The annealed extremes extend the
    range further in both directions while keeping realized m = 20 throughout.
    """
    cands = []
    for _ in range(POOL):
        p = permute._derangement(list(permute.AA20), rng)
        cands.append((perm_cost(p, seq)[0], p))
    for i in range(n_anneal):
        p = _anneal(seq, rng, sign=+1 if i % 2 == 0 else -1)
        cands.append((perm_cost(p, seq)[0], p))
    cands.sort(key=lambda t: t[0])
    lo, hi = cands[0][0], cands[-1][0]
    edges = np.linspace(lo, hi, N_BINS + 1)
    per_bin = max(1, N_PERMS // N_BINS)
    picked = []
    for b in range(N_BINS):
        inbin = [c for c in cands if edges[b] <= c[0] <= edges[b + 1]]
        if not inbin:
            continue
        idx = rng.choice(len(inbin), size=min(per_bin, len(inbin)), replace=False)
        picked.extend(inbin[i] for i in idx)
    return [p for _, p in picked]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(HERE, "..", "results", "h2_c12_dose_response.csv")

    cons, rad = check_c4()
    print("C12 dose-response  device=%s  model=%s" % (device, MODEL), flush=True)
    print("BLOSUM sign assert OK: sev(K->R)=%.0f < sev(K->W)=%.0f\n" % (cons, rad), flush=True)

    import esm
    m, alphabet = getattr(esm.pretrained, MODEL)()
    wrap = jac_mod.ESM2Jacobian(m, alphabet, device=device, batch_size=BATCH)
    print("loaded %s\n" % MODEL, flush=True)

    rows = []
    for pid in TARGETS:
        seq = EXTRA[pid]
        length = len(seq)
        rng = np.random.default_rng(20260805)

        perms = [("random_%03d" % i, p) for i, p in enumerate(sample_stratified(seq, rng))]
        # Labelled anchors, so the curve can be read against the three-level contrast it
        # replaces. These are NOT extra design points -- they are the old design, plotted.
        for lbl, builder in (("In-class", "In-class"), ("Cross-class", "Cross-class")):
            for s in (0, 1):
                r2 = np.random.default_rng(1000 * s + 20)
                perms.append(("%s_s%d" % (lbl, s), permute.PERM_BUILDERS[builder](20, r2)))

        base = wrap.contact_map(seq)
        base_top = jac_mod.top_contacts(base, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        n_pairs = sum(max(0, length - MIN_SEP - i) for i in range(length))
        chance = len(base_top) / float(n_pairs)
        print("%s L=%d  chance=%.4f  %d permutations" % (pid, length, chance, len(perms)),
              flush=True)

        t0 = time.time()
        for k, (lbl, perm) in enumerate(perms):
            cw, cu = perm_cost(perm, seq)
            realized = sum(1 for a, b in perm.items() if a != b)
            ps = wrap.contact_map(permute.apply_perm(seq, perm))
            ov = jac_mod.set_overlap(
                base_top, jac_mod.top_contacts(ps, min_sep=MIN_SEP, top_frac=TOP_FRAC))
            rows.append(dict(protein=pid, length=length, label=lbl, m_realized=realized,
                             cost_weighted=round(cw, 5), cost_unweighted=round(cu, 5),
                             overlap=round(ov, 5), chance=round(chance, 5)))
            with open(out_path, "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            if (k + 1) % 10 == 0 or k == len(perms) - 1:
                print("   %3d/%d  (%.0fs elapsed)" % (k + 1, len(perms), time.time() - t0),
                      flush=True)

        # ---- per-protein readout, computed here so a preemption still reports something ----
        # Amendment 12a(a): rho is over the m=20 derangements ONLY. The In-class and
        # Cross-class anchors have realized m of 15 and 16, so including them would let
        # intervention size vary with cost -- the confound this design exists to remove.
        mine = [r for r in rows if r["protein"] == pid and r["m_realized"] == 20]
        anchors = [r for r in rows if r["protein"] == pid and r["m_realized"] != 20]
        cw = np.array([r["cost_weighted"] for r in mine])
        cu = np.array([r["cost_unweighted"] for r in mine])
        ov = np.array([r["overlap"] for r in mine])
        print("  rho over %d full derangements (m=20); %d anchors excluded (m=%s)"
              % (len(mine), len(anchors),
                 ",".join(str(r["m_realized"]) for r in anchors) or "-"), flush=True)
        rho_w, rho_u = spearman(cw, ov), spearman(cu, ov)
        verdict = ("CONFIRMED" if rho_w <= -0.6 else
                   "REFUTED (|rho| < 0.3)" if abs(rho_w) < 0.3 else "WEAK (-0.6 < rho < -0.3)")
        print("  cost range %.2f to %.2f   overlap range %.3f to %.3f"
              % (cw.min(), cw.max(), ov.min(), ov.max()), flush=True)
        print("  SPEARMAN rho(weighted cost, overlap)   = %+.3f   %s" % (rho_w, verdict),
              flush=True)
        print("  SPEARMAN rho(unweighted cost, overlap) = %+.3f   %s" % (
            rho_u, "agrees" if abs(rho_u - rho_w) < 0.15 else "DISAGREES -- report both"),
            flush=True)
        for lbl in ("In-class", "Cross-class"):
            anc = [r for r in mine if r["label"].startswith(lbl)]
            if anc:
                print("  anchor %-13s cost %+.2f  overlap %.3f"
                      % (lbl, np.mean([r["cost_weighted"] for r in anc]),
                         np.mean([r["overlap"] for r in anc])), flush=True)
        print("", flush=True)

    print("wrote %s (%d rows)" % (out_path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

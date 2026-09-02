"""C10 -- does ESM-2's collapse cross the classical Potts baseline?

Registered as protocol Amendment 10 (ubiquitin) and 10a (protein G B1) before running.

C9 measured ESM-2 650M falling from P@L/2 = 0.763 to 0.053 (= chance) on 1UBQ under an m=20
random alphabet relabeling. Potts has a SINGLE precision number that holds under every
relabeling, because Proposition 2 makes its contact scores exactly invariant. If that number
sits above ESM-2's permuted number, the finding upgrades from "degrades to chance" to "falls
below the classical model it replaced".

Both models are scored on the IDENTICAL candidate pair set: query positions outside the MSA's
coverage are dropped for both by setting mapping[i] = -1, and k is set from the full query
length for both.

Writes results/h2_c10_potts_crossing.csv.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mrf                      # noqa: E402
import permute                  # noqa: E402
import structures               # noqa: E402

MSA_DIR = os.path.join(ROOT, "data", "proteingym", "DMS_msa_files", "DMS_msa_files")

# (name, query sequence, pdb id, chain, a2m filename)
TARGETS = [
    ("ubiquitin",
     "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
     "1UBQ", "A", "RL40A_YEAST_full_11-26-2021_b01.a2m"),
    ("protein_G_B1",
     "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
     "1PGB", "A", "SPG1_STRSG_full_b0.1.a2m"),
]

MIN_SEP = 12
TOP_FRAC = 0.5
M = 20
SEEDS = [0, 1]
MAX_GAP_FRAC = 0.5          # drop MSA rows more than half gaps in the covered region
MIN_REGION = 40             # a shorter overlap is not a domain
MIN_REGION_IDENTITY = 0.80  # Amendment 10a: below this the target is dropped, not patched
ESM_BATCH = int(os.environ.get("C10_ESM_BATCH", "16"))


def read_a2m(path):
    """a2m -> list of match-column strings (uppercase and '-' only)."""
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return ["".join(c for c in s if c.isupper() or c == "-") for s in seqs]


def locate_region(query_match, target):
    """Ungapped slide of the MSA query against the target sequence.

    Handles both directions: the MSA query may start partway into the target (ubiquitin, where
    the alignment covers residues 10-76) or the target may be a sub-range of the MSA query
    (protein G B1 inside full-length SPG1). Returns
    (q_start, q_end, t_start, identity) with query_match[q_start:q_end] aligned to
    target[t_start : t_start + (q_end - q_start)], 1:1 and ungapped.
    """
    nq, nt = len(query_match), len(target)
    best = None
    for s in range(-nq + 1, nt):
        lo = max(0, -s)                       # first query index in the overlap
        hi = min(nq, nt - s)                  # one past the last
        n = hi - lo
        if n < MIN_REGION:
            continue
        same = sum(1 for i in range(lo, hi) if query_match[i] == target[i + s])
        frac = same / float(n)
        if best is None or (frac, n) > (best[0], best[3] - best[2]):
            best = (frac, s, lo, hi)
    if best is None:
        return None
    _, s, lo, hi = best

    # Trim runs of >4 consecutive mismatches off each end -- that is where we leave the domain.
    def trim(a, b, step):
        i, misses, last_good = a, 0, a
        while (i < b) if step > 0 else (i > b):
            if query_match[i] == target[i + s]:
                misses, last_good = 0, i
            else:
                misses += 1
                if misses > 4:
                    return last_good
            i += step
        return last_good
    lo2 = trim(lo, hi, 1)
    hi2 = trim(hi - 1, lo - 1, -1) + 1
    if hi2 - lo2 < MIN_REGION:
        lo2, hi2 = lo, hi
    ident = sum(1 for i in range(lo2, hi2) if query_match[i] == target[i + s]) / float(hi2 - lo2)
    return lo2, hi2, lo2 + s, ident


def run_target(name, target, pdb_id, chain, a2m, wrap, rows):
    print("\n=== %s (%s_%s, L=%d) ===" % (name, pdb_id, chain, len(target)), flush=True)
    seqs = read_a2m(os.path.join(MSA_DIR, a2m))
    query = seqs[0]
    loc = locate_region(query, target)
    if loc is None:
        print("  no usable region found -- SKIPPED")
        return
    q0, q1, t0, ident = loc
    cov_len = q1 - q0
    print("  MSA depth %d, match cols %d" % (len(seqs), len(query)))
    print("  region: MSA cols %d-%d -> query residues %d-%d (%d aa), identity %.3f"
          % (q0, q1 - 1, t0 + 1, t0 + cov_len, cov_len, ident))
    if ident < MIN_REGION_IDENTITY:
        print("  identity %.3f < %.2f -- TARGET DROPPED (Amendment 10a)"
              % (ident, MIN_REGION_IDENTITY))
        return

    region = [s[q0:q1] for s in seqs]
    keep = [s for s in region if s.count("-") <= MAX_GAP_FRAC * cov_len]
    print("  rows kept after <=%.0f%% gap filter: %d of %d"
          % (100 * MAX_GAP_FRAC, len(keep), len(region)))

    path = structures.fetch_pdb(pdb_id)
    pdb_seq, coords = structures.ca_cb_coords(path, chain)
    cmap = structures.contact_map(coords)
    mapping, stats = structures.align_to_query(pdb_seq, target, return_stats=True)
    mapping = list(mapping)
    print("  %s_%s  coverage %.0f%%  identity %.0f%%"
          % (pdb_id, chain, 100 * stats["coverage"], 100 * stats["identity"]))

    restricted = list(mapping)
    for i in range(len(target)):
        if i < t0 or i >= t0 + cov_len:
            restricted[i] = -1

    idx = np.array([-1 if (x is None or x < 0) else int(x) for x in restricted])
    ii, jj = np.triu_indices(len(target), k=MIN_SEP)
    ok = (idx[ii] >= 0) & (idx[jj] >= 0)
    ii, jj = ii[ok], jj[ok]
    n_true = int(cmap[idx[ii], idx[jj]].sum())
    chance = n_true / float(ii.size)
    print("  restricted: %d of %d positions, %d candidate pairs, %d true LR contacts, "
          "chance %.4f" % (cov_len, len(target), ii.size, n_true, chance))

    def record(model, regime, seed, prec, n_eval, note=""):
        rows.append({"target": name, "model": model,
                     "regime": regime, "m": M if regime.startswith("random") else 0,
                     "seed": seed, "precision": round(prec, 6), "n_eval": n_eval,
                     "chance": round(chance, 6),
                     "x_chance": round(prec / chance, 3) if chance > 0 else "",
                     "note": note})
        print("  %-8s %-18s seed=%s  P@L/2 = %.4f  (%.1fx chance)  n_eval=%d %s"
              % (model, regime, seed, prec, prec / chance if chance > 0 else float("nan"),
                 n_eval, note), flush=True)

    def embed(s_small):
        full = np.zeros((len(target), len(target)))
        full[t0:t0 + cov_len, t0:t0 + cov_len] = s_small
        return full

    # ---- Potts ----------------------------------------------------------------------
    s_potts = mrf.contact_map(keep)
    p, n_eval = structures.precision_at_topk(embed(s_potts), cmap, restricted,
                                             min_sep=MIN_SEP, top_frac=TOP_FRAC)
    record("potts", "identity", 0, p, n_eval)

    for seed in SEEDS:
        rng = np.random.default_rng(1000 * seed + M)
        perm = permute.PERM_BUILDERS["random"](M, rng)
        s_perm = mrf.contact_map(permute.apply_perm_msa(keep, perm))
        d = float(np.abs(s_potts - s_perm).max())
        pp, ne = structures.precision_at_topk(embed(s_perm), cmap, restricted,
                                              min_sep=MIN_SEP, top_frac=TOP_FRAC)
        record("potts", "random", seed, pp, ne, note="max|dS|=%.1e" % d)
        assert d < 1e-9, "PROPOSITION 2 VIOLATED on %s: max|dS| = %.3e" % (name, d)

    # ---- ESM-2 ----------------------------------------------------------------------
    if wrap is None:
        print("  ESM-2 unavailable -- Potts rows only")
        return
    s_esm = wrap.contact_map(target)
    pf, nf = structures.precision_at_topk(s_esm, cmap, mapping,
                                          min_sep=MIN_SEP, top_frac=TOP_FRAC)
    record("esm2", "identity-fulllen", 0, pf, nf, note="C9 comparison")
    pe, ne = structures.precision_at_topk(s_esm, cmap, restricted,
                                          min_sep=MIN_SEP, top_frac=TOP_FRAC)
    record("esm2", "identity", 0, pe, ne)

    for seed in SEEDS:
        rng = np.random.default_rng(1000 * seed + M)
        perm = permute.PERM_BUILDERS["random"](M, rng)
        sp = wrap.contact_map(permute.apply_perm(target, perm))
        pp, npx = structures.precision_at_topk(sp, cmap, restricted,
                                               min_sep=MIN_SEP, top_frac=TOP_FRAC)
        record("esm2", "random", seed, pp, npx)

    # ---- the registered comparison, per target --------------------------------------
    def mean_of(model, regime):
        v = [r["precision"] for r in rows
             if r["target"] == name and r["model"] == model and r["regime"] == regime]
        return float(np.mean(v)) if v else float("nan")

    potts, esm_id, esm_pm = (mean_of("potts", "identity"), mean_of("esm2", "identity"),
                             mean_of("esm2", "random"))
    print("  --- %s: chance %.4f | Potts %.4f | ESM-2 id %.4f | ESM-2 pi %.4f"
          % (name, chance, potts, esm_id, esm_pm))
    if potts <= 1.5 * chance:
        print("  VERDICT %s: Potts near chance -- UNDERPOWERED, WITHDRAWN (branch 3)" % name)
    elif esm_id <= potts:
        print("  VERDICT %s: ESM-2 never beat Potts here; crossing framing DROPPED "
              "for this target (branch 4)" % name)
    elif esm_pm >= potts:
        print("  VERDICT %s: prediction 4 REFUTED -- permuted ESM-2 not below Potts "
              "(branch 2)" % name)
    else:
        print("  VERDICT %s: CROSSING -- ESM-2 goes from %.2fx Potts to %.2fx Potts"
              % (name, esm_id / potts, esm_pm / potts))


def main():
    print("C10 Potts crossing point  (Amendments 10 + 10a)", flush=True)
    try:
        import torch
        import jacobian as jac_mod
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrap = jac_mod.load_esm2("esm2_t33_650M_UR50D", device=device)
        wrap.batch_size = ESM_BATCH
        print("loaded ESM-2 650M on %s (batch %d)" % (device, ESM_BATCH), flush=True)
    except Exception as exc:                                   # noqa: BLE001
        print("ESM-2 unavailable (%s) -- Potts rows only" % exc)
        wrap = None

    rows = []
    for name, seq, pdb_id, chain, a2m in TARGETS:
        run_target(name, seq, pdb_id, chain, a2m, wrap, rows)

    out = os.path.join(HERE, "..", "results", "h2_c10_potts_crossing.csv")
    with open(out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s  (%d rows)" % (os.path.abspath(out), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

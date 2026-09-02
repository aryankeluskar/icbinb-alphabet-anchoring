"""C14 -- is the contact-map collapse just "the relabeled sequence is out-of-distribution"?

Registered as protocol Amendment 14 BEFORE this file was written.

THE OBJECTION. A randomly relabeled sequence is arguably not a protein as far as the model is
concerned: `A -> W` everywhere makes tryptophan the most common residue. So a reviewer can
accept every result so far and still say the model is being fed gibberish, and that the
In-class-vs-random dissociation just means In-class relabelings stay nearer the
training distribution. C6 answers the EDIT-DISTANCE form of that confound. Hamming distance is
not the model's own notion of distance, so C6 does not answer this form.

THE MEASUREMENT. The model's own average log-likelihood of a sequence,

    LL(s) = (1/L) sum_i log softmax(logits(s)[i])[s_i]

restricted to the same 20-residue support the categorical Jacobian uses, so the two quantities
are computed under one convention. The model-perceived shift is

    dLL(pi) = LL(x) - LL(pi(x))     nats/residue, positive = less protein-like TO THIS MODEL

One forward pass per cell. Label-free, structure-free, alignment-free, like the audit itself.

WHAT IS AND IS NOT THE DV. Marginally dLL and overlap must be related -- the arms differ on
both. The registered question is mediation: does regime still predict overlap at MATCHED dLL?

Cells are not chosen here. They are read from the committed main-sweep CSV and the
permutations regenerated from the recorded seeds, so every dLL joins to an `overlap` that was
measured before dLL was conceived of.

Writes results/h2_c14_perceived_shift.csv.
"""
import os

# Pinned before numpy/torch: this runs on CPU by default (the shared GPU sits at 79/82 GB).
# MUST be 1, not a larger pool. The box has 8 cores at load ~33; an 8-thread pool thrashes and
# measured ~33x SLOWER than single-threaded on the same matmul. A first attempt at 8 threads
# got 0.53 cores of useful work and was on pace for 13 hours; this is the same trap recorded
# in the project notes, walked into again.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

BATCH = 8      # cells per forward pass; all sequences for one protein share a length

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
from run_h2_esm2 import EXTRA, load_pfam_queries   # noqa: E402

SWEEP_CSV = os.path.join(HERE, "..", "results", "h2_esm2_invariance.csv")
OUT_CSV = os.path.join(HERE, "..", "results", "h2_c14_perceived_shift.csv")
MODEL = "esm2_t33_650M_UR50D"

# Registered decision thresholds (Amendment 14). Not tunable after the fact.
MEDIATED_DR2 = 0.05
MEDIATED_PARTIAL = 0.20
NOT_MEDIATED_DR2 = 0.15
NOT_MEDIATED_PARTIAL = 0.30


@torch.no_grad()
def mean_loglik_batch(wrap, seqs):
    """Batched form of mean_loglik. Every sequence in `seqs` must have the same length.

    Identical arithmetic to mean_loglik -- same tokenizer, same BOS/EOS masking, same
    20-residue support -- just amortising the forward pass across cells. Within one protein
    every relabeled sequence has the protein's length, so batching is always legal here.
    """
    toks = torch.stack([wrap._tokenize(s) for s in seqs])
    logits = wrap._logits(toks).float()                        # (B, L, 20)
    lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
    out = []
    for b, s in enumerate(seqs):
        idx = [permute.AA20.index(c) for c in s]
        out.append(float(np.mean([lp[b, i, a] for i, a in enumerate(idx)])))
    return out


@torch.no_grad()
def mean_loglik(wrap, seq):
    """Average log p(residue | sequence) over the 20-residue support, in nats/residue.

    Uses wrap._logits so the BOS/EOS-masking convention is identical to the Jacobian's. The
    support restriction means this is log p(s_i | s, s_i is one of the 20 standard residues);
    it is deliberately NOT a calibrated likelihood, and only DIFFERENCES between a sequence and
    its own relabeling are ever used.
    """
    toks = wrap._tokenize(seq).unsqueeze(0)
    logits = wrap._logits(toks)[0].float()                     # (L, 20)
    logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    idx = [permute.AA20.index(c) for c in seq]
    return float(np.mean([logprobs[i, a] for i, a in enumerate(idx)]))


def _ranks(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties, so a regime arm with repeated values is not given spurious resolution
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / counts)[inv]
    return r


def _r2(y, cols):
    """R^2 of an OLS fit of y on [1, *cols]."""
    design = np.column_stack([np.ones(len(y))] + list(cols))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design.dot(beta)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - resid.dot(resid) / ss_tot if ss_tot > 0 else float("nan")


def _partial_spearman(y, x, z):
    """Spearman-style partial correlation of y with x, controlling z. All inputs ranked."""
    def resid(a, b):
        design = np.column_stack([np.ones(len(b)), b])
        beta, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design.dot(beta)
    ry, rx = resid(y, z), resid(x, z)
    denom = np.sqrt(ry.dot(ry) * rx.dot(rx))
    return float(ry.dot(rx) / denom) if denom > 0 else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() and os.environ.get("C14_CUDA") else "cpu"

    with open(SWEEP_CSV) as fh:
        sweep = [r for r in csv.DictReader(fh)]
    sweep = [r for r in sweep if r["model"] == MODEL and r["regime"] != "identity"]

    pfam_dir = os.path.join(ROOT, "data", "pfam")
    proteins = dict(EXTRA)
    proteins.update(load_pfam_queries(pfam_dir))

    missing = sorted({r["protein"] for r in sweep} - set(proteins))
    if missing:
        print("WARNING: %d sweep proteins not resolvable, skipped: %s"
              % (len(missing), ", ".join(missing)), flush=True)
        sweep = [r for r in sweep if r["protein"] not in missing]

    print("C14 model-perceived shift   device=%s  cells=%d  proteins=%d"
          % (device, len(sweep), len({r["protein"] for r in sweep})), flush=True)

    wrap = jac_mod.load_esm2(MODEL, device=device)

    # Group by protein so a batch always shares one sequence length.
    by_protein = collections.OrderedDict()
    for r in sweep:
        by_protein.setdefault(r["protein"], []).append(r)

    rows = []
    done = 0
    t_start = time.time()
    for pid, cells in by_protein.items():
        seq = proteins[pid]
        ll_x = mean_loglik(wrap, seq)
        print("  %-28s L=%-4d LL(x)=%+.4f  (%d cells)"
              % (pid, len(seq), ll_x, len(cells)), flush=True)

        # Same seeding convention as run_h2_esm2.py, so each permutation is the one that
        # produced that row's overlap.
        prepared = []
        for r in cells:
            rng = np.random.default_rng(1000 * int(r["seed"]) + int(r["m"]))
            perm = permute.PERM_BUILDERS[r["regime"]](int(r["m"]), rng)
            prepared.append((r, permute.apply_perm(seq, perm),
                             sum(1 for a, b in perm.items() if a != b)))

        for start in range(0, len(prepared), BATCH):
            chunk = prepared[start:start + BATCH]
            lls = mean_loglik_batch(wrap, [p[1] for p in chunk])
            for (r, pseq, realized), ll_p in zip(chunk, lls):
                rows.append(dict(
                    protein=pid, length=len(seq), regime=r["regime"], m=int(r["m"]),
                    seed=int(r["seed"]), m_realized=realized,
                    hamming=sum(1 for a, b in zip(seq, pseq) if a != b),
                    ll_identity=round(ll_x, 6), ll_permuted=round(ll_p, 6),
                    delta_ll=round(ll_x - ll_p, 6),
                    overlap=float(r["overlap"]), chance=float(r["chance"])))
            done += len(chunk)
            # Write after every batch: this is a multi-hour CPU job on a shared box and a
            # single write at the end would put all of it at risk.
            with open(OUT_CSV, "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("    %d/%d  (%.0fs)" % (done, len(sweep), time.time() - t_start), flush=True)

    print("\nwrote %s (%d rows)" % (OUT_CSV, len(rows)), flush=True)

    analyse(rows)
    return 0


def analyse(rows):
    print("\n" + "=" * 74, flush=True)
    print("C14 ANALYSIS  (Amendment 14; thresholds fixed before measurement)", flush=True)
    print("=" * 74, flush=True)

    print("\nmean delta_LL by regime x m  (nats/residue; + = less protein-like to the model)")
    print("  %-14s %8s %8s %8s %8s" % ("regime", "m=2", "m=6", "m=10", "m=20"))
    for regime in ("random", "In-class", "Cross-class"):
        cells = []
        for m in (2, 6, 10, 20):
            sel = [r["delta_ll"] for r in rows if r["regime"] == regime and r["m"] == m]
            cells.append("%8.3f" % np.mean(sel) if sel else "%8s" % "-")
        print("  %-14s %s" % (regime, " ".join(cells)), flush=True)

    print("\nmean overlap by regime x m")
    for regime in ("random", "In-class", "Cross-class"):
        cells = []
        for m in (2, 6, 10, 20):
            sel = [r["overlap"] for r in rows if r["regime"] == regime and r["m"] == m]
            cells.append("%8.3f" % np.mean(sel) if sel else "%8s" % "-")
        print("  %-14s %s" % (regime, " ".join(cells)), flush=True)

    # ---- the registered mediation test: In-class vs random only --------------------
    sel = [r for r in rows if r["regime"] in ("random", "In-class")]
    y = _ranks([r["overlap"] for r in sel])
    x = _ranks([r["delta_ll"] for r in sel])
    g = np.array([1.0 if r["regime"] == "In-class" else 0.0 for r in sel])

    r2_base = _r2(y, [x])
    r2_full = _r2(y, [x, g])
    d_r2 = r2_full - r2_base
    partial = _partial_spearman(y, g, x.reshape(-1, 1))
    marginal = float(np.corrcoef(y, x)[0, 1])

    print("\nn = %d cells (random + In-class), ESM-2 650M" % len(sel), flush=True)
    print("  marginal Spearman(overlap, delta_LL)      = %+.3f" % marginal, flush=True)
    print("  R^2  overlap ~ delta_LL                   =  %.3f" % r2_base, flush=True)
    print("  R^2  overlap ~ delta_LL + regime          =  %.3f" % r2_full, flush=True)
    print("  delta-R^2 attributable to regime          =  %.3f" % d_r2, flush=True)
    print("  partial Spearman(overlap, regime | dLL)   = %+.3f" % partial, flush=True)

    print("\nVERDICT:", flush=True)
    if d_r2 < MEDIATED_DR2 and abs(partial) < MEDIATED_PARTIAL:
        print("  BRANCH A -- MEDIATED. The registered prediction (not mediated) is WRONG.",
              flush=True)
        print("  Overlap is a function of the model's own perceived shift; biochemistry", flush=True)
        print("  matters only insofar as it keeps the input near the training distribution.",
              flush=True)
        print("  This is a MEASURED Delta, not a constructed one. Reorganise outline sec.4.",
              flush=True)
    elif d_r2 >= NOT_MEDIATED_DR2 and abs(partial) >= NOT_MEDIATED_PARTIAL:
        print("  BRANCH B -- NOT MEDIATED, as registered. Perceived-OOD-ness does not", flush=True)
        print("  explain the dissociation: at matched delta_LL, In-class relabelings", flush=True)
        print("  still retain substantially more of the contact set. The failure is more",
              flush=True)
        print("  specific than generic distribution shift.", flush=True)
    else:
        print("  BRANCH C -- AMBIGUOUS (dR2=%.3f, partial=%+.3f fall between the registered"
              % (d_r2, partial), flush=True)
        print("  thresholds). Report the numbers, claim neither branch.", flush=True)

    # ---- the control that must hold in every branch ------------------------------------
    print("\nNEGATIVE CONTROL (holds in every branch): Potts on the SAME relabeled alignment",
          flush=True)
    print("  is exactly invariant, max|dS| ~ 1e-14 (Proposition 2). A large delta_LL therefore",
          flush=True)
    print("  does NOT mean the coevolutionary computation is impossible on that input -- only",
          flush=True)
    print("  that THIS model finds the input unfamiliar. Never write delta_LL as 'the task got",
          flush=True)
    print("  harder'.", flush=True)

    # ---- within-cell check: immune to the regime confound by construction ---------------
    print("\nWithin-(regime,m) seed-level Spearman(overlap, delta_LL) -- varies only the seed,",
          flush=True)
    print("so it cannot be driven by the regime contrast:", flush=True)
    for regime in ("random", "In-class"):
        for m in (2, 6, 10, 20):
            grp = [r for r in rows if r["regime"] == regime and r["m"] == m]
            if len(grp) < 6:
                continue
            a = _ranks([r["overlap"] for r in grp])
            b = _ranks([r["delta_ll"] for r in grp])
            if a.std() == 0 or b.std() == 0:
                continue
            print("  %-14s m=%-3d n=%-3d rho=%+.3f"
                  % (regime, m, len(grp), np.corrcoef(a, b)[0, 1]), flush=True)


if __name__ == "__main__":
    sys.exit(main())

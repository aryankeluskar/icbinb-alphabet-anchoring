"""C16 -- does ESM-2's *own* published contact predictor show the same alphabet anchoring?

Registered as protocol Amendment 16 BEFORE this file was written and before any C16 number
existed.

Every contact result in this project so far uses the categorical Jacobian (Zhang &
Ovchinnikov) -- OUR instrument, built on top of ESM-2. A reviewer is entitled to ask whether
alphabet anchoring is a property of the model or of the readout. ESM-2 ships a DIFFERENT
contact predictor: the supervised attention-map logistic regression of Rao et al. (2020),
exposed as `model.predict_contacts()` and distributed as the `-contact-regression.pt` sibling
checkpoints. Different function of the same network -- attention maps rather than logit
differences under single-site mutation, supervised rather than unsupervised. If the anchoring
is real it must appear in both. It is also ~1500x cheaper: one forward pass instead of Lx20.

REGISTERED PREDICTIONS (650M, mean over proteins and seeds; thresholds are the module-level
constants below and are NOT tunable after the fact):

  P1 collapse             overlap(random, m=20)                          <= 0.15
  P2 biochemical grading  overlap(In-class, m=20) - overlap(random, m=20) >= 0.20
  P3 instrument agreement Spearman rho(native-head overlap, Jacobian overlap) >= +0.6

The pre-committed interpretation table is reproduced verbatim by `analyse()`; which branch the
paper takes is printed in words, not just PASS/FAIL.

WHAT C16 CANNOT LICENSE (Amendment 16, final paragraph): it cannot compare the two
instruments' ABSOLUTE overlap levels as a measure of quality -- different predictors,
different top-L/2 sets, one supervised and one not. Only the within-instrument pattern across
cells, and the rank correlation between instruments, are interpreted.

------------------------------------------------------------------------------------------
TWO POINTS WHERE THE AMENDMENT IS UNDER-SPECIFIED OR SELF-INCONSISTENT, RESOLVED HERE IN THE
OPEN RATHER THAN SILENTLY.

(1) `min_sep`. Amendment 16 says the DV is "identical to the main sweep and therefore directly
    comparable", and then parenthesises "|i-j| >= 24". Those two clauses contradict each
    other: Amendment 3 relaxed min_sep from 24 to 12 for exactly this (short, L=51-164)
    protein set, and the main sweep, C6-C15 and the committed `h2_esm2_invariance.csv` all use
    12. Comparability is the operative requirement -- P3 joins native-head overlaps against
    Jacobian overlaps cell by cell, and a min_sep mismatch would confound that test with a DV
    mismatch. So the PRIMARY DV here uses the main sweep's `MIN_SEP` (imported, not
    re-declared, so it cannot drift). The amendment's literal >=24 reading is computed too and
    stored in the extra column `overlap_sep24`, and P1/P2 are re-printed at min_sep=24 as a
    labelled sensitivity after the verdict, so this choice cannot hide a different outcome.

(2) `Cross-class, m=20`. The main sweep ran only seed 0 there, on the comment "m=20 Cross-class is
    deterministic". That comment is FALSE: `Cross-class_perm` skips pairs whose residues are
    already used, so the shuffled pair order matters, and seeds 0/1/2 give 16/14/16 moved
    types. Amendment 16 registers seeds {0,1,2} for every cell, so all three are run. The
    two missing Jacobian counterparts simply drop out of the P3 join, which is reported.

Writes results/h2_c16_native_head.csv.
"""
import os

# MUST be 1. The box has 8 cores at load ~33; a larger BLAS pool thrashes and measured ~33x
# slower on the same matmul. C14 was first launched at 8 threads and paced 13 hours; at 1
# thread with batching it finished in 11 minutes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse    # noqa: E402
import csv         # noqa: E402
import sys         # noqa: E402
import time        # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402
from scipy.stats import spearmanr   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import jacobian as jac_mod   # noqa: E402
import permute               # noqa: E402
# The cell grid, the protein set and the DV parameters are IMPORTED, never re-declared, so
# C16 cannot silently drift away from the sweep it is supposed to be matched to.
from run_h2_esm2 import (EXTRA, load_pfam_queries, MIN_SEP, TOP_FRAC,   # noqa: E402
                         M_VALUES, REGIMES, SEEDS)

OUT_CSV = os.path.join(HERE, "..", "results", "h2_c16_native_head.csv")

# The committed main-sweep CSV, for P3. Chosen because it is the only results file carrying
# the categorical-Jacobian DV on the C16 cell grid with all four join keys plus the model
# column: columns are protein,length,model,regime,m,seed,overlap,spearman,chance,n_pairs.
# (h2_c8_scale.csv is the scale sweep on a different 4-protein set; h2_c6/h2_c12 are different
# designs with different arms; h2_c9 is precision-against-structure, not self-overlap.)
MAIN_SWEEP_CSV = os.path.join(HERE, "..", "results", "h2_esm2_invariance.csv")

# Registered model first; the rest are descriptive and only run under --all-scales.
REGISTERED_MODEL = "esm2_t33_650M_UR50D"
DESCRIPTIVE_MODELS = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"]

# Registered thresholds (Amendment 16). Not tunable after the fact.
P1_MAX_RANDOM_M20 = 0.15
P2_MIN_GAP = 0.20
P3_MIN_SPEARMAN = 0.6
# The interpretation table also splits P3 at 0.3 ("disagree" vs "partial agreement").
P3_PARTIAL_FLOOR = 0.3

SMOKE_REGIME = "random"
SMOKE_M = 20
SMOKE_SEED = 0


def n_eligible_pairs(length, min_sep=MIN_SEP):
    """Number of |i-j| >= min_sep pairs, the denominator of the chance floor."""
    return sum(max(0, length - min_sep - i) for i in range(length))


def native_contact_map(wrap, seq):
    """ESM-2's own supervised contact head -> symmetric (L, L) numpy array.

    VERIFIED AGAINST THE INSTALLED fair-esm 2.0.0 SOURCE (not assumed):

      * esm/model/esm2.py:146  `predict_contacts(tokens) = self(tokens,
        return_contacts=True)["contacts"]`, and forward() sets need_head_weights=True whenever
        return_contacts is set, so one plain forward pass is all that is needed.
      * esm/modules.py:317 ContactPredictionHead.forward -- "Performs symmetrization, apc, and
        computes a logistic regression on the output features". SPECIAL TOKENS: it zeroes the
        EOS row/column (`tokens.ne(eos_idx)` outer product) then slices `[..., :-1, :-1]`, and
        slices `[..., 1:, 1:]` for BOS. So the returned map is (B, L, L) with BOS/EOS ALREADY
        STRIPPED -- no trimming here. APC and SYMMETRISATION are applied INTERNALLY
        (`apc(symmetrize(attentions))`, modules.py:355, with symmetrize = x + x^T and apc the
        standard row*col/total correction), followed by a per-pair logistic regression over
        the layers*heads attention channels. The output is therefore symmetric already, in
        [0, 1]. Amendment 16 forbids any extra post-processing, and none is applied.
      * REGRESSION WEIGHTS: the head is a real supervised regression whose weights live in the
        sibling `<model>-contact-regression.pt`. esm/pretrained.py:52 `esm.pretrained.<name>()`
        goes through load_model_and_alphabet_hub -> _download_model_and_regression_data, which
        fetches the regression checkpoint for every non-esm1v model and merges it into the
        state dict (pretrained.py:187) before a STRICT load. So `jacobian.load_esm2` -- which
        calls exactly that -- DOES bring the head in, and no local loader is needed; without
        the merge fair-esm only warns ("Regression weights not found, predicting contacts will
        not produce correct results") and leaves the head randomly initialised, which is why
        `assert_contact_head_loaded` below checks the loaded weights against the file on disk
        instead of trusting the absence of a warning.

    BOS/EOS ARE NOT MASKED HERE. `ESM2Jacobian._logits` replaces them with the mask token
    because the Jacobian paper requires it; that is a property of that readout, not of the
    model, and the published contact head is called on plain tokens. We call the model
    directly rather than through `_logits` so no masking is applied.
    """
    toks = wrap._tokenize(seq).unsqueeze(0).to(wrap.device)
    with torch.no_grad():
        contacts = wrap.model.predict_contacts(toks)
    scores = contacts[0].float().cpu().numpy()
    assert scores.shape == (len(seq), len(seq)), \
        "contact head returned %s for L=%d -- special tokens not stripped as expected" % (
            scores.shape, len(seq))
    return scores


def assert_contact_head_loaded(wrap, model_name):
    """Hard gate: the supervised regression weights really are the published ones.

    A randomly initialised head would still return a plausible-looking (L, L) map, so this is
    checked against the `-contact-regression.pt` on disk rather than inferred.
    """
    ckpt = os.path.join(os.environ.get("TORCH_HOME", os.path.join(ROOT, ".cache", "torch")),
                        "hub", "checkpoints", "%s-contact-regression.pt" % model_name)
    if not os.path.exists(ckpt):
        raise RuntimeError("no %s -- cannot verify the supervised head; C16 is meaningless "
                           "with an untrained contact head" % ckpt)
    data = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    live = wrap.model.contact_head.regression
    for key, tensor in (("contact_head.regression.weight", live.weight),
                        ("contact_head.regression.bias", live.bias)):
        want = data[key].float()
        got = tensor.detach().float().cpu()
        if not torch.allclose(want, got, atol=0, rtol=0):
            raise RuntimeError("%s does not match %s -- the published regression head did NOT "
                               "load" % (key, ckpt))
    print("   contact-regression head verified against %s (%d features)"
          % (os.path.basename(ckpt), live.weight.numel()), flush=True)


def assert_tokenization_matches_fair_esm(wrap, seq):
    """Gate: our tokens are byte-identical to fair-esm's own batch converter."""
    _, _, official = wrap.alphabet.get_batch_converter()([("x", seq)])
    ours = wrap._tokenize(seq).unsqueeze(0)
    if not torch.equal(official, ours):
        raise RuntimeError("tokenization disagrees with fair-esm's batch_converter; BOS/EOS "
                           "convention is wrong and predict_contacts would be misaligned")


def run_model(model_name, proteins, device, out_path, smoke=False):
    """Every C16 cell for one model. Returns the list of row dicts."""
    print("\n=== %s  device=%s  proteins=%d ===" % (model_name, device, len(proteins)),
          flush=True)
    wrap = jac_mod.load_esm2(model_name, device=device)
    assert_contact_head_loaded(wrap, model_name)

    rows = []
    for pid, seq in sorted(proteins.items()):
        length = len(seq)
        n_pairs = n_eligible_pairs(length)
        n_pairs24 = n_eligible_pairs(length, 24)
        assert n_pairs > 0 and n_pairs24 > 0, "%s is too short to rank long-range pairs" % pid

        t0 = time.time()
        assert_tokenization_matches_fair_esm(wrap, seq)
        base_scores = native_contact_map(wrap, seq)
        base_top = jac_mod.top_contacts(base_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        base_top24 = jac_mod.top_contacts(base_scores, min_sep=24, top_frac=TOP_FRAC)
        n_contacts = len(base_top)
        chance = n_contacts / float(n_pairs)

        print("\n%s  L=%d  baseline %.1fs  (chance overlap %.3f)"
              % (pid, length, time.time() - t0, chance), flush=True)

        def cell(regime, m, seed, perm):
            pseq = permute.apply_perm(seq, perm)
            pscores = native_contact_map(wrap, pseq)
            ov = jac_mod.set_overlap(
                base_top, jac_mod.top_contacts(pscores, min_sep=MIN_SEP, top_frac=TOP_FRAC))
            ov24 = jac_mod.set_overlap(
                base_top24, jac_mod.top_contacts(pscores, min_sep=24, top_frac=TOP_FRAC))
            return dict(model=model_name, protein=pid, length=length, regime=regime, m=m,
                        seed=seed, m_realized=sum(1 for a, b in perm.items() if a != b),
                        hamming=sum(1 for a, b in zip(seq, pseq) if a != b),
                        overlap=ov, chance=chance, n_contacts=n_contacts,
                        overlap_sep24=ov24)

        # Identity condition. Recomputed for real, not asserted to 1.0 on paper: it is the
        # pipeline's own correctness gate, and it must come out at exactly 1.000.
        ident = permute.identity_perm()
        assert permute.apply_perm(seq, ident) == seq, "identity permutation altered the sequence"
        row = cell("identity", 0, 0, ident)
        if row["overlap"] != 1.0:
            raise RuntimeError("identity overlap %.6f != 1.000 for %s -- the readout is not "
                               "deterministic and every C16 number would be invalid"
                               % (row["overlap"], pid))
        rows.append(row)
        print("   %-13s m=%-3d overlap=%.3f  (identity gate PASS)" % ("identity", 0, 1.0),
              flush=True)

        regimes = [SMOKE_REGIME] if smoke else REGIMES
        for regime in regimes:
            for m in ([SMOKE_M] if smoke else M_VALUES):
                ovs = []
                for seed in ([SMOKE_SEED] if smoke else SEEDS):
                    # The sweep's seeding convention, so the permutation is the one that
                    # produced the categorical-Jacobian number in the same cell.
                    rng = np.random.default_rng(1000 * seed + m)
                    r = cell(regime, m, seed, permute.PERM_BUILDERS[regime](m, rng))
                    rows.append(r)
                    ovs.append(r["overlap"])
                print("   %-13s m=%-3d overlap=%.3f+-%.3f" % (regime, m, np.mean(ovs),
                                                              np.std(ovs)), flush=True)

        if smoke:
            break
        # Rewrite the whole file after each protein, so a crash loses at most one protein.
        write_csv(out_path, rows)

    del wrap
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def write_csv(path, rows):
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as fh:
        out = []
        for r in csv.DictReader(fh):
            r["m"] = int(r["m"])
            r["seed"] = int(r["seed"])
            r["length"] = int(r["length"])
            r["m_realized"] = int(r["m_realized"])
            r["hamming"] = int(r["hamming"])
            r["n_contacts"] = int(r["n_contacts"])
            for k in ("overlap", "chance", "overlap_sep24"):
                if k in r:
                    r[k] = float(r[k])
            out.append(r)
    return out


def load_jacobian_cells(path=MAIN_SWEEP_CSV, model=REGISTERED_MODEL):
    """(protein, regime, m, seed) -> categorical-Jacobian overlap, for the 650M rows."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rd = csv.DictReader(fh)
        need = {"protein", "model", "regime", "m", "seed", "overlap"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            raise RuntimeError("%s lacks the join columns %s" % (path, sorted(missing)))
        return {(r["protein"], r["regime"], int(r["m"]), int(r["seed"])): float(r["overlap"])
                for r in rd if r["model"] == model}


def _mean(rows, model, regime, m, key="overlap"):
    sel = [r[key] for r in rows
           if r["model"] == model and r["regime"] == regime and r["m"] == m]
    return float(np.mean(sel)) if sel else float("nan")


def analyse(rows):
    print("\n" + "=" * 78, flush=True)
    print("C16 ANALYSIS  (Amendment 16; thresholds fixed before measurement)", flush=True)
    print("=" * 78, flush=True)

    reg = [r for r in rows if r["model"] == REGISTERED_MODEL]
    n_prot = len({r["protein"] for r in reg})
    print("registered model: %s   proteins=%d  cells=%d" % (REGISTERED_MODEL, n_prot, len(reg)),
          flush=True)

    # ---- (a) per-regime x m table of mean overlap, 650M -----------------------------------
    print("\nMean native-head overlap, %s  (identity = %.3f)"
          % (REGISTERED_MODEL, _mean(reg, REGISTERED_MODEL, "identity", 0)), flush=True)
    print("  %-14s %8s %8s %8s %8s" % ("regime", "m=2", "m=6", "m=10", "m=20"), flush=True)
    for regime in REGIMES:
        vals = [_mean(reg, REGISTERED_MODEL, regime, m) for m in M_VALUES]
        print("  %-14s %8.3f %8.3f %8.3f %8.3f" % (regime, vals[0], vals[1], vals[2], vals[3]),
              flush=True)
    print("  (mean chance floor %.4f)"
          % float(np.mean([r["chance"] for r in reg])), flush=True)

    for mdl in DESCRIPTIVE_MODELS:
        sub = [r for r in rows if r["model"] == mdl]
        if not sub:
            continue
        print("\nDESCRIPTIVE ONLY, not registered: %s" % mdl, flush=True)
        for regime in REGIMES:
            vals = [_mean(sub, mdl, regime, m) for m in M_VALUES]
            print("  %-14s %8.3f %8.3f %8.3f %8.3f"
                  % (regime, vals[0], vals[1], vals[2], vals[3]), flush=True)

    # ---- (b) the three registered tests ---------------------------------------------------
    rand20 = _mean(reg, REGISTERED_MODEL, "random", 20)
    cons20 = _mean(reg, REGISTERED_MODEL, "In-class", 20)
    gap = cons20 - rand20
    rho, n_join, dropped = p3_spearman(reg)

    p1 = rand20 <= P1_MAX_RANDOM_M20
    p2 = gap >= P2_MIN_GAP
    p3 = (not np.isnan(rho)) and rho >= P3_MIN_SPEARMAN

    print("\n" + "-" * 78, flush=True)
    print("REGISTERED TESTS (Amendment 16)", flush=True)
    print("  P1 collapse            overlap(random, m=20) = %.3f  <= %.2f   %s"
          % (rand20, P1_MAX_RANDOM_M20, "PASS" if p1 else "FAIL"), flush=True)
    print("  P2 biochemical grading overlap(cons,m=20) - overlap(rand,m=20) = %.3f - %.3f "
          "= %.3f  >= %.2f   %s"
          % (cons20, rand20, gap, P2_MIN_GAP, "PASS" if p2 else "FAIL"), flush=True)
    print("  P3 instrument agreement Spearman rho(native, Jacobian) = %s over %d cells  "
          ">= %+.2f   %s" % ("%.3f" % rho if not np.isnan(rho) else "n/a", n_join,
                             P3_MIN_SPEARMAN, "PASS" if p3 else "FAIL"), flush=True)

    # ---- (c) the pre-committed interpretation table, in words -----------------------------
    print("\n" + "-" * 78, flush=True)
    print("VERDICT -- the branch Amendment 16 committed the paper to, before the numbers:",
          flush=True)
    if p1 and p2 and p3:
        print("  ALL THREE HOLD. The paper states that the anchoring is a property of the",
              flush=True)
        print("  MODEL, not of our readout, and that it reproduces in ESM-2's own published",
              flush=True)
        print("  contact predictor. The categorical Jacobian remains the primary instrument",
              flush=True)
        print("  because it is unsupervised; the native head becomes the robustness check.",
              flush=True)
    else:
        if not p1:
            print("  P1 FAILS -- the native head is robust where the Jacobian collapses.",
                  flush=True)
            print("  THE ANCHORING CLAIM IS INSTRUMENT-SPECIFIC. Every contact claim in the",
                  flush=True)
            print("  paper is narrowed to 'the categorical-Jacobian readout', in the ABSTRACT",
                  flush=True)
            print("  and in section 3 -- not in a footnote. This is a refutation of the general",
                  flush=True)
            print("  claim and is reported as one.", flush=True)
        elif not p2:
            print("  P1 HOLDS, P2 FAILS. Both readouts collapse, but the biochemical grading is",
                  flush=True)
            print("  Jacobian-specific. Section 4.1's regime story is narrowed to the Jacobian;",
                  flush=True)
            print("  the collapse claim survives unchanged.", flush=True)
        else:
            print("  P1 and P2 hold; only P3 falls short. See the P3 branch below.", flush=True)
        if np.isnan(rho):
            print("  P3 COULD NOT BE EVALUATED: no matched cells (see the join report above).",
                  flush=True)
        elif rho < P3_PARTIAL_FLOOR:
            print("  P3 = %.3f < %.2f -- the two instruments DISAGREE cell by cell. Report both,"
                  % (rho, P3_PARTIAL_FLOOR), flush=True)
            print("  side by side, and claim nothing about their agreement. DO NOT average them.",
                  flush=True)
        elif rho < P3_MIN_SPEARMAN:
            print("  %.2f <= P3 = %.3f < %.2f -- report the correlation WITH ITS VALUE and"
                  % (P3_PARTIAL_FLOOR, rho, P3_MIN_SPEARMAN), flush=True)
            print("  describe the agreement as PARTIAL. Do not call it a replication.",
                  flush=True)
        else:
            print("  P3 = %.3f passes, so the instruments agree cell by cell even though the"
                  % rho, flush=True)
            print("  registered prediction(s) above did not all hold.", flush=True)

    # ---- sensitivity: the amendment's literal |i-j| >= 24 reading -------------------------
    if any("overlap_sep24" in r for r in reg):
        r20 = _mean(reg, REGISTERED_MODEL, "random", 20, "overlap_sep24")
        c20 = _mean(reg, REGISTERED_MODEL, "In-class", 20, "overlap_sep24")
        print("\nSENSITIVITY, not a second test: Amendment 16 parenthesises |i-j| >= 24 while",
              flush=True)
        print("the sweep it calls identical uses 12 (Amendment 3). At min_sep=24 the same two",
              flush=True)
        print("registered comparisons read  P1 %.3f <= %.2f %s   P2 %.3f >= %.2f %s"
              % (r20, P1_MAX_RANDOM_M20, "PASS" if r20 <= P1_MAX_RANDOM_M20 else "FAIL",
                 c20 - r20, P2_MIN_GAP, "PASS" if (c20 - r20) >= P2_MIN_GAP else "FAIL"),
              flush=True)
        print("The primary DV stays at min_sep=%d because P3 joins against Jacobian overlaps"
              % MIN_SEP, flush=True)
        print("measured at that value; a mismatch there would confound the agreement test.",
              flush=True)

    print("\nC16 cannot compare the two instruments' ABSOLUTE overlap levels as a measure of",
          flush=True)
    print("quality: different predictors, different top-L/2 sets, one supervised and one not.",
          flush=True)
    print("Only the within-instrument pattern across cells and the rank correlation between",
          flush=True)
    print("instruments are interpreted.", flush=True)


def p3_spearman(reg_rows):
    """Spearman rho between the native-head and categorical-Jacobian overlaps.

    Joins on (protein, regime, m, seed) for the 650M rows of the committed main sweep. Cells
    missing on either side are DROPPED and the drop is reported, never silent.
    """
    print("\n" + "-" * 78, flush=True)
    print("P3 join against %s" % os.path.relpath(MAIN_SWEEP_CSV, ROOT), flush=True)
    jac = load_jacobian_cells()
    if jac is None:
        print("  MISSING FILE -- P3 cannot be computed.", flush=True)
        return float("nan"), 0, 0
    jac_proteins = {k[0] for k in jac}

    # Identity cells are 1.000 on BOTH instruments by construction; including them would
    # inflate rho with a point that carries no information. They are excluded from the
    # registered value and the alternative is printed below so nothing is hidden.
    cells = [r for r in reg_rows if r["regime"] != "identity"]

    xs, ys = [], []
    drop_no_protein, drop_no_cell = [], []
    for r in cells:
        key = (r["protein"], r["regime"], r["m"], r["seed"])
        if key in jac:
            xs.append(r["overlap"])
            ys.append(jac[key])
        elif r["protein"] not in jac_proteins:
            drop_no_protein.append(key)
        else:
            drop_no_cell.append(key)

    n_drop = len(drop_no_protein) + len(drop_no_cell)
    print("  matched %d cells; dropped %d of %d" % (len(xs), n_drop, len(cells)), flush=True)
    if drop_no_protein:
        prots = sorted({k[0] for k in drop_no_protein})
        print("    %d dropped because the protein is absent from the Jacobian CSV entirely "
              "(%s)" % (len(drop_no_protein), ", ".join(prots)), flush=True)
    if drop_no_cell:
        print("    %d dropped because the protein is present but that (regime, m, seed) cell "
              "was never run: %s" % (len(drop_no_cell),
                                     ", ".join("%s/%s m=%d s=%d" % k for k in
                                               sorted(drop_no_cell)[:6])), flush=True)
    n_ident = len(reg_rows) - len(cells)
    print("    (%d identity cells excluded by design: both instruments are 1.000 there by "
          "construction)" % n_ident, flush=True)

    if len(xs) < 3:
        print("  too few matched cells for a rank correlation.", flush=True)
        return float("nan"), len(xs), n_drop
    rho = float(spearmanr(xs, ys).statistic)
    with_ident = float(spearmanr(xs + [1.0] * n_ident, ys + [1.0] * n_ident).statistic)
    print("  rho = %+.4f   (with the identity cells added back: %+.4f)" % (rho, with_ident),
          flush=True)
    return rho, len(xs), n_drop


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all-scales", action="store_true",
                    help="also run 8M/35M/150M as DESCRIPTIVE arms after the registered 650M")
    ap.add_argument("--smoke", action="store_true",
                    help="one protein x %s x m=%d x seed=%d, print and exit; writes no CSV"
                         % (SMOKE_REGIME, SMOKE_M, SMOKE_SEED))
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-run analyse() on the committed CSV without recomputing anything")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.analyse_only:
        if not os.path.exists(OUT_CSV):
            print("no %s to analyse" % OUT_CSV, flush=True)
            return 1
        analyse(read_csv(OUT_CSV))
        return 0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    proteins = dict(EXTRA)
    proteins.update(load_pfam_queries(os.path.join(ROOT, "data", "pfam")))

    print("C16 native contact head   device=%s  proteins=%d  cells/protein=%d"
          % (device, len(proteins),
             1 + len(REGIMES) * len(M_VALUES) * len(SEEDS)), flush=True)
    for k, v in sorted(proteins.items()):
        print("   %-28s L=%d" % (k, len(v)), flush=True)

    if args.smoke:
        pid = sorted(proteins)[0]
        print("\nSMOKE: %s only, %s m=%d seed=%d, no CSV written"
              % (pid, SMOKE_REGIME, SMOKE_M, SMOKE_SEED), flush=True)
        run_model(REGISTERED_MODEL, {pid: proteins[pid]}, device, OUT_CSV, smoke=True)
        print("\nsmoke OK", flush=True)
        return 0

    # Registered model first, always.
    models = [REGISTERED_MODEL] + (DESCRIPTIVE_MODELS if args.all_scales else [])
    rows = []
    t0 = time.time()
    for model_name in models:
        rows += run_model(model_name, proteins, device, OUT_CSV)
        write_csv(OUT_CSV, rows)

    print("\nwrote %s (%d rows, %.0fs)" % (OUT_CSV, len(rows), time.time() - t0), flush=True)
    analyse(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Alphabet-equivariance audit for protein language models.

Run this on any pLM to ask one question:

    Does the model's contact prediction change when you rename the amino acids?

Coevolutionary contact prediction is a function of alignment column statistics. A global
bijective relabeling of the alphabet permutes those statistics without altering them -- the
histogram is the same histogram with its bins renamed -- so **any** function of them is exactly
invariant. This is a theorem, not a benchmark result (see experiments/H2-alphabet-permutation/
theory.md, Propositions 1 and 2). It gives a necessary condition, and its contrapositive is
what this script measures:

    A model whose contact predictions change under global alphabet relabeling is not
    computing a function of alignment column statistics.

WHY THIS IS CHEAP ENOUGH TO BE WORTH RUNNING. The audit is **label-free and structure-free**.
It needs no experimental structure, no ground-truth contacts, no MSA, and no held-out set --
only one sequence and L x 20 masked forward passes, because the model is compared against
*itself*. There is nothing to leak and nothing to tune, so there is no version of this audit
that can be gamed by a better baseline or a bigger evaluation set.

WHAT A SCORE MEANS. `E_random` is the fraction of the model's own top-L/2 long-range contacts
that survive a random relabeling. A model satisfying the necessary condition scores 1.000
exactly. A model whose predictions are unrelated to its own scores at the chance rate, which is
the top-L/2 budget divided by the number of candidate pairs -- typically 0.02, so the dynamic
range is large and the reference point is unambiguous.

`E_In-class` repeats the audit with a relabeling that maps residues within biochemical
similarity groups. Reporting both matters: E_In-class >> E_random says the model is keyed
on biochemical identity rather than on column statistics, and it also rules out the objection
that the relabeled sequence is simply a harder input, since both relabelings change the same
number of positions.

USAGE

    # a fair-esm model by name
    python alphabet_audit.py --esm esm2_t30_150M_UR50D --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

    # an AMPLIFY checkpoint directory
    python alphabet_audit.py --amplify /path/to/AMPLIFY_120M_2024 --fasta seqs.fa

    # anything else: import and pass your own wrapper
    from alphabet_audit import audit
    report = audit(my_wrapper, sequence)      # needs only .contact_map(str) -> LxL array

The third form is the point. `contact_map` is the entire interface: any model that can score
L x 20 single-residue substitutions can be audited, whatever its architecture or tokenizer.

VERIFIED AGAINST THE PAPER'S OWN NUMBERS. This is not a reimplementation of the experiment
code -- it calls the same `jacobian` and `permute` modules -- and `--self-test` proves it by
reproducing four committed cells of the C8 scale sweep exactly (ESM-2 8M, ubiquitin, m=20:
random 0.15789 / 0.00000, In-class 0.21053 / 0.34211, chance 0.01827).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import jacobian as jac_mod       # noqa: E402
import permute                   # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5
DEFAULT_M = 20
DEFAULT_SEEDS = (0, 1, 2)


def audit(model, seq, m=DEFAULT_M, seeds=DEFAULT_SEEDS, min_sep=MIN_SEP, top_frac=TOP_FRAC,
          regimes=("random", "In-class"), verbose=False):
    """Audit one sequence. `model` needs only `.contact_map(str) -> (L, L) array`.

    Returns a dict with, per regime, the mean/sd overlap across seeds and the realized m --
    the number of residues the sampled permutation actually moved, which for `In-class` is
    capped at 15 because five residues (A, C, G, H, P) have no partner in the similarity
    groups. Reporting requested m without realized m overstates the intervention.
    """
    length = len(seq)
    base = model.contact_map(seq)
    base_top = jac_mod.top_contacts(base, min_sep=min_sep, top_frac=top_frac)
    n_pairs = sum(max(0, length - min_sep - i) for i in range(length))
    chance = len(base_top) / float(n_pairs)

    report = {"length": length, "n_top": len(base_top), "chance": chance, "m_requested": m,
              "regimes": {}}
    for regime in regimes:
        overlaps, realized = [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + m)
            perm = permute.PERM_BUILDERS[regime](m, rng)
            realized.append(sum(1 for a, b in perm.items() if a != b))
            pscores = model.contact_map(permute.apply_perm(seq, perm))
            overlaps.append(jac_mod.set_overlap(
                base_top, jac_mod.top_contacts(pscores, min_sep=min_sep, top_frac=top_frac)))
            if verbose:
                print("    %-13s seed %d  overlap %.3f" % (regime, seed, overlaps[-1]),
                      flush=True)
        report["regimes"][regime] = {
            "mean": float(np.mean(overlaps)), "sd": float(np.std(overlaps)),
            "x_chance": float(np.mean(overlaps) / chance) if chance > 0 else float("nan"),
            "m_realized": int(np.mean(realized)), "overlaps": [float(x) for x in overlaps]}
    return report


def format_report(name, seq_id, rep):
    """Human-readable single-sequence report."""
    out = []
    out.append("  %s  L=%d  top-L/2=%d contacts  chance=%.4f"
               % (seq_id, rep["length"], rep["n_top"], rep["chance"]))
    for regime, r in rep["regimes"].items():
        # The verdict attaches to the RANDOM arm only. `In-class` deliberately relabels
        # within biochemical groups, so a high score there is diagnostic of the mechanism
        # rather than evidence of equivariance -- calling it a pass would invert the meaning.
        verdict = ""
        if regime == "random":
            verdict = ("   SATISFIES the necessary condition" if r["mean"] > 0.999
                       else "   VIOLATES the necessary condition")
        out.append("    %-13s m=%2d (realized %2d)   E = %.3f +- %.3f   = %5.1fx chance%s"
                   % (regime, rep["m_requested"], r["m_realized"], r["mean"], r["sd"],
                      r["x_chance"], verdict))
    rr = rep["regimes"]
    if "random" in rr and "In-class" in rr and rr["random"]["mean"] > 1e-9:
        ratio = rr["In-class"]["mean"] / rr["random"]["mean"]
        out.append("    dissociation  E_In-class / E_random = %.1fx" % ratio)
        out.append("      (>1 means the model is keyed on biochemical identity, not on column")
        out.append("       statistics; both relabelings change the same number of positions)")
    return "\n".join(out)


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")

# Four cells of the committed C8 scale sweep (results/h2_c8_scale.csv, ESM-2 8M, ubiquitin,
# m=20). If the audit and the experiment ever diverge, the audit is not the same instrument
# that produced the paper's numbers and its output should not be trusted.
SELF_TEST_EXPECTED = {"random": [0.15789, 0.00000],
                      "In-class": [0.21053, 0.34211],
                      "chance": 0.01827}


def self_test(device, batch_size):
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = jac_mod.ESM2Jacobian(esm_model, alphabet, device=device, batch_size=batch_size)
    rep = audit(model, UBIQUITIN, m=20, seeds=(0, 1))
    ok = True
    print("\nself-test vs committed C8 cells (ESM-2 8M, ubiquitin, m=20)")
    d = abs(rep["chance"] - SELF_TEST_EXPECTED["chance"])
    print("  chance        got %.5f  expected %.5f   %s"
          % (rep["chance"], SELF_TEST_EXPECTED["chance"], "OK" if d < 1e-4 else "MISMATCH"))
    ok &= d < 1e-4
    for regime in ("random", "In-class"):
        got = rep["regimes"][regime]["overlaps"]
        exp = SELF_TEST_EXPECTED[regime]
        worst = max(abs(a - b) for a, b in zip(got, exp))
        print("  %-13s got %s  expected %s   %s"
              % (regime, ["%.5f" % x for x in got], ["%.5f" % x for x in exp],
                 "OK" if worst < 1e-4 else "MISMATCH"))
        ok &= worst < 1e-4
    print("\nself-test %s" % ("PASSED -- the audit is the same instrument that produced the "
                              "paper's numbers." if ok else "FAILED -- do not trust output."))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if "--self-test" in sys.argv:
        import torch
        ap2 = argparse.ArgumentParser(add_help=False)
        ap2.add_argument("--self-test", action="store_true")
        ap2.add_argument("--device", default=None)
        ap2.add_argument("--batch-size", type=int, default=8)
        a2, _ = ap2.parse_known_args()
        dev = a2.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self_test(dev, a2.batch_size)
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce four committed C8 cells and exit")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--esm", metavar="NAME", help="fair-esm model name, e.g. esm2_t33_650M_UR50D")
    src.add_argument("--amplify", metavar="DIR", help="AMPLIFY checkpoint directory")
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--seq", help="a single amino-acid sequence")
    inp.add_argument("--fasta", help="FASTA file; every record is audited")
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="residues to relabel (default 20)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every seed")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.esm:
        import esm
        esm_model, alphabet = getattr(esm.pretrained, args.esm)()
        model = jac_mod.ESM2Jacobian(esm_model, alphabet, device=device,
                                     batch_size=args.batch_size)
        name = args.esm
    else:
        import amplify_jac
        model = amplify_jac.AMPLIFYJacobian(args.amplify, device=device,
                                            batch_size=args.batch_size)
        name = os.path.basename(args.amplify.rstrip("/"))

    seqs = {"query": args.seq} if args.seq else read_fasta(args.fasta)

    print("\nAlphabet-equivariance audit -- %s  (device=%s, %d sequence(s))"
          % (name, device, len(seqs)))
    print("A model computing a function of alignment column statistics scores E = 1.000\n")

    reports = {}
    for sid, seq in sorted(seqs.items()):
        bad = sorted(set(seq) - set(permute.AA20))
        if bad:
            print("  %s: skipped, non-standard characters %s" % (sid, "".join(bad)))
            continue
        rep = audit(model, seq, m=args.m, seeds=tuple(args.seeds), verbose=args.verbose)
        reports[sid] = rep
        print(format_report(name, sid, rep))
        print()

    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump({"model": name, "reports": reports}, fh, indent=2)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

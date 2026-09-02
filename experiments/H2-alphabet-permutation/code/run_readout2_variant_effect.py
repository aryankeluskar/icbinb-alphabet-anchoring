"""H2 second readout: zero-shot variant effect under alphabet relabeling.

Locked in protocol-readout2-variant-effect.md before this ran.

The transformation renames letters. A variant K41M becomes pi(K)41pi(M), the wild type
becomes pi(WT), the alignment becomes pi(MSA), and the wet-lab fitness value is untouched
because it belongs to a physical protein and not to our notation. A model with an
alphabet-abstract computation would score the renamed variant exactly as it scored the
original, so its Spearman against DMS would not move.

Writes results/h2_readout2_variant_effect.csv incrementally.
"""
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import permute      # noqa: E402

PG = os.path.join(ROOT, "data", "proteingym")
DMS_DIR = os.path.join(PG, "DMS_ProteinGym_substitutions",
                       "DMS_ProteinGym_substitutions")
MSA_DIR = os.path.join(PG, "DMS_msa_files", "DMS_msa_files")
ZS_DIR = os.path.join(PG, "zero_shot_substitutions_scores")

N_ASSAYS = 15
MAX_LEN = 400
MIN_ROWS = 200
M_VALUES = [2, 6, 10, 20]
REGIMES = ["random", "In-class", "Cross-class"]
SEEDS = [0, 1, 2]
BATCH = 32
PUBLISHED_COL = "ESM2_650M"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def read_a2m(path, max_rows=20000):
    """a2m: lowercase letters and dots are insertions relative to the query."""
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                if len(seqs) >= max_rows:
                    break
            elif line:
                cur.append(line)
    if cur and len(seqs) < max_rows:
        seqs.append("".join(cur))
    # Match columns only: drop insert columns, which are lowercase or '.'.
    return ["".join(c for c in s if not c.islower() and c != ".").upper()
            for s in seqs if s]


def pssm_scores(msa, muts, pseudo=1.0):
    """Site-independent log-odds, the standard simple baseline.

    Exactly invariant to a consistent relabeling: pi permutes the rows of the count matrix
    and permutes which row each mutation reads, so every log-odds value is preserved.
    Asserted numerically in main() rather than taken on trust.
    """
    if not msa:
        return None
    length = len(msa[0])
    idx = {a: i for i, a in enumerate(permute.AA20)}
    counts = np.full((length, 20), pseudo, dtype=float)
    for s in msa:
        if len(s) != length:
            continue
        for j, c in enumerate(s):
            k = idx.get(c)
            if k is not None:
                counts[j, k] += 1.0
    freq = counts / counts.sum(axis=1, keepdims=True)
    logf = np.log(freq)
    out = []
    for wt, pos, mt in muts:
        if pos >= length or wt not in idx or mt not in idx:
            out.append(np.nan)
        else:
            out.append(logf[pos, idx[mt]] - logf[pos, idx[wt]])
    return np.array(out)


class Scorer(object):
    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        import esm
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model = self.model.eval().to(device)
        self.bc = self.alphabet.get_batch_converter()
        self.device = device
        self.aa_tok = {a: self.alphabet.get_idx(a) for a in permute.AA20}

    @torch.no_grad()
    def masked_marginals(self, seq, positions):
        """log p at each masked position, shape (len(positions), 20)."""
        _, _, toks = self.bc([("wt", seq)])
        out = np.zeros((len(positions), 20), dtype=np.float32)
        cols = [self.aa_tok[a] for a in permute.AA20]
        for start in range(0, len(positions), BATCH):
            chunk = positions[start:start + BATCH]
            batch = toks.repeat(len(chunk), 1).to(self.device)
            for r, p in enumerate(chunk):
                batch[r, p + 1] = self.alphabet.mask_idx      # +1 for BOS
            logits = self.model(batch)["logits"]
            lp = torch.log_softmax(logits.float(), dim=-1)
            for r, p in enumerate(chunk):
                out[start + r] = lp[r, p + 1, cols].cpu().numpy()
        return out


def pick_assays():
    msa_for = {}
    for fn in sorted(os.listdir(MSA_DIR)):
        msa_for.setdefault(fn.split("_theta")[0].split("_full")[0], fn)
    cands = []
    for fn in sorted(os.listdir(DMS_DIR)):
        if not fn.endswith(".csv"):
            continue
        assay = fn[:-4]
        key = "_".join(assay.split("_")[:2])
        if key not in msa_for:
            continue
        with open(os.path.join(DMS_DIR, fn)) as fh:
            rows = list(csv.DictReader(fh))
        singles = [r for r in rows if ":" not in r["mutant"]]
        if len(singles) < MIN_ROWS or not singles:
            continue
        if len(singles[0]["mutated_sequence"]) > MAX_LEN:
            continue
        cands.append((assay, msa_for[key], len(singles)))
    rng = np.random.default_rng(0)
    if len(cands) > N_ASSAYS:
        sel = rng.choice(len(cands), N_ASSAYS, replace=False)
        cands = [cands[i] for i in sorted(sel)]
    return cands


def published_rho(assay, muts, dms):
    """The free validity gate: reproduce ProteinGym's own ESM2_650M number."""
    path = os.path.join(ZS_DIR, assay + ".csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        if PUBLISHED_COL not in (rdr.fieldnames or []):
            return None
        want = {"%s%d%s" % (w, p + 1, m) for w, p, m in muts}
        pub, ref = [], []
        for r in rdr:
            if r["mutant"] in want:
                try:
                    pub.append(float(r[PUBLISHED_COL]))
                    ref.append(float(r["DMS_score"]))
                except (ValueError, TypeError):
                    pass
    return spearman(pub, ref) if len(pub) > 50 else None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assays = pick_assays()
    out_path = os.path.join(HERE, "..", "results",
                            "h2_readout2_variant_effect.csv")
    print("readout 2: variant effect under relabeling  device=%s  assays=%d"
          % (device, len(assays)), flush=True)
    for a, m, n in assays:
        print("   %-46s singles=%d" % (a, n), flush=True)

    scorer = Scorer(device=device)
    rows = []

    for assay, msa_fn, _ in assays:
        with open(os.path.join(DMS_DIR, assay + ".csv")) as fh:
            recs = [r for r in csv.DictReader(fh) if ":" not in r["mutant"]]
        muts, dms, wt_seq = [], [], recs[0]["mutated_sequence"]
        for r in recs:
            tok = r["mutant"]
            try:
                w, pos, mt = tok[0], int(tok[1:-1]) - 1, tok[-1]
                d = float(r["DMS_score"])
            except (ValueError, IndexError):
                continue
            if w in permute.AA20 and mt in permute.AA20 and 0 <= pos < len(wt_seq):
                muts.append((w, pos, mt))
                dms.append(d)
        if len(muts) < MIN_ROWS:
            print("  %s: too few usable singles after parsing, skipped" % assay,
                  flush=True)
            continue
        # Reconstruct WT from any variant by reverting its single substitution.
        wt = list(recs[0]["mutated_sequence"])
        w0, p0, _ = muts[0]
        wt[p0] = w0
        wt = "".join(wt)
        dms = np.array(dms, float)
        positions = sorted({p for _, p, _ in muts})
        pos_row = {p: i for i, p in enumerate(positions)}
        aa_col = {a: i for i, a in enumerate(permute.AA20)}
        msa = read_a2m(os.path.join(MSA_DIR, msa_fn))

        def esm_rho(seq, mapped):
            lp = scorer.masked_marginals(seq, positions)
            s = [lp[pos_row[p], aa_col[mapped[mt]]] - lp[pos_row[p], aa_col[mapped[w]]]
                 for w, p, mt in muts]
            return spearman(s, dms)

        ident = {a: a for a in permute.AA20}
        rho0 = esm_rho(wt, ident)
        pssm0 = spearman(pssm_scores(msa, muts), dms)
        pub = published_rho(assay, muts, dms)
        gate = ("n/a" if pub is None
                else ("PASS" if abs(abs(rho0) - abs(pub)) < 0.05 else "FAIL"))
        print("\n%s  n=%d L=%d  msa=%d" % (assay, len(muts), len(wt), len(msa)),
              flush=True)
        print("  identity: ESM2 rho=%+.4f   published=%s   gate=%s   PSSM rho=%+.4f"
              % (rho0, "n/a" if pub is None else "%+.4f" % pub, gate, pssm0),
              flush=True)
        rows.append(dict(assay=assay, n=len(muts), regime="identity", m=0, seed=0,
                         esm_rho=round(rho0, 5), pssm_rho=round(pssm0, 5),
                         published_rho="" if pub is None else round(pub, 5),
                         gate=gate))

        for regime in REGIMES:
            for m in M_VALUES:
                e_d, p_d = [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + m)
                    perm = permute.PERM_BUILDERS[regime](m, rng)
                    pw = permute.apply_perm(wt, perm)
                    pmsa = permute.apply_perm_msa(msa, perm)
                    pmuts = [(perm[w], p, perm[mt]) for w, p, mt in muts]
                    r_e = esm_rho(pw, perm)
                    r_p = spearman(pssm_scores(pmsa, pmuts), dms)
                    e_d.append(r_e - rho0)
                    p_d.append(r_p - pssm0)
                    rows.append(dict(assay=assay, n=len(muts), regime=regime, m=m,
                                     seed=seed, esm_rho=round(r_e, 5),
                                     pssm_rho=round(r_p, 5), published_rho="",
                                     gate=""))
                # Registered prediction 1 is an assert, not a hope.
                if max(abs(x) for x in p_d) > 1e-9:
                    print("    !! PSSM NOT INVARIANT at %s m=%d (max|dRho|=%.2e)"
                          % (regime, m, max(abs(x) for x in p_d)), flush=True)
                print("    %-13s m=%-3d  ESM dRho=%+.3f+-%.3f   PSSM dRho=%+.1e"
                      % (regime, m, np.mean(e_d), np.std(e_d),
                         max(abs(x) for x in p_d)), flush=True)

        with open(out_path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nwrote %s (%d rows)" % (out_path, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

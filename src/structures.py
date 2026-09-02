"""Experimental contact maps from RCSB structures, with no dependencies beyond numpy.

Why this module exists. Every contact number in the project so far is a SELF-CONSISTENCY
number: how much does a model's own prediction move under some transformation. That is a
clean measurement, but it cannot say whether the prediction was ever right. This module
supplies the missing half -- true CB-CB contact maps from deposited structures, so a
predicted map can be scored as long-range precision@L/2 against experiment.

The load-bearing step is not the geometry, it is the ALIGNMENT. A query sequence and a PDB
chain rarely agree residue-for-residue: termini go unobserved, loops are disordered, the
deposited construct carries point mutations, and sometimes the "same" protein is a
different species' ortholog. An off-by-one mapping produces precision numbers that look
plausible and are entirely wrong, so `align_to_query` refuses to guess: it tries exact
containment first, falls back to Needleman-Wunsch, and reports identity statistics that the
caller is expected to look at.

Design notes:
  - biopython is not installed and is not going to be; the mmCIF/PDB parsers here are
    deliberately small and only understand `_atom_site` / `ATOM`.
  - HETATM is skipped by construction. That means modified residues (MSE, PCA, ...) are
    absent from the observed sequence and show up as gaps in the alignment. This is the
    In-class choice: a residue we did not parse becomes "unobserved" and is excluded
    from scoring rather than silently misaligned.
  - nothing is written outside data/pdb/ under the project root; $HOME is never touched.
"""
import os
import shutil
import subprocess
import time
import urllib.request

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(PROJ_ROOT, "data", "pdb")

RCSB_URL = "https://files.rcsb.org/download/%s.cif"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# --------------------------------------------------------------------------------------
# 1. fetching
# --------------------------------------------------------------------------------------

def fetch_pdb(pdb_id, cache_dir=DEFAULT_CACHE, retries=3, timeout=60, force=False):
    """Download <pdb_id>.cif from RCSB into `cache_dir` and return the local path.

    Cached files are reused. Raises RuntimeError if every attempt fails -- callers should
    report the target as unavailable rather than quietly substituting another structure.
    """
    pdb_id = pdb_id.strip().upper()
    if not pdb_id:
        raise ValueError("empty pdb_id")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "%s.cif" % pdb_id)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not force:
        return path

    url = RCSB_URL % pdb_id
    tmp = path + ".part"
    errors = []
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 200:
                raise RuntimeError("suspiciously small response (%d bytes)" % len(data))
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return path
        except Exception as exc:                                    # noqa: BLE001
            errors.append("urllib attempt %d: %r" % (attempt + 1, exc))
            time.sleep(1.5 * (attempt + 1))

    # urllib can trip over the cluster's CA situation in ways curl does not; try it.
    curl = shutil.which("curl")
    if curl is not None:
        try:
            subprocess.run([curl, "-sSL", "--fail", "--max-time", str(timeout),
                            "-o", tmp, url], check=True)
            if os.path.getsize(tmp) > 200:
                os.replace(tmp, path)
                return path
            errors.append("curl: response too small")
        except Exception as exc:                                    # noqa: BLE001
            errors.append("curl: %r" % exc)

    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError("could not fetch %s from RCSB\n  %s" % (pdb_id, "\n  ".join(errors)))


# --------------------------------------------------------------------------------------
# 2. parsing
# --------------------------------------------------------------------------------------

def _split_cif_tokens(line):
    """Whitespace split that respects '...' and \"...\" quoting."""
    toks = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            quote = c
            i += 1
            start = i
            while i < n:
                if line[i] == quote and (i + 1 >= n or line[i + 1] in " \t"):
                    break
                i += 1
            toks.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            toks.append(line[start:i])
    return toks


def _read_atom_site(path):
    """Yield dicts of the `_atom_site` loop of an mmCIF file."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        tags = []
        while j < n and lines[j].lstrip().startswith("_"):
            tags.append(lines[j].strip().split()[0])
            j += 1
        if not tags or not tags[0].startswith("_atom_site."):
            i = j
            continue

        cols = {t.split(".", 1)[1]: k for k, t in enumerate(tags)}
        ncol = len(tags)
        pending = []
        while j < n:
            stripped = lines[j].strip()
            if stripped.startswith("#") or stripped == "loop_" or stripped.startswith("_"):
                break
            if stripped:
                pending.extend(_split_cif_tokens(stripped))
                while len(pending) >= ncol:
                    row = pending[:ncol]
                    del pending[:ncol]
                    yield {k: row[v] for k, v in cols.items()}
            j += 1
        return
    raise ValueError("no _atom_site loop found in %s" % path)


def _residues_from_cif(path, chain):
    """Ordered list of (resid_key, resname, {atom_name: xyz}) for one chain, model 1."""
    out = []
    index = {}
    model0 = None
    for rec in _read_atom_site(path):
        if rec.get("group_PDB", "ATOM") != "ATOM":
            continue
        model = rec.get("pdbx_PDB_model_num", "1")
        if model0 is None:
            model0 = model
        if model != model0:
            continue
        asym = rec.get("auth_asym_id") or rec.get("label_asym_id")
        if asym in (".", "?", None):
            asym = rec.get("label_asym_id")
        if asym != chain:
            continue
        alt = rec.get("label_alt_id", ".")
        if alt not in (".", "?", "A"):
            continue
        seqid = rec.get("auth_seq_id") or rec.get("label_seq_id")
        icode = rec.get("pdbx_PDB_ins_code", ".")
        if icode in ("?", None):
            icode = "."
        key = (seqid, icode)
        name = (rec.get("auth_atom_id") or rec.get("label_atom_id", "")).strip('"')
        resname = (rec.get("auth_comp_id") or rec.get("label_comp_id", "")).upper()
        try:
            xyz = (float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"]))
        except (KeyError, ValueError):
            continue
        if key not in index:
            index[key] = len(out)
            out.append((key, resname, {}))
        atoms = out[index[key]][2]
        if name not in atoms:                       # keep the first altloc seen
            atoms[name] = xyz
    return out


def _residues_from_pdb(path, chain):
    """Same as _residues_from_cif for legacy fixed-column PDB files."""
    out = []
    index = {}
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            key = (line[22:26].strip(), line[26])
            resname = line[17:20].strip().upper()
            name = line[12:16].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if key not in index:
                index[key] = len(out)
                out.append((key, resname, {}))
            atoms = out[index[key]][2]
            if name not in atoms:
                atoms[name] = xyz
    return out


def parse_chain(path, chain):
    """Full parse of one chain: (seq, coords, resids, used_ca).

    seq      one-letter string over OBSERVED residues only, in file order
    coords   (L, 3) float array of CB (CA for glycine / missing CB)
    resids   list of "<auth_seq_id><icode>" strings, parallel to seq
    used_ca  boolean list, True where the CA fallback was taken
    """
    if path.lower().endswith((".cif", ".mmcif")):
        residues = _residues_from_cif(path, chain)
    else:
        residues = _residues_from_pdb(path, chain)
    if not residues:
        raise ValueError("no ATOM records for chain %r in %s" % (chain, path))

    seq, xyz, resids, used_ca = [], [], [], []
    for key, resname, atoms in residues:
        one = THREE_TO_ONE.get(resname, "X")
        cb = atoms.get("CB")
        ca = atoms.get("CA")
        if resname == "GLY" or cb is None:
            pos, fallback = ca, True
        else:
            pos, fallback = cb, False
        if pos is None:                     # neither CB nor CA -- unusable, drop it
            continue
        seq.append(one)
        xyz.append(pos)
        resids.append(key[0] + (key[1] if key[1] not in (".", " ") else ""))
        used_ca.append(fallback)
    return "".join(seq), np.asarray(xyz, dtype=float), resids, used_ca


def ca_cb_coords(path, chain):
    """(seq, coords) for one chain: observed one-letter sequence and (L,3) CB coordinates.

    CB is used everywhere except glycine and residues with no CB atom deposited, where CA
    stands in. Only ATOM records of the first model are read, so ligands, waters, and
    modified residues (which are HETATM) are absent from both outputs.
    """
    seq, coords, _resids, _used_ca = parse_chain(path, chain)
    return seq, coords


# --------------------------------------------------------------------------------------
# 3. geometry
# --------------------------------------------------------------------------------------

def contact_map(coords, cutoff=8.0):
    """Boolean (L,L) map, True where the CB-CB distance is below `cutoff` Angstrom.

    The diagonal and near-diagonal are True by construction; callers are expected to apply
    a |i-j| separation filter. Non-finite coordinates never form contacts.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    ok = np.isfinite(dist)
    return (dist < cutoff) & ok


def n_long_range(cmap, min_sep=12):
    """Number of true contacts with |i-j| >= min_sep, counted once per pair."""
    length = cmap.shape[0]
    iu = np.triu_indices(length, k=min_sep)
    return int(cmap[iu].sum())


# --------------------------------------------------------------------------------------
# 4. alignment
# --------------------------------------------------------------------------------------

NEG = float("-inf")


def _needleman_wunsch(a, b, gap=-1.0, match=1.0, mismatch=0.0):
    """Global alignment, identity scoring, gap -1. Returns [(i_a, i_b)] with None for gaps.

    One refinement over the textbook version, and it is not cosmetic. With linear gap costs
    a deletion can be split into several scattered gaps at EXACTLY the same score, and the
    textbook traceback then picks an arbitrary one. Concretely: hide a 9-residue loop of
    ubiquitin and plain NW returns an alignment that is 100% identical over every mapped
    position -- and silently anchors two query residues onto the wrong copies of K and I a
    few positions away. That is precisely the failure mode that produces believable, wrong
    contact precision.

    The fix is a tie-break, not a change of objective: a per-gap-OPENING surcharge of
    eps = 0.4 / (na + nb). Since match/mismatch/gap are integers, every linear-gap
    alignment score is an integer, and the total surcharge is bounded by 0.4 < 1. So the
    surcharge can never promote a linearly sub-optimal alignment; among the alignments that
    tie at the linear optimum it just selects the one with the fewest, hence longest and
    most contiguous, indels. That is the biologically right choice for disordered loops and
    unresolved termini.

    Implemented as a three-state Gotoh recursion (M = aligned column, X = gap in b,
    Y = gap in a) because the surcharge has to know whether a gap is being opened.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return [(i, None) for i in range(na)] + [(None, j) for j in range(nb)]
    eps = 0.4 / float(na + nb)          # pure tie-breaker, provably score-preserving
    go = gap - eps                      # cost of the first residue of a gap
    ge = gap                            # cost of each subsequent residue

    # rows are reused; pointers must be kept for the whole matrix.
    # pointer values: 0 -> came from M, 1 -> from X, 2 -> from Y
    ptr_m = np.zeros((na + 1, nb + 1), dtype=np.int8)
    ptr_x = np.zeros((na + 1, nb + 1), dtype=np.int8)
    ptr_y = np.zeros((na + 1, nb + 1), dtype=np.int8)

    m_prev = [NEG] * (nb + 1)
    x_prev = [NEG] * (nb + 1)
    y_prev = [NEG] * (nb + 1)
    m_prev[0] = 0.0
    for j in range(1, nb + 1):
        y_prev[j] = go + ge * (j - 1)
        ptr_y[0, j] = 0 if j == 1 else 2

    for i in range(1, na + 1):
        ai = a[i - 1]
        m_cur = [NEG] * (nb + 1)
        x_cur = [NEG] * (nb + 1)
        y_cur = [NEG] * (nb + 1)
        x_cur[0] = go + ge * (i - 1)
        ptr_x[i, 0] = 0 if i == 1 else 1
        for j in range(1, nb + 1):
            s = match if ai == b[j - 1] else mismatch

            # M: extend any state diagonally
            best, src = m_prev[j - 1], 0
            if x_prev[j - 1] > best:
                best, src = x_prev[j - 1], 1
            if y_prev[j - 1] > best:
                best, src = y_prev[j - 1], 2
            m_cur[j] = best + s
            ptr_m[i, j] = src

            # X: gap in b, consume a[i-1]
            best, src = m_prev[j] + go, 0
            cand = x_prev[j] + ge
            if cand > best:
                best, src = cand, 1
            cand = y_prev[j] + go
            if cand > best:
                best, src = cand, 2
            x_cur[j] = best
            ptr_x[i, j] = src

            # Y: gap in a, consume b[j-1]
            best, src = m_cur[j - 1] + go, 0
            cand = y_cur[j - 1] + ge
            if cand > best:
                best, src = cand, 2
            cand = x_cur[j - 1] + go
            if cand > best:
                best, src = cand, 1
            y_cur[j] = best
            ptr_y[i, j] = src

        m_prev, x_prev, y_prev = m_cur, x_cur, y_cur

    ends = (m_prev[nb], x_prev[nb], y_prev[nb])
    state = int(np.argmax(ends))
    pairs = []
    i, j = na, nb
    while i > 0 or j > 0:
        if state == 0:
            nxt = ptr_m[i, j]
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif state == 1:
            nxt = ptr_x[i, j]
            pairs.append((i - 1, None))
            i -= 1
        else:
            nxt = ptr_y[i, j]
            pairs.append((None, j - 1))
            j -= 1
        if i == 0 and j == 0:
            break
        if i == 0:
            state = 2
        elif j == 0:
            state = 1
        else:
            state = int(nxt)
    pairs.reverse()
    return pairs


def align_to_query(pdb_seq, query_seq, return_stats=False):
    """Map every query position onto a pdb_seq index, or None where unobserved.

    Returns a list of length len(query_seq). Entry k is the index into `pdb_seq` (and hence
    into the coordinate array from `ca_cb_coords`) of the residue that corresponds to query
    position k, or None if that position is not present in the structure.

    Strategy, in order:
      1. query_seq is a substring of pdb_seq       -> exact, offset mapping
      2. pdb_seq is a substring of query_seq       -> exact, structure is a fragment
      3. Needleman-Wunsch, identity scoring, gap -1

    With return_stats=True, also returns a dict with `method`, `n_mapped`, `n_identical`
    (aligned positions whose residue letters agree) and `identity` (n_identical /
    n_mapped). CHECK THESE. A low identity means the deposited construct is a mutant or a
    different species' ortholog; the mapping may still be correct but the contact map is
    then a homolog's, not the query's.
    """
    mapping = [None] * len(query_seq)
    method = None

    pos = pdb_seq.find(query_seq)
    if pos >= 0:
        method = "exact:query-in-pdb"
        for k in range(len(query_seq)):
            mapping[k] = pos + k
    else:
        pos = query_seq.find(pdb_seq)
        if pos >= 0:
            method = "exact:pdb-in-query"
            for k in range(len(pdb_seq)):
                mapping[pos + k] = k
        else:
            method = "needleman-wunsch"
            for qi, pi in _needleman_wunsch(query_seq, pdb_seq):
                if qi is not None and pi is not None:
                    mapping[qi] = pi

    if not return_stats:
        return mapping

    n_mapped = sum(1 for m in mapping if m is not None)
    n_ident = sum(1 for k, m in enumerate(mapping)
                  if m is not None and query_seq[k] == pdb_seq[m])
    stats = {
        "method": method,
        "n_mapped": n_mapped,
        "n_query": len(query_seq),
        "n_pdb": len(pdb_seq),
        "n_identical": n_ident,
        "identity": (n_ident / float(n_mapped)) if n_mapped else 0.0,
        "coverage": n_mapped / float(len(query_seq)) if query_seq else 0.0,
        "mismatches": [(k, query_seq[k], pdb_seq[m]) for k, m in enumerate(mapping)
                       if m is not None and query_seq[k] != pdb_seq[m]],
    }
    return mapping, stats


# --------------------------------------------------------------------------------------
# 5. scoring
# --------------------------------------------------------------------------------------

def _normalise_mapping(mapping, length):
    arr = np.full(length, -1, dtype=int)
    for k, m in enumerate(mapping[:length]):
        if m is not None and m >= 0:
            arr[k] = int(m)
    return arr


def precision_at_topk(scores, true_contacts, mapping, min_sep=12, top_frac=0.5,
                      return_details=False):
    """Long-range contact precision of `scores` against an experimental contact map.

    scores          (Lq, Lq) predicted coupling / contact scores in QUERY coordinates
    true_contacts   (Lp, Lp) boolean map in PDB coordinates (from `contact_map`)
    mapping         query -> pdb index list from `align_to_query` (None = unobserved)

    Candidate pairs are i < j with j - i >= min_sep AND both positions mapped; unmapped
    positions are dropped entirely rather than counted as non-contacts, since "we did not
    observe it" is not evidence against a contact. The top k = max(1, int(Lq * top_frac))
    candidates by score are compared with the true map.

    Returns (precision, n_eval) where n_eval is the number of top pairs actually scored
    (= min(k, number of evaluable candidates)). Note that k is set from the FULL query
    length, so a chain with many unobserved positions yields the same k but fewer
    candidates; if n_eval < k the denominator shrank and the number is not comparable to a
    fully observed target without saying so.
    """
    scores = np.asarray(scores, dtype=float)
    true_contacts = np.asarray(true_contacts, dtype=bool)
    length = scores.shape[0]
    idx = _normalise_mapping(mapping, length)

    ii, jj = np.triu_indices(length, k=min_sep)
    keep = (idx[ii] >= 0) & (idx[jj] >= 0)
    ii, jj = ii[keep], jj[keep]
    n_cand = int(ii.size)

    k = max(1, int(length * top_frac))
    if n_cand == 0:
        result = (float("nan"), 0)
        if return_details:
            return result + ({"k": k, "n_candidates": 0, "n_true_hits": 0},)
        return result

    vals = scores[ii, jj]
    order = np.argsort(-vals, kind="stable")[:k]
    pi, pj = idx[ii[order]], idx[jj[order]]
    hits = true_contacts[pi, pj]
    n_eval = int(order.size)
    prec = float(hits.sum()) / n_eval

    if return_details:
        return prec, n_eval, {"k": k, "n_candidates": n_cand,
                              "n_true_hits": int(hits.sum())}
    return prec, n_eval

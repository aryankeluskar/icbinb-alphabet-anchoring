"""FastAPI backend for the alphabet-anchoring explainer.

Loads real ESM-2 and Potts models, serves contact maps and overlap scores
to the HTML explainer. Keep the GPU free when not viewing: run warmup.py
before opening the explainer, and kill this server when done.

Usage:
    source /scratch/author/icbinb/src/activate_env.sh
    python /scratch/author/icbinb/src/server.py --model esm2_t33_650M_UR50D
    # or, for a smaller model that loads faster:
    python /scratch/author/icbinb/src/server.py --model esm2_t30_150M_UR50D

The server listens on http://localhost:8765 and serves CORS headers so the
HTML file (opened from disk) can connect.
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Protein registry — sequences, MSAs, and PDB structures
# ---------------------------------------------------------------------------

MSA_DIR = os.path.join(ROOT, "data", "proteingym", "DMS_msa_files", "DMS_msa_files")
PDB_DIR = os.path.join(ROOT, "data", "pdb")

PROTEINS: Dict[str, Dict[str, Any]] = {
    "ubiquitin": {
        "seq": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "pdb": ("1UBQ", "A"),
        "msa": "RL40A_YEAST_full_11-26-2021_b01.a2m",
        "description": "Ubiquitin — 76 residues, 100% identity to 1UBQ chain A. The paper's headline target.",
    },
    "protein_G_B1": {
        "seq": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        "pdb": ("1PGB", "A"),
        "msa": "SPG1_STRSG_full_b0.1.a2m",
        "description": "Protein G B1 domain — 56 residues, 100% identity to 1PGB chain A.",
    },
    "cytochrome_c": {
        "seq": "GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",
        "pdb": ("1HRC", "A"),
        "msa": None,
        "description": "Cytochrome c — 104 residues. Human sequence; 1HRC is horse heart (88.5% identity). No MSA on disk.",
    },
    "lysozyme_T4": {
        "seq": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
        "pdb": ("2LZM", "A"),
        "msa": None,
        "description": "T4 lysozyme — 164 residues. Pseudo-wild-type vs true WT 2LZM (differs at T54C/A97C).",
    },
}


def read_a2m(path: str) -> List[str]:
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


def load_msa(protein_name: str) -> Optional[List[str]]:
    """Load the MSA for a protein, if one is on disk."""
    info = PROTEINS.get(protein_name)
    if not info or not info.get("msa"):
        return None
    path = os.path.join(MSA_DIR, info["msa"])
    if not os.path.exists(path):
        return None
    return read_a2m(path)


# ---------------------------------------------------------------------------
# Model holder — loaded lazily via /api/warmup
# ---------------------------------------------------------------------------

import permute  # noqa: E402
import mrf      # noqa: E402

MIN_SEP = 12
TOP_FRAC = 0.5

_state: Dict[str, Any] = {
    "model_name": None,
    "jacobian_wrapper": None,
    "device": "cpu",
    "loaded": False,
    "load_time": None,
    "n_params": None,
    "msa_cache": {},
    "potts_cache": {},
}


def _get_msa(protein_name: str) -> Optional[List[str]]:
    if protein_name not in _state["msa_cache"]:
        _state["msa_cache"][protein_name] = load_msa(protein_name)
    return _state["msa_cache"][protein_name]


def _get_potts_contacts(protein_name: str) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
    """Compute Potts contact map (and top-L/2 set) for a protein, with caching."""
    if protein_name in _state["potts_cache"]:
        return _state["potts_cache"][protein_name]
    msa = _get_msa(protein_name)
    if not msa:
        return None
    seq = PROTEINS[protein_name]["seq"]
    L = len(seq)
    # Locate the query region in the MSA (ungapped slide)
    query_match = msa[0]
    query_ungapped = "".join(c for c in query_match if c in permute.AA20)
    # Find where the query sits
    pos = query_ungapped.find(seq)
    if pos < 0:
        pos = seq.find(query_ungapped[:50])
        if pos < 0:
            return None
        q_start = pos
        q_end = pos + len(query_ungapped)
    else:
        q_start = pos
        q_end = pos + len(seq)

    # Extract the match columns covering the query region
    match_cols = [i for i, c in enumerate(query_match) if c.isupper() or c == "-"]
    if len(match_cols) != len(query_ungapped):
        # Fall back: just use the first sequence's uppercase positions
        pass

    # Build a sub-MSA aligned to the query sequence positions
    sub_msa = []
    for s in msa:
        match_seq = "".join(c for c in s if c.isupper() or c == "-")
        if len(match_seq) > q_end:
            sub = match_seq[q_start:q_end]
            if sub.count("-") < L * 0.5:
                sub_msa.append(sub)

    if len(sub_msa) < 10:
        return None

    scores = mrf.contact_map(sub_msa)
    from jacobian import top_contacts as _top_contacts
    top = _top_contacts(scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
    result = (top, scores)
    _state["potts_cache"][protein_name] = result
    return result


def _compute_esm_contacts(seq: str) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Run ESM-2 Jacobian and return (top-L/2 contacts, full score matrix)."""
    jac_mod = _state["jacobian_wrapper"]
    scores = jac_mod.contact_map(seq)
    from jacobian import top_contacts as _top_contacts
    top = _top_contacts(scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
    return top, scores


def _compute_overlap(base_top, perm_top) -> float:
    sa, sb = set(base_top), set(perm_top)
    if not sa:
        return float("nan")
    return len(sa & sb) / float(len(sa))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="Alphabet Anchoring Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuditRequest(BaseModel):
    protein: str = "ubiquitin"
    regime: str = "random"
    m: int = 20
    seed: int = 0
    model: Optional[str] = None


class WarmupRequest(BaseModel):
    model: str = "esm2_t33_650M_UR50D"


@app.get("/api/status")
def status():
    return {
        "loaded": _state["loaded"],
        "model_name": _state["model_name"],
        "device": _state["device"],
        "n_params": _state["n_params"],
        "load_time": _state["load_time"],
        "proteins": {
            name: {
                "seq": info["seq"],
                "length": len(info["seq"]),
                "has_msa": info.get("msa") is not None,
                "description": info["description"],
            }
            for name, info in PROTEINS.items()
        },
    }


@app.post("/api/warmup")
def warmup(req: WarmupRequest):
    if _state["loaded"] and _state["model_name"] == req.model:
        return {"status": "already_loaded", "model": req.model}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    try:
        import esm
        model, alphabet = getattr(esm.pretrained, req.model)()
        from jacobian import ESM2Jacobian
        wrapper = ESM2Jacobian(model, alphabet, device=device, batch_size=32)
        _state["jacobian_wrapper"] = wrapper
        _state["model_name"] = req.model
        _state["device"] = device
        _state["loaded"] = True
        _state["load_time"] = time.time() - t0
        _state["n_params"] = sum(p.numel() for p in model.parameters())
        return {
            "status": "loaded",
            "model": req.model,
            "device": device,
            "n_params": _state["n_params"],
            "load_time": round(_state["load_time"], 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.get("/api/proteins")
def get_proteins():
    return {
        name: {
            "seq": info["seq"],
            "length": len(info["seq"]),
            "has_msa": info.get("msa") is not None,
            "description": info["description"],
        }
        for name, info in PROTEINS.items()
    }


@app.post("/api/audit")
def run_audit(req: AuditRequest):
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /api/warmup first.")
    if req.protein not in PROTEINS:
        raise HTTPException(status_code=400, detail=f"Unknown protein: {req.protein}")
    if req.regime not in ("random", "In-class", "Cross-class"):
        raise HTTPException(status_code=400, detail=f"Unknown regime: {req.regime}")

    seq = PROTEINS[req.protein]["seq"]
    L = len(seq)

    # Build permutation
    rng = np.random.default_rng(1000 * req.seed + req.m)
    builder = permute.PERM_BUILDERS[req.regime]
    perm = builder(req.m, rng)
    permuted_seq = permute.apply_perm(seq, perm)
    realized_m = sum(1 for a, b in perm.items() if a != b)

    # ESM-2: compute contact maps
    t0 = time.time()
    base_top, base_scores = _compute_esm_contacts(seq)
    esm_time = time.time() - t0

    t1 = time.time()
    perm_top, perm_scores = _compute_esm_contacts(permuted_seq)
    esm_perm_time = time.time() - t1

    esm_overlap = _compute_overlap(base_top, perm_top)

    # Potts: compute if MSA available
    potts_result = _get_potts_contacts(req.protein)
    potts_overlap = None
    potts_top = None
    potts_perm_top = None

    if potts_result is not None:
        potts_top, potts_scores = potts_result
        # Potts is invariant by theorem — recompute to verify
        msa = _get_msa(req.protein)
        perm_msa = permute.apply_perm_msa(msa, perm)
        potts_perm_scores = mrf.contact_map(perm_msa)
        from jacobian import top_contacts as _top_contacts
        potts_perm_top = _top_contacts(potts_perm_scores, min_sep=MIN_SEP, top_frac=TOP_FRAC)
        potts_overlap = _compute_overlap(potts_top, potts_perm_top)

    # Ground truth contacts (if PDB available)
    true_contacts = None
    pdb_info = PROTEINS[req.protein].get("pdb")
    if pdb_info:
        pdb_id, chain = pdb_info
        cif_path = os.path.join(PDB_DIR, f"{pdb_id}.cif")
        if os.path.exists(cif_path):
            try:
                import structures
                pdb_seq, coords = structures.ca_cb_coords(cif_path, chain)
                cmap = structures.contact_map(coords, cutoff=8.0)
                mapping = structures.align_to_query(pdb_seq, seq)
                true_set = set()
                for i in range(L):
                    for j in range(i + MIN_SEP, L):
                        mi, mj = mapping[i], mapping[j]
                        if mi is not None and mj is not None and cmap[mi, mj]:
                            true_set.add((i, j))
                true_contacts = list(true_set)
            except Exception:
                pass

    # Serialize contact sets as lists of [i, j] pairs
    def serialize_pairs(pairs):
        return [[int(i), int(j)] for i, j in pairs]

    # Extract score matrix for visualization (top-L/2 region only, to keep payload small)
    # Send a downsampled version: for each pair (i,j) with |i-j| >= MIN_SEP, send the score
    n_pairs = sum(1 for i in range(L) for j in range(i + MIN_SEP, L))
    # For large proteins, subsample to keep response < 1MB
    max_pairs = 2000
    pair_indices = []
    pair_scores_base = []
    pair_scores_perm = []
    step = max(1, n_pairs // max_pairs)
    count = 0
    for i in range(L):
        for j in range(i + MIN_SEP, L):
            if count % step == 0:
                pair_indices.append([i, j])
                pair_scores_base.append(float(base_scores[i, j]))
                pair_scores_perm.append(float(perm_scores[i, j]))
            count += 1

    return {
        "protein": req.protein,
        "regime": req.regime,
        "m_requested": req.m,
        "m_realized": realized_m,
        "seed": req.seed,
        "sequence": seq,
        "permuted_sequence": permuted_seq,
        "permutation": {a: b for a, b in perm.items() if a != b},
        "length": L,
        "esm": {
            "overlap": float(esm_overlap),
            "base_top": serialize_pairs(base_top),
            "perm_top": serialize_pairs(perm_top),
            "compute_time": round(esm_time + esm_perm_time, 2),
            "pair_scores_base": pair_scores_base,
            "pair_scores_perm": pair_scores_perm,
            "pair_indices": pair_indices,
        },
        "potts": {
            "available": potts_result is not None,
            "overlap": float(potts_overlap) if potts_overlap is not None else None,
            "base_top": serialize_pairs(potts_top) if potts_top else [],
            "perm_top": serialize_pairs(potts_perm_top) if potts_perm_top else [],
        },
        "ground_truth": {
            "available": true_contacts is not None,
            "contacts": serialize_pairs(true_contacts) if true_contacts else [],
        },
        "model": _state["model_name"],
        "n_params": _state["n_params"],
        "device": _state["device"],
    }


@app.post("/api/unload")
def unload():
    """Free the GPU."""
    if _state["jacobian_wrapper"] is not None:
        import torch
        del _state["jacobian_wrapper"].model
        _state["jacobian_wrapper"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _state["loaded"] = False
    _state["model_name"] = None
    _state["n_params"] = None
    return {"status": "unloaded"}


@app.get("/", response_class=HTMLResponse)
def serve_explainer():
    """Serve the interactive HTML explainer."""
    candidates = [
        os.path.join("/tmp", f)
        for f in os.listdir("/tmp")
        if f.startswith("2026-") and f.endswith("-explanation-alphabet-anchoring.html")
    ]
    if not candidates:
        return HTMLResponse(
            "<h1>Explainer HTML not found</h1>"
            "<p>Run the explain skill to generate it first.</p>",
            status_code=404,
        )
    candidates.sort(reverse=True)  # most recent first
    return FileResponse(candidates[0], media_type="text/html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Alphabet anchoring explorer backend")
    ap.add_argument("--model", default="esm2_t33_650M_UR50D",
                    help="ESM-2 model name (default: esm2_t33_650M_UR50D)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-warmup", action="store_true",
                    help="Don't load the model on startup; wait for /api/warmup")
    args = ap.parse_args()

    if not args.no_warmup:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        print(f"Loading {args.model} on {device}...", flush=True)
        try:
            import esm
            model, alphabet = getattr(esm.pretrained, args.model)()
            from jacobian import ESM2Jacobian
            wrapper = ESM2Jacobian(model, alphabet, device=device, batch_size=32)
            _state["jacobian_wrapper"] = wrapper
            _state["model_name"] = args.model
            _state["device"] = device
            _state["loaded"] = True
            _state["load_time"] = time.time() - t0
            _state["n_params"] = sum(p.numel() for p in model.parameters())
            print(f"  Loaded in {_state['load_time']:.1f}s "
                  f"({_state['n_params'] / 1e6:.0f}M params, {device})", flush=True)
        except Exception as e:
            print(f"  WARNING: model load failed: {e}", flush=True)
            print(f"  Server will start anyway; POST /api/warmup to retry.", flush=True)

    print(f"\nServer starting on http://localhost:{args.port}", flush=True)
    print(f"  Open http://localhost:{args.port}/ in your browser for the explainer.", flush=True)
    print(f"  (VS Code auto-forwards this port to your Mac.)", flush=True)
    print(f"  Press Ctrl+C to stop (frees the GPU).\n", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

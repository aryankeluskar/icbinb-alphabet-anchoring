#!/usr/bin/env python
"""
make_fair_esm_weights.py
========================
Rebuild fair-esm-format ESM-2 checkpoints from the HuggingFace weights.

WHY THIS EXISTS  (fallback / disaster-recovery -- normally NOT needed)
---------------------------------------------------------------------
fair-esm downloads its checkpoints from https://dl.fbaipublicfiles.com/fair-esm/.
On 2026-08-05 that host intermittently served HTTP 403 AccessDenied for ~20
minutes before recovering, which makes `esm.pretrained.esm2_*()` fail hard with
no fallback. If that happens again (or the bucket is finally retired), run this
script: HuggingFace hosts the *same* weights as `facebook/esm2_*`, just under
HF's `EsmForMaskedLM` parameter names.

HF's ESM is a faithful 1:1 port of fair-esm -- identical alphabet, identical
ordering, identical maths -- so the two state dicts differ only by key names.
This script downloads the HF weights, renames the keys back to fair-esm layout,
and writes a `<name>.pt` + `<name>-contact-regression.pt` pair.

Correctness is checked by loading the *converted* file back through fair-esm's
own `load_model_and_alphabet_local` and comparing logits against HF (<1e-3).

Usage:
    python make_fair_esm_weights.py                       # 150M + 650M
    python make_fair_esm_weights.py esm2_t6_8M_UR50D      # one model
    python make_fair_esm_weights.py --install <name>      # also copy into the
                                                          # torch-hub cache so
                                                          # esm.pretrained.<name>()
                                                          # picks it up offline
"""
import argparse
import os
import sys

PROJ = "/scratch/author/icbinb"
CACHE = os.path.join(PROJ, ".cache")
os.environ["HF_HOME"] = os.path.join(CACHE, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE, "torch")
os.environ["XDG_CACHE_HOME"] = CACHE
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HUGGINGFACE_HUB_CACHE", None)  # ~/.bashrc points it elsewhere
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE, "huggingface", "hub")

import torch  # noqa: E402
import esm  # noqa: E402
from esm.model.esm2 import ESM2  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

# fair-esm name -> (HF repo id, num_layers, embed_dim, attention_heads)
MODELS = {
    "esm2_t6_8M_UR50D":     ("facebook/esm2_t6_8M_UR50D",     6,  320,  20),
    "esm2_t12_35M_UR50D":   ("facebook/esm2_t12_35M_UR50D",   12, 480,  20),
    "esm2_t30_150M_UR50D":  ("facebook/esm2_t30_150M_UR50D",  30, 640,  20),
    "esm2_t33_650M_UR50D":  ("facebook/esm2_t33_650M_UR50D",  33, 1280, 20),
    "esm2_t36_3B_UR50D":    ("facebook/esm2_t36_3B_UR50D",    36, 2560, 40),
}
# NOTE: fair-esm's "150M" checkpoint is `esm2_t30_150M_UR50D`. The task text says
# `esm2_t8_150M_UR50D`, which does not exist upstream; we alias it so that name
# also works (see ALIASES below).
ALIASES = {"esm2_t8_150M_UR50D": "esm2_t30_150M_UR50D"}

# Converted files live here. They are only *used* if copied/symlinked into the
# real torch-hub cache dir (torch.hub.get_dir()/checkpoints), which `--install`
# does -- so a healthy genuine download is never silently shadowed.
CKPT_DIR = os.path.join(PROJ, ".env_build", "fair_esm_from_hf")


def convert(name: str) -> str:
    hf_id, n_layers, embed_dim, n_heads = MODELS[name]
    out_pt = os.path.join(CKPT_DIR, f"{name}.pt")
    out_reg = os.path.join(CKPT_DIR, f"{name}-contact-regression.pt")
    if os.path.exists(out_pt) and os.path.exists(out_reg):
        print(f"[{name}] already present, skipping")
        return out_pt

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[{name}] downloading HF weights from {hf_id} ...", flush=True)
    hf = AutoModelForMaskedLM.from_pretrained(hf_id)
    hf.eval()
    hf_sd = hf.state_dict()

    # Build an empty fair-esm model purely to get the canonical key set and the
    # non-learned buffers (rotary inv_freq etc.) exactly right.
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    fe = ESM2(num_layers=n_layers, embed_dim=embed_dim,
              attention_heads=n_heads, alphabet=alphabet, token_dropout=True)
    tgt = fe.state_dict()

    assert fe.embed_tokens.weight.shape == hf_sd["esm.embeddings.word_embeddings.weight"].shape, \
        "alphabet size mismatch between fair-esm and HF -- vocabularies differ!"

    def put(dst, src):
        assert dst in tgt, f"unexpected fair-esm key {dst}"
        assert src in hf_sd, f"missing HF key {src}"
        assert tgt[dst].shape == hf_sd[src].shape, \
            f"shape mismatch {dst}{tuple(tgt[dst].shape)} vs {src}{tuple(hf_sd[src].shape)}"
        tgt[dst] = hf_sd[src].clone()
        done.add(dst)

    done = set()
    put("embed_tokens.weight", "esm.embeddings.word_embeddings.weight")
    for i in range(n_layers):
        f, h = f"layers.{i}", f"esm.encoder.layer.{i}"
        for w in ("weight", "bias"):
            put(f"{f}.self_attn.q_proj.{w}", f"{h}.attention.self.query.{w}")
            put(f"{f}.self_attn.k_proj.{w}", f"{h}.attention.self.key.{w}")
            put(f"{f}.self_attn.v_proj.{w}", f"{h}.attention.self.value.{w}")
            put(f"{f}.self_attn.out_proj.{w}", f"{h}.attention.output.dense.{w}")
            put(f"{f}.self_attn_layer_norm.{w}", f"{h}.attention.LayerNorm.{w}")
            put(f"{f}.fc1.{w}", f"{h}.intermediate.dense.{w}")
            put(f"{f}.fc2.{w}", f"{h}.output.dense.{w}")
            put(f"{f}.final_layer_norm.{w}", f"{h}.LayerNorm.{w}")
    for w in ("weight", "bias"):
        put(f"emb_layer_norm_after.{w}", f"esm.encoder.emb_layer_norm_after.{w}")
        put(f"lm_head.dense.{w}", f"lm_head.dense.{w}")
        put(f"lm_head.layer_norm.{w}", f"lm_head.layer_norm.{w}")
        put(f"contact_head.regression.{w}", f"esm.contact_head.regression.{w}")
    put("lm_head.weight", "lm_head.decoder.weight")
    put("lm_head.bias", "lm_head.bias")

    # Rotary inv_freq: do NOT just let the fresh model recompute it. The value
    # baked into Meta's original checkpoints differs from
    # 1/(10000**(arange(0,d,2)/d)) by ~6.6e-5 (it was generated on an older
    # torch), and that alone shifts logits by ~3.5e-3. HF preserved the exact
    # original tensor, so copy it. transformers 5.x keeps one shared buffer at
    # `esm.rotary_embeddings.inv_freq`; older versions had one per layer.
    inv = hf_sd.get("esm.rotary_embeddings.inv_freq")
    leftover = sorted(set(tgt) - done)
    for k in leftover:
        assert "inv_freq" in k, f"unmapped learned param: {k}"
        per_layer = hf_sd.get(
            f"esm.encoder.layer.{k.split('.')[1]}.attention.self.rotary_embeddings.inv_freq")
        src = per_layer if per_layer is not None else inv
        if src is not None:
            assert src.shape == tgt[k].shape
            tgt[k] = src.clone()
    print(f"[{name}] mapped {len(done)} tensors, "
          f"{len(leftover)} rotary buffers "
          f"({'from HF' if inv is not None else 'RECOMPUTED -- expect ~3e-3 drift'})")

    # fair-esm's loader reads model_data['cfg']['model'].<attr>, and merges the
    # separate contact-regression file into model_data['model'] before a
    # strict=True load. So we emit exactly that pair of files.
    cfg = argparse.Namespace(
        encoder_layers=n_layers,
        encoder_embed_dim=embed_dim,
        encoder_attention_heads=n_heads,
        token_dropout=True,
    )
    reg = {k: tgt.pop(k) for k in ("contact_head.regression.weight",
                                   "contact_head.regression.bias")}
    torch.save({"cfg": {"model": cfg}, "model": tgt}, out_pt)
    torch.save({"model": reg}, out_reg)
    print(f"[{name}] wrote {out_pt} ({os.path.getsize(out_pt)/2**20:.0f} MiB)")
    print(f"[{name}] wrote {out_reg}")

    del hf, hf_sd, fe, tgt
    return out_pt


def verify(name: str):
    """Load the CONVERTED file through fair-esm and compare against HF."""
    hf_id = MODELS[name][0]
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

    # GOTCHA: torch >= 2.6 flipped torch.load's `weights_only` default to True,
    # and fair-esm 2.0.0's load_model_and_alphabet_local() calls torch.load with
    # no argument. ESM checkpoints carry an argparse.Namespace config object, so
    # the load dies with UnpicklingError. Allow-list that one class.
    # (esm.pretrained.esm2_*() is unaffected -- it goes through
    #  torch.hub.load_state_dict_from_url, which still defaults weights_only=False.)
    torch.serialization.add_safe_globals([argparse.Namespace])

    # load_model_and_alphabet_local reads the file we just wrote (and its
    # sibling -contact-regression.pt) with a strict=True state_dict load, so any
    # missing/renamed/misshaped key would raise here.
    m, alpha = esm.pretrained.load_model_and_alphabet_local(
        os.path.join(CKPT_DIR, f"{name}.pt"))
    m.eval()
    _, _, toks = alpha.get_batch_converter()([("p", seq)])
    with torch.no_grad():
        o = m(toks, repr_layers=[m.num_layers], return_contacts=True)
    a, contacts = o["logits"], o["contacts"]
    del m, o

    tok = AutoTokenizer.from_pretrained(hf_id)
    hm = AutoModelForMaskedLM.from_pretrained(hf_id).eval()
    with torch.no_grad():
        b = hm(**tok(seq, return_tensors="pt")).logits
    del hm

    d = (a - b).abs().max().item()
    print(f"[{name}] converted-file logits vs HF: max|diff| = {d:.3e}  "
          f"| contacts {tuple(contacts.shape)}  "
          f"{'OK' if d < 1e-3 else 'MISMATCH!!'}")
    assert d < 1e-3, "converted weights do not reproduce HF outputs!"


def install(name: str):
    """Copy converted files into the torch-hub cache fair-esm downloads into."""
    import shutil
    dst_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(dst_dir, exist_ok=True)
    for suffix in (".pt", "-contact-regression.pt"):
        src = os.path.join(CKPT_DIR, name + suffix)
        dst = os.path.join(dst_dir, name + suffix)
        if os.path.exists(dst):
            print(f"[{name}] {dst} already exists (genuine download?) -- not overwriting")
            continue
        shutil.copy2(src, dst)
        print(f"[{name}] installed -> {dst}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--install"]
    do_install = "--install" in sys.argv
    names = [ALIASES.get(n, n) for n in (args or
             ["esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])]
    for n in names:
        convert(n)
        verify(n)
        if do_install:
            install(n)
    print("\nAll checkpoints converted and numerically verified.")
    print(f"Written to: {CKPT_DIR}")
    if not do_install:
        print("Re-run with --install to place them in the torch-hub cache.")

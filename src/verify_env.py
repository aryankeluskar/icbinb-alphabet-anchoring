#!/usr/bin/env python
"""
verify_env.py -- End-to-end proof that the icbinb ESM-2 environment works on GPU.

Checks, in order:
  1. torch version / CUDA availability / GPU name
  2. fair-esm  esm2_t30_150M_UR50D  -> forward pass on GPU, logits + contact map
  3. fair-esm  esm2_t33_650M_UR50D  -> forward pass on GPU, logits + contact map
  4. transformers AutoModelForMaskedLM("facebook/esm2_t33_650M_UR50D") -> forward pass

All caches (HF hub + torch hub) are pinned under /scratch/author/icbinb/.cache
so that nothing is ever written to $HOME.

Run:  bash /scratch/author/icbinb/src/activate_env.sh   # or source it
      python /scratch/author/icbinb/src/verify_env.py
"""

# --- Pin caches to /scratch BEFORE importing torch / transformers -----------
import os

PROJ = "/scratch/author/icbinb"
CACHE = os.path.join(PROJ, ".cache")
os.environ["HF_HOME"] = os.path.join(CACHE, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE, "torch")
os.environ["XDG_CACHE_HOME"] = CACHE
os.environ["MPLCONFIGDIR"] = os.path.join(CACHE, "matplotlib")
# The legacy vars below OVERRIDE HF_HOME inside huggingface_hub, and ~/.bashrc
# points both at /scratch/author/deceptive-llms/models/ (another project).
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE, "huggingface", "hub")
for _d in (os.environ["HF_HOME"], os.environ["TORCH_HOME"], os.environ["MPLCONFIGDIR"], os.environ["HF_HUB_CACHE"]):
    os.makedirs(_d, exist_ok=True)

# A conda env is not a venv, so ~/.local/lib/python3.11/site-packages would
# otherwise shadow this env's packages. Fail loudly rather than silently import
# the wrong scipy/tqdm/esm from $HOME.
if any(".local" in p for p in __import__("sys").path):
    raise SystemExit(
        "REFUSING TO RUN: ~/.local site-packages is on sys.path.\n"
        "Set PYTHONNOUSERSITE=1 (source src/activate_env.sh)."
    )

import sys
import time

import torch


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------- 1. torch --
hr("1. TORCH / CUDA")
print(f"python                     : {sys.version.split()[0]}  ({sys.executable})")
print(f"torch.__version__          : {torch.__version__}")
print(f"torch.version.cuda         : {torch.version.cuda}")
print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
print(f"torch.cuda.is_available()  : {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA is NOT available -- are you on a GPU node?"
print(f"torch.cuda.device_count()  : {torch.cuda.device_count()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"compute capability         : sm_{cc[0]}{cc[1]}")
free_b, total_b = torch.cuda.mem_get_info(0)
print(f"GPU memory free/total      : {free_b/2**30:.1f} GiB / {total_b/2**30:.1f} GiB")

device = torch.device("cuda:0")

# tiny sanity matmul on the GPU
_a = torch.randn(512, 512, device=device)
print(f"sanity matmul on GPU       : {(_a @ _a).sum().item():.3f}  -> OK")

# ------------------------------------------------------------- 2/3. fair-esm --
import esm  # noqa: E402

print(f"\nfair-esm module            : {esm.__file__}")
assert hasattr(esm, "pretrained") and hasattr(
    esm.pretrained, "esm2_t33_650M_UR50D"
), "`import esm` did NOT resolve to fair-esm (wrong `esm` package installed?)"
print("esm.pretrained.esm2_t33_650M_UR50D resolves -> correct `fair-esm` package")

SEQS = [
    ("prot1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("prot2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]


def run_fair_esm(loader_name):
    hr(f"fair-esm: {loader_name}")
    t0 = time.time()
    model, alphabet = getattr(esm.pretrained, loader_name)()
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time()-t0:.1f}s | params = {n_params/1e6:.1f}M "
          f"| layers = {model.num_layers} | embed_dim = {model.embed_dim}")

    batch_converter = alphabet.get_batch_converter()
    _labels, _strs, tokens = batch_converter(SEQS)
    tokens = tokens.to(device)
    print(f"input tokens shape         : {tuple(tokens.shape)}  (batch, seq_len)")

    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers], return_contacts=True)

    logits = out["logits"]
    contacts = out["contacts"]
    reps = out["representations"][model.num_layers]
    print(f"LOGITS shape               : {tuple(logits.shape)}   dev={logits.device}")
    print(f"CONTACT-MAP shape          : {tuple(contacts.shape)}   dev={contacts.device}")
    print(f"representations shape      : {tuple(reps.shape)}")
    print(f"logits  [min, max]         : [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"contacts[min, max]         : [{contacts.min():.4f}, {contacts.max():.4f}]")
    assert logits.is_cuda and contacts.is_cuda, "output not on GPU!"
    assert torch.isfinite(logits).all(), "non-finite logits!"
    assert contacts.shape[0] == len(SEQS)
    print(f"peak GPU mem               : {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    del model, out, logits, contacts, reps
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"--> {loader_name} GPU forward pass OK")


# NOTE: there is no `esm2_t8_150M_UR50D` in fair-esm. The ESM-2 line is
# t6_8M / t12_35M / t30_150M / t33_650M / t36_3B / t48_15B -- the 150M model is
# `esm2_t30_150M_UR50D` (t8 and 150M got crossed in the request).
run_fair_esm("esm2_t30_150M_UR50D")
run_fair_esm("esm2_t33_650M_UR50D")

# ------------------------------------------------------ 4. HF transformers --
hr("4. HuggingFace transformers: facebook/esm2_t33_650M_UR50D")
import transformers  # noqa: E402
from transformers import AutoTokenizer, AutoModelForMaskedLM  # noqa: E402

print(f"transformers.__version__   : {transformers.__version__}")
print(f"HF_HOME                    : {os.environ['HF_HOME']}")

HF_ID = "facebook/esm2_t33_650M_UR50D"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(HF_ID)
hf_model = AutoModelForMaskedLM.from_pretrained(HF_ID).to(device).eval()
print(f"loaded in {time.time()-t0:.1f}s | params = "
      f"{sum(p.numel() for p in hf_model.parameters())/1e6:.1f}M")

batch = tok([s for _, s in SEQS], return_tensors="pt", padding=True).to(device)
print(f"input_ids shape            : {tuple(batch['input_ids'].shape)}")
with torch.no_grad():
    hf_out = hf_model(**batch, output_hidden_states=True)
print(f"LOGITS shape               : {tuple(hf_out.logits.shape)}   dev={hf_out.logits.device}")
print(f"last hidden state shape    : {tuple(hf_out.hidden_states[-1].shape)}")
print(f"logits [min, max]          : [{hf_out.logits.min():.3f}, {hf_out.logits.max():.3f}]")
assert hf_out.logits.is_cuda and torch.isfinite(hf_out.logits).all()
print(f"peak GPU mem               : {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

# masked-token prediction as a semantic smoke test
masked = SEQS[0][1][:20] + tok.mask_token + SEQS[0][1][21:]
mb = tok(masked, return_tensors="pt").to(device)
with torch.no_grad():
    ml = hf_model(**mb).logits
mask_pos = (mb["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
top5 = ml[0, mask_pos].topk(5)
print("masked-LM top-5 at pos 20  : "
      + ", ".join(f"{tok.convert_ids_to_tokens(i.item())}({p:.2f})"
                  for p, i in zip(top5.values, top5.indices)))
print(f"(ground truth residue was '{SEQS[0][1][20]}')")
print("--> HuggingFace AutoModelForMaskedLM GPU forward pass OK")

# ------------------------------------------------------------------ summary --
hr("ALL CHECKS PASSED")
import importlib.metadata as md  # noqa: E402

for pkg in ["torch", "fair-esm", "transformers", "numpy", "scipy", "pandas",
            "matplotlib", "seaborn", "scikit-learn", "biotite", "biopython",
            "tqdm", "h5py", "pyarrow"]:
    try:
        print(f"  {pkg:<14} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:<14} !! NOT INSTALLED !!")
print(f"\nweights cached under: {CACHE}")

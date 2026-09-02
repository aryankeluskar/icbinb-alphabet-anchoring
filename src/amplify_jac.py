"""Categorical-Jacobian wrapper for AMPLIFY, reusing the ESM-2 pipeline exactly.

Why this file exists rather than a `transformers` one-liner: `AutoTokenizer`/`AutoModel` with
`trust_remote_code=True` **hangs indefinitely** on this box against the local AMPLIFY
checkpoints (observed twice, >7 min with no output and no traceback, under
`HF_HUB_OFFLINE=1`). The checkpoint ships everything needed to bypass the Auto* machinery:
`amplify.py` with the model class, `config.json`, `model.safetensors`, and a 27-entry
`tokenizer.json`. We load those directly.

The class subclasses `jacobian.ESM2Jacobian` and overrides only tokenization and the logit
forward, so `jacobian()` and `contact_map()` — the mean-centring, symmetrisation, per-block
Frobenius and APC — are *literally the same code* that produced every ESM-2 number. That is
the point: a cross-architecture comparison is only meaningful if the readout is identical.
"""
import importlib.util
import json
import os

import torch

import jacobian as jac_mod

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _install_xformers_shim():
    """AMPLIFY's amplify.py hard-imports xformers, which is not installed and has no wheel
    for torch 2.11+cu128. Only two symbols are used, and both have exact torch equivalents.

    Correctness note: `memory_efficient_attention` is called ONLY on the CUDA branch of
    `_att_block`; the CPU branch already calls `scaled_dot_product_attention` directly. The
    shim below is written to be *numerically the same computation as that CPU branch*, so the
    two code paths agree by construction rather than by approximation.
    """
    import sys
    import types
    if "xformers.ops" in sys.modules:
        return
    import torch.nn.functional as F

    class SwiGLU(torch.nn.Module):
        """xformers.ops.SwiGLU with _pack_weights=True (which is its default).

        The checkpoint's parameter names confirm the packed layout:
        `ffn.w12.weight` is [2*hidden, in] and `ffn.w3.weight` is [out, hidden].
        """

        def __init__(self, in_features, hidden_features, out_features=None, bias=True):
            super().__init__()
            out_features = out_features or in_features
            self.w12 = torch.nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, **kwargs):
        # xformers layout is (B, M, H, K); torch SDPA wants (B, H, M, K).
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=attn_bias, dropout_p=p)
        return out.transpose(1, 2)

    ops = types.ModuleType("xformers.ops")
    ops.SwiGLU = SwiGLU
    ops.memory_efficient_attention = memory_efficient_attention
    pkg = types.ModuleType("xformers")
    pkg.ops = ops
    sys.modules["xformers"] = pkg
    sys.modules["xformers.ops"] = ops


def _load_amplify_module(ckpt_dir):
    """Import the checkpoint's own amplify.py without going through transformers' Auto*.

    Two obstacles: the hard xformers import (shimmed above), and `from .rmsnorm import ...`
    relative imports, which need amplify.py to be loaded *as a submodule of a package*. We
    synthesise that package with `__path__` pointing at the checkpoint directory, so the normal
    import machinery resolves `.rmsnorm` and `.rotary` from the checkpoint's own files.
    """
    import sys
    import types
    _install_xformers_shim()

    pkg_name = "amplify_ckpt_%s" % abs(hash(ckpt_dir))
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [ckpt_dir]
        sys.modules[pkg_name] = pkg

    mod_name = pkg_name + ".amplify"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(ckpt_dir, "amplify.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_vocab(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tokenizer.json")) as fh:
        return json.load(fh)["model"]["vocab"]


class AMPLIFYJacobian(jac_mod.ESM2Jacobian):
    """Same Jacobian and contact-map code as ESM-2; different tokenizer and forward."""

    def __init__(self, ckpt_dir, device="cuda", batch_size=32, mask_bos_eos=True):
        amp = _load_amplify_module(ckpt_dir)
        with open(os.path.join(ckpt_dir, "config.json")) as fh:
            cfg_dict = json.load(fh)
        cfg = amp.AMPLIFYConfig(**{k: v for k, v in cfg_dict.items()
                                   if k not in ("architectures", "auto_map", "model_type",
                                                "transformers_version", "torch_dtype")})
        model = amp.AMPLIFY(cfg)

        from safetensors.torch import load_file
        state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("  [amplify] missing=%d unexpected=%d  %s %s"
                  % (len(missing), len(unexpected), list(missing)[:3], list(unexpected)[:3]))

        self.vocab = _load_vocab(ckpt_dir)
        self.cls_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        self.unk_idx = self.vocab["<unk>"]

        # NOTE: deliberately NOT calling super().__init__ -- it expects a fair-esm alphabet
        # object. The attribute contract that jacobian()/contact_map() rely on is set here.
        self.model = model.eval().to(device)
        self.device = device
        self.batch_size = batch_size
        self.mask_bos_eos = mask_bos_eos
        self.mask_idx = self.vocab["<mask>"]
        self.aa_idx = [self.vocab[a] for a in AA20]
        self.n_params = sum(p.numel() for p in model.parameters())

    def _tokenize(self, seq):
        toks = [self.cls_idx]
        toks += [self.vocab.get(c, self.unk_idx) for c in seq]
        toks += [self.eos_idx]
        return torch.tensor(toks, dtype=torch.long)

    @torch.no_grad()
    def _logits(self, batch_toks):
        batch_toks = batch_toks.to(self.device)
        if self.mask_bos_eos:
            batch_toks = batch_toks.clone()
            batch_toks[:, 0] = self.mask_idx
            batch_toks[:, -1] = self.mask_idx
        out = self.model(batch_toks)
        logits = out.logits if hasattr(out, "logits") else out[0]   # (B, L+2, 27)
        logits = logits[:, 1:-1, :]                                 # drop BOS/EOS positions
        return logits[:, :, self.aa_idx]                            # (B, L, 20)


def load_amplify(year=2024, root=None, device="cuda", batch_size=32):
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "amplify")
    ckpt = os.path.join(root, "AMPLIFY_120M_%d" % year)
    if not os.path.isdir(ckpt):
        raise IOError("no AMPLIFY checkpoint at %s" % ckpt)
    return AMPLIFYJacobian(ckpt, device=device, batch_size=batch_size)

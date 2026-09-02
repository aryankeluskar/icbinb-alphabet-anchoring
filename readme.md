# Code and Dataset (released anonymously)

Code and committed results for **"Protein Language Models Fail To Capture Contact Predictions Under Invariant Relabeling"** (submitted to *I Can't Believe It's Not Better: Failure Modes of AI in Biology*, NeurIPS 2026).

## Environment

Python 3.11+, a recent PyTorch with CUDA, and:

```bash
pip install torch fair-esm transformers numpy scipy pandas matplotlib seaborn scikit-learn biotite tqdm h5py pyarrow accelerate safetensors
```

**AMPLIFY yearly checkpoints** (`chandar-lab/AMPLIFY_120M`, revision `AMPLIFY_120M_<YEAR>`, 2011–2024) must be loaded in **fp32**. Those branches ship a pre-fix `rmsnorm.py`; `trust_remote_code=True` loads that revision's code, which diverges silently under autocast / fp16 / bf16.

## Quick start

CPU-only self-test of the audit against four committed C8 cells (ESM-2 8M, ubiquitin, *m* = 20):

```bash
python src/alphabet_audit.py --self-test
```

Audit any fair-esm model on one sequence (no structure, no MSA):

```bash
python src/alphabet_audit.py \
  --esm esm2_t33_650M_UR50D \
  --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

Reproduce the Potts invariance gate (no GPU):

```bash
python experiments/H2-alphabet-permutation/code/test_mrf_invariance.py
python experiments/H2-alphabet-permutation/code/test_theory_conditions.py
```

Regenerate the paper figure from committed CSVs (no model download):

```bash
python experiments/H2-alphabet-permutation/code/make_figure_v2.py
python experiments/H2-alphabet-permutation/code/analyze_h2.py
```


## Data licenses

Third-party assets keep their upstream licenses. AMPLIFY checkpoints are MIT, Pfam alignments are CC0, SCOPe / ASTRAL is free for academic use and ProteinGym is MIT. The categorical Jacobian follows Zhang, Wayment-Steele, Brixi & Ovchinnikov, *PNAS* 2024; the inverse-covariance Potts follows Dauparas et al. AMPLIFY yearly weights are from Spinner et al. (arXiv:2507.22210), published as git branches of `chandar-lab/AMPLIFY_120M`.

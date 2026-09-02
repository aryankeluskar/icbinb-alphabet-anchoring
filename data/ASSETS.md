# ASSETS.md — data & model reconnaissance

Compiled 2026-08-05. Working root: `/scratch/author/icbinb`.
Everything marked DOWNLOADED was verified on disk (byte counts / checksums / row counts), not just linked.

---

## BLOCKERS

Read this section first — these change the experimental plan.

### B1. Historical UniRef ships as one monolithic XML tarball per release. No FASTA, no per-subset access. (HARD)
This is the single biggest cost driver in the plan.

- The path everyone assumes — `<release>/uniref/uniref100/uniref100.fasta.gz` — **returns HTTP 404 for every archived release** (verified for 2016_01, 2020_01, 2024_01). That layout exists *only* under `current_release/`.
- Each archived release ships exactly one file: `<release>/uniref/uniref<YYYY_MM>.tar.gz`, containing `uniref100.tar` → `uniref100.xml.gz`, then `uniref90.tar`, then `uniref50.tar`.
- Consequences:
  1. **XML only.** You must convert `uniref100.xml.gz` → FASTA yourself.
  2. **You cannot fetch UniRef50 alone for a historical year.** All three subsets sit inside one gzip stream, so there is no HTTP range access. Getting only 2024's UniRef50 still costs the full 233.8 GB transfer — and `uniref100.tar` comes *first* in the archive, so UniRef50 is at the far end. You can stream (`curl … | tar -x`) to avoid storing it, but the network cost is unavoidable.
- Cost: 2012 + 2016 + 2020 + 2024 alone ≈ **419 GB of transfer**. A full 2011–2024 sweep is ≈ 1.4 TB.

**PARTIAL MITIGATION — already on disk, may remove the need entirely.** Two sources give per-year corpus statistics for free:
1. `repos/data-saturation-and-scaling/Results/uniref/uniref_years.csv` — UniProt / UniRef100 / UniRef90 / UniRef50 counts for **every year 2011–2025** (§2.1). Cross-checks against UniProt's own `relnotes`.
2. UniProt `relnotes` per release (§3), free to fetch.

So the terabyte is only needed if you require **per-protein** homolog-entry dates rather than **corpus-level** size-by-year. Decide which the experiment actually needs *before* committing bandwidth — and if it is per-protein, first check whether UniParc/UniProt accession first-seen dates (queryable via the UniProt REST API, no bulk download) suffice.

### B2. The GREMLIN web server is dead. The MSA corpus behind Zhang & Ovchinnikov is not redistributed. (MEDIUM)
- `gremlin.bakerlab.org` → **HTTP 404** (both http and https, redirects to `/404.html`). `gremlin2.bakerlab.org` → connection failure (HTTP 000). `files.ipd.uw.edu/pub/` → HTTP 403.
- `github.com/sokrypton/GREMLIN_CPP` is alive but its README contains **no dataset download URLs**.
- What survives: the paper's own repo ships the *sequences* — `data/full_seq_dict.json` (2,170 PDB chains) and `data/selected_protein.json` (1,431 selected). The **alignments themselves are gone**.
- **Mitigation (already executed):** ProteinGym ships 217 ready-made deep MSAs (`DMS_msa_files`, downloaded) and Pfam full alignments are live via InterPro (downloaded). Use those as the coevolution corpus. If you specifically need the GREMLIN chains' MSAs, you must regenerate them (jackhmmer/hhblits against a UniRef release) — which re-invokes blocker B1.

### B3. `pLMs-interpretability` has NO LICENSE file. (LEGAL — low effort, real risk)
No LICENSE, no requirements.txt, no setup.py. Absent a license, the code is technically all-rights-reserved. **Cite and reimplement the method (it is ~35 lines); do not vendor the file into a released artifact** without asking the authors.

### B4. AMPLIFY yearly sweep is 120M-only, at 250K steps. (SCOPE LIMIT — not a blocker, but it constrains claims)
The 2011–2024 checkpoints exist only for **AMPLIFY_120M**. `AMPLIFY_350M` has no yearly branches. The yearly models were trained to **250K steps**, whereas the headline `main` 120M/350M are trained to **1M steps** — so **yearly checkpoints are not directly comparable to the published 120M/350M numbers**. Any scaling claim must compare yearly-vs-yearly, or against `AMPLIFY_120M_base`.

### B5. AMPLIFY yearly branches carry the pre-fix RMSNorm — fp32 only. (SILENT-CORRUPTION TRAP)
`trust_remote_code=True` loads modeling code **from the same revision**, and the yearly branches ship the old `rmsnorm.py`. The mixed-precision fix landed on `main` on 2025-05-28 and was never backported.
- yearly: `x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + eps) * weight`
- main: `self._norm(x.float()).type_as(x)` then `* weight`

Equivalent in fp32 (the config default), **divergent under autocast/fp16/bf16**. Run the yearly sweep in fp32, or backport the fix uniformly. This will not raise an error — it will just quietly give wrong numbers.

### Resolved non-blockers (previously assumed to be problems)
- **AMPLIFY yearly checkpoints ARE public.** See §2 — they were hard to find because they are git *branches*, not repos.
- **ESM-3 is no longer gated.** See §8 — EvolutionaryScale is now Chan Zuckerberg Biohub and the model is plain MIT, ungated.
- **ProteinGym substitutions is not "mostly singles."** It is 72% multi-mutant. See §1.

---

## Master asset table

| # | Asset | Status | Local path | Size | URL / repo id | Note |
|---|---|---|---|---|---|---|
| 1a | ProteinGym DMS substitutions (217 assays) | **DOWNLOADED** | `data/proteingym/DMS_ProteinGym_substitutions/` | 1016 MB | `marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip` | 217 CSVs, 2,465,767 mutants |
| 1b | ProteinGym reference file | **DOWNLOADED** | `repos/ProteinGym/reference_files/DMS_substitutions.csv` | 204 KB | ships in the git repo | 46 cols incl. Neff/L, taxon, multi-mutant counts |
| 1c | Zero-shot baseline scores (subs) | **DOWNLOADED** | `data/proteingym/zero_shot_substitutions_scores/` | 4.9 GB | `…/zero_shot_substitutions_scores.zip` | **95 models precomputed** — huge compute saving |
| 1d | CV folds — multiples | **DOWNLOADED** | `data/proteingym/cv_folds_multiples_substitutions/` | 631 MB | `…/cv_folds_multiples_substitutions.zip` | 69 assays, has `mutation_depth` (=k) column |
| 1e | CV folds — singles | **DOWNLOADED** | `data/proteingym/cv_folds_singles_substitutions/` | 407 MB | `…/cv_folds_singles_substitutions.zip` | 217 assays |
| 1f | ProteinGym DMS indels | **DOWNLOADED** | `data/proteingym/DMS_ProteinGym_indels/` | 194 MB | `…/DMS_ProteinGym_indels.zip` | 66 assays |
| 1g | ProteinGym AF2 structures | **DOWNLOADED** | `data/proteingym/ProteinGym_AF2_structures/` | 83 MB | `…/ProteinGym_AF2_structures.zip` | 198 PDBs |
| 1h | ProteinGym MSAs (217 assays) | **DOWNLOADED** | `data/proteingym/DMS_msa_files/` | 5.2 GB | `…/DMS_msa_files.zip` | `.a2m`, filenames = `MSA_filename` col |
| 1i | ProteinGym MSA weights | **DOWNLOADED** | `data/proteingym/DMS_msa_weights/` | 45 MB | `…/DMS_msa_weights.zip` | redundancy weights |
| 1j | ProteinGym repo | **DOWNLOADED** | `repos/ProteinGym/` | 182 MB | `github.com/OATML-Markslab/ProteinGym` | MIT |
| 2e | **Spinner et al. paper repo + results** | **DOWNLOADED** | `repos/data-saturation-and-scaling/` | 24 MB | `github.com/Align-to-Innovate/data-saturation-and-scaling` | **See §2.1 — big shortcut** |
| 2a | **AMPLIFY yearly 2011–2024 (120M)** | **DOWNLOADED** | `data/amplify/AMPLIFY_120M_<year>/` | 14 × 451 MiB | `chandar-lab/AMPLIFY_120M` @ `revision=AMPLIFY_120M_<year>` | **PUBLIC — as git branches.** MIT, ungated |
| 2b | AMPLIFY_120M base + main | **DOWNLOADED** | `data/amplify/AMPLIFY_120M_{base,main}/` | 2 × 451 MiB | same repo, `AMPLIFY_120M_base` / `main` | |
| 2c | AMPLIFY_350M | **DOWNLOADED** | `data/amplify/AMPLIFY_350M_main/` | 1.32 GiB | `chandar-lab/AMPLIFY_350M` | **no yearly branches** |
| 2d | AMPLIFY training corpus UR100P | AVAILABLE | — | 173.3 GiB | `chandar-lab/UR100P` (dataset) | CC-BY-4.0, ungated. Not pulled (large) |
| 3 | Historical UniRef releases | AVAILABLE (costly) | — | 6.1 GB (2010) → 266.7 GB (2025) | `ftp.uniprot.org/pub/databases/uniprot/previous_releases/` | **See BLOCKER B1.** CC-BY-4.0 |
| 4a | ASTRAL40 SCOPe 2.08 seqs | **DOWNLOADED** | `data/scope/astral-…-40-2.08.fa` | 4.3 MB | `scop.berkeley.edu/downloads/scopeseq-2.08/` | free for academic use |
| 4b | ASTRAL95 SCOPe 2.08 seqs | **DOWNLOADED** | `data/scope/astral-…-95-2.08.fa` | 9.8 MB | same | |
| 4c | SCOPe hierarchy (cla/des/hie) | **DOWNLOADED** | `data/scope/dir.{cla,des,hie}.scope.2.08-stable.txt` | 59 MB total | `scop.berkeley.edu/downloads/parse/` | 2.08 is the newest; 2.09 → 404 |
| 4d | SCOPe pdbstyle-40 structures | AVAILABLE | — | 1.04 GB | `scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz` | needed by PLM-GUARD; not pulled |
| 5a | GREMLIN PDB_EXP MSAs | **NOT PUBLIC** | — | — | `gremlin.bakerlab.org` → **404** | **BLOCKER B2** |
| 5b | GREMLIN chain seqs (surviving) | **DOWNLOADED** | `repos/pLMs-interpretability/data/full_seq_dict.json` | 648 KB | in the paper repo | 2,170 chains + 1,431 selected |
| 5c | Pfam full alignments PF00072/76/13 | **DOWNLOADED** | `data/pfam/PF000*_full.aln.gz` | 183 MB total | InterPro API (see §5) | gzip-verified |
| 5d | Pfam seed alignments (same 3) | **DOWNLOADED** | `data/pfam/PF000*_seed.aln.gz` | 62 KB total | InterPro API | |
| 5e | Pfam-A.seed (ALL families) | **DOWNLOADED** | `data/pfam/Pfam-A.seed.gz` | 194 MB | `ftp.ebi.ac.uk/pub/databases/Pfam/current_release/` | CC0. Pfam 38.2, 30,134 families |
| 5f | Pfam-A.full (ALL families) | AVAILABLE | — | **24.0 GB** | same FTP | >20 GB — reported only, per instructions |
| 6 | Zhang & Ovchinnikov code | **DOWNLOADED** | `repos/pLMs-interpretability/` | 36 MB | `github.com/zzhangzzhang/pLMs-interpretability` | **no license** — BLOCKER B3 |
| 7a | PLM-GUARD code | **DOWNLOADED** | `repos/PLMGuard/` | 1.2 MB | `github.com/batmen-lab/PLMGuard` | Apache-2.0 |
| 7b | PLM-GUARD Zenodo data | **IN FLIGHT** | `data/plmguard_zenodo/` | 8.82 GB | `doi.org/10.5281/zenodo.19795993` | CC-BY-4.0. Zenodo is slow; resumable — see §7 |
| 7c | PLM-GUARD precomputed results | AVAILABLE | — | **49.08 GB** | `huggingface.co/datasets/Hanhanhanhaner/PLMGuard` | >20 GB — reported only. 79 files, ungated |
| 8a | ESM-2 650M | AVAILABLE | — | 2.61 GB | `facebook/esm2_t33_650M_UR50D` | MIT, ungated |
| 8b | ESM-2 3B | AVAILABLE | — | 11.37 GB | `facebook/esm2_t36_3B_UR50D` | MIT. **No safetensors — .bin only** |
| 8c | ESM-1v ensemble (×5) | AVAILABLE | — | 13.05 GB total | `facebook/esm1v_t33_650M_UR90S_{1..5}` | **no declared license** on HF |
| 8d | ESM-3 open | AVAILABLE — **UNGATED** | — | 2.80 GB (repo ≈5.5 GB) | `biohub/esm3-sm-open-v1` | **MIT now.** Gating removed |
| 8e | ESM-C | AVAILABLE | — | 1.33 / 2.30 / 25.41 GB | `biohub/ESMC-{300M,600M,6B}` | license tags `mit` + `other` |

Downloaded: **27 GB** under `data/`, **385 MB** under `repos/` (the PLM-GUARD tarball is still transferring; everything else is complete).

Original `.zip` archives are retained alongside their extracted directories — delete them (`rm data/proteingym/*.zip`) to reclaim ~1.9 GB once you are satisfied with the extractions.

---

## 1. ProteinGym — and the multi-mutant question

**Version used: v1.3** (latest). Full manifest of downloadable files is in `repos/ProteinGym/README.md`.

Working download command (note the SSL fix — see §9):
```bash
export CURL_CA_BUNDLE=/scratch/author/icbinb/.cache/ca-bundle-plus.crt
VERSION="v1.3"; FILENAME="DMS_ProteinGym_substitutions.zip"
curl -O https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${FILENAME}
unzip ${FILENAME}
```

### 1.1 The reference file
`repos/ProteinGym/reference_files/DMS_substitutions.csv` — **ships in the git repo, no download needed**. 217 rows × 46 columns, including everything asked for: `DMS_id`, `UniProt_ID`, `taxon`, `source_organism`, `seq_len`, `DMS_total_number_mutants`, `MSA_N_eff`, `MSA_Neff_L`, `MSA_Neff_L_category`, `MSA_num_seqs`, `selection_type`, `coarse_selection_type`, plus `pdb_file` and `weight_file_name`. Column semantics are documented in `reference_files_description.md`.

Critically it also has `includes_multiple_mutants` (bool), `DMS_number_single_mutants`, and `DMS_number_multiple_mutants`.

### 1.2 MULTI-MUTANT DATA — corrects the brief's assumption

> The brief expected "the substitutions set is mostly singles." **That is not the case.**

I counted `:` separators in the `mutant` column across all 217 downloaded CSVs (`data/k_distribution.py`, raw output `data/k_distribution.json`):

| | mutants | share |
|---|---:|---:|
| k=1 (singles) | 696,311 | 28.24% |
| **k≥2 (multi)** | **1,769,456** | **71.76%** |
| TOTAL | 2,465,767 | 100% |

**The substitutions benchmark is ~72% multi-mutant by row count.** 69 of 217 assays contain k≥2. Max observed k = **44**.

Multi-mutant data lives **inside the substitutions set** — not in the indels set (indels are insertions/deletions, a different axis) and not exclusively in the supervised splits.

Global k distribution (head):

| k | mutants | share |
|---:|---:|---:|
| 1 | 696,311 | 28.24% |
| 2 | 826,245 | 33.51% |
| 3 | 84,134 | 3.41% |
| 4 | 187,850 | 7.62% |
| 5 | 92,213 | 3.74% |
| 6 | 130,395 | 5.29% |
| 7 | 149,318 | 6.06% |
| 8 | 134,593 | 5.46% |
| 9 | 90,108 | 3.65% |
| 10 | 45,816 | 1.86% |
| 11–44 | 27,073 | 1.10% |

**Full per-assay k-breakdown for all 69 assays is in `data/MULTI_MUTANT_INVENTORY.md`.** Top of that table:

| DMS_id | total | k=1 | k=2 | k=3 | k=4 | k≥5 | max k | Neff/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPG1_STRSG_Olson_2014 | 536,962 | 1,045 | 535,917 | 0 | 0 | 0 | 2 | 0.00 |
| HIS7_YEAST_Pokusaeva_2019 | 496,137 | 168 | 1,475 | 7,627 | 25,927 | 460,940 | 28 | 23.97 |
| PHOT_CHLRE_Chen_2023 | 167,529 | 2,122 | 176 | 978 | 3,565 | 160,688 | 15 | 5215.65 |
| SPG1_STRSG_Wu_2016 | 149,360 | 76 | 2,091 | 26,019 | 121,174 | 0 | 4 | 1.34 |
| GRB2_HUMAN_Faure_2021 | 63,366 | 1,034 | 62,332 | 0 | 0 | 0 | 2 | 6.78 |
| GFP_AEQVI_Sarkisyan_2016 | 51,714 | 1,084 | 12,777 | 12,336 | 9,387 | 16,130 | 15 | 0.06 |
| CAPSD_AAV2S_Sinai_2021 | 42,328 | 532 | 10,833 | 6,906 | 6,646 | 17,411 | 28 | 0.26 |
| PABP_YEAST_Melamed_2013 | 37,708 | 1,187 | 36,521 | 0 | 0 | 0 | 2 | 1.42 |

**Important structural caveat for experimental design:** the k≥2 mass is extremely concentrated and mostly shallow.
- Of the 69 multi-mutant assays, **58 are k=2-only** (max k = 2) — and **50 of those are `Tsuboyama_2023_*`** stability assays with only a few hundred doubles each. The 8 non-Tsuboyama k=2-only assays are `SPG1_STRSG_Olson_2014`, `GRB2_HUMAN_Faure_2021`, `PABP_YEAST_Melamed_2013`, `RASK_HUMAN_Weng_2022_abundance`, `RASK_HUMAN_Weng_2022_binding-DARPin_K55`, `A4_HUMAN_Seuma_2022`, `YAP1_HUMAN_Araya_2012`, `DLG4_HUMAN_Faure_2021`.
- Only **11 assays** have any k≥3 at all.
- **Only 9 assays have any k≥5 data**: `HIS7_YEAST_Pokusaeva_2019`, `PHOT_CHLRE_Chen_2023`, `GFP_AEQVI_Sarkisyan_2016`, `CAPSD_AAV2S_Sinai_2021`, `Q8WTC7_9CNID_Somermeyer_2022`, `D7PM05_CLYGR_Somermeyer_2022`, `Q6WV12_9MAXI_Somermeyer_2022`, `F7YBW8_MESOW_Ding_2023`, `GCN4_YEAST_Staller_2018`.
- Two assays (`SPG1_STRSG_Olson_2014`, `HIS7_YEAST_Pokusaeva_2019`) alone account for **58.3%** of all multi-mutant rows. Any unweighted average over "multi-mutant data" is really an average over those two proteins — weight or subsample deliberately.

### 1.3 Supervised splits for multi-mutants
`cv_folds_multiples_substitutions/` contains exactly **69 CSVs — one per multi-mutant assay**, columns:
`mutant, mutated_sequence, DMS_score, DMS_score_bin, mutation_depth, fold_rand_multiples`

`mutation_depth` **is k**, precomputed for you. `fold_rand_multiples` is ProteinGym's official 5-fold split. `cv_folds_singles_substitutions/` has all 217 assays with `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`.

### 1.4 Precomputed zero-shot baseline scores — YES, and they cover multi-mutants
`zero_shot_substitutions_scores/` = **217 CSVs (one per assay, exactly matching the DMS assay list — verified by `comm`)**, each with `mutant, mutated_sequence, DMS_score, DMS_score_bin` + **95 model score columns**.

Includes everything asked for and much more: `EVE_single`/`EVE_ensemble`, `ESM1v_single`/`ESM1v_ensemble`, `ESM1b`, `ESM2_{8M,35M,150M,650M,3B,15B}`, `ESM3`, `ESMC-{300M,600M}`, `EVmutation`, `DeepSequence_*`, `GEMME`, `MSA_Transformer_*`, `Tranception_*`, `TranceptEVE_*`, `Progen2_*`, `Progen3_*`, `RITA_*`, `CARP_*`, `SaProt_*`, `ProSST-*`, `ProtSSN_*`, `VESPA*`, `ESM-IF1`, `ProteinMPNN`, `MIF`/`MIFST`, `S2F`/`S3F`, `SiteRM`, `ESCOTT`, `VenusREM`, `RSALOR`, `PoET`, `xTrimoPGLM-*`, `Site_Independent`, `Unirep*`, `Wavenet`, `MULAN_small`.

**Verified they cover k≥2:** row counts match the DMS files exactly for every multi-mutant assay checked (e.g. `SPG1_STRSG_Olson_2014` 536,962 = 536,962; `HIS7_YEAST_Pokusaeva_2019` 496,137 = 496,137), and I confirmed populated scores on a k=7 mutant across ESM2_650M, ESM1v_ensemble, EVE_ensemble, GEMME, ESM3, ESMC-600M, TranceptEVE_L.

This saves an enormous amount of compute — **do not re-score these baselines.**

Also precomputed and available (not pulled): `DMS_supervised_substitutions_scores.zip` (3.3 GB).

---

## 2. AMPLIFY yearly snapshots — **PUBLIC** (headline finding)

> The brief flagged this as the make-or-break asset. **They are public.** They are simply not discoverable by searching HuggingFace, because they are **git branches inside one repo**, not separate repos.

Paper: *Scaling and Data Saturation in Protein Language Models*, arXiv **2507.22210**. The availability statement is in the PDF body (not on the abstract page):

> "Pretrained AMPLIFY model checkpoints for all years (2011–2024) can be found at: `https://huggingface.co/chandar-lab/AMPLIFY_120M/tree/AMPLIFY_120M_<YEAR>`"

Paper code: `github.com/Align-to-Innovate/data-saturation-and-scaling` (code/figures only, no weights).

**Verified independently twice** (subagent + my own download): `…/AMPLIFY_120M/refs` returns 16 branches — `main`, `AMPLIFY_120M_base`, and `AMPLIFY_120M_2011` … `_2024`, **all 14 years present, none missing**. Every branch's `model.safetensors` is exactly **473,126,988 bytes** with a **unique sha256**, confirming genuinely distinct weights rather than aliases. My locally computed checksums match the independently computed ones (2011 `ef09aefd3b7c849d`, 2024 `4e809585375333c3`, base `5cdd05fcfa647ed4`, …).

**MIT licensed. Not gated. No token, no access request.**

Working load:
```python
from transformers import AutoModel
m = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M",
                              revision="AMPLIFY_120M_2017",   # 2011..2024
                              trust_remote_code=True)
```
Downloaded locally to `data/amplify/AMPLIFY_120M_<year>/` (7.4 GB total, incl. base, main, and 350M).

**See BLOCKERS B4 (120M-only, 250K vs 1M steps) and B5 (pre-fix RMSNorm → fp32 only). Both matter.**

Training corpus `chandar-lab/UR100P` is public (173.3 GiB, CC-BY-4.0) but the **per-year UniRef100 snapshot corpora are not published** — only the resulting weights. So you can use the yearly *models* but cannot directly inspect the yearly *data* without reconstructing it (blocker B1).

Adjacent, possibly useful: `chandar-lab/CoPeP` + `chandar-lab/copep-checkpoints` — continual-learning protein dataset with per-year task splits from 2015.

### 2.1 The paper's own repo ships reusable results — cloned, 24 MB

`repos/data-saturation-and-scaling/` (`github.com/Align-to-Innovate/data-saturation-and-scaling`). Code + figures only, no weights, but two files are immediately valuable:

**`Results/uniref/uniref_years.csv` — per-year corpus sizes 2011–2025, free.** This is the single most useful small file found, because it delivers a large part of what blocker B1's terabyte download was for:

| year | UniProt | UniRef100 | UniRef90 | UniRef50 |
|---:|---:|---:|---:|---:|
| 2011 | 13,069,501 | 11,659,891 | 7,623,063 | 3,653,743 |
| 2012 | 19,968,488 | 15,688,962 | 9,843,844 | 4,606,913 |
| 2014 | 52,159,208 | 33,613,081 | 20,200,107 | 9,370,012 |
| 2016 | 60,268,458 | 72,946,704 | 39,362,473 | 16,038,089 |
| 2018 | 108,184,003 | 133,853,533 | 69,029,793 | 30,071,646 |
| 2020 | 178,316,438 | 216,491,817 | 107,153,647 | 39,232,797 |
| 2022 | 230,895,644 | 297,827,854 | 144,113,457 | 51,333,317 |
| 2024 | 250,322,721 | 390,790,959 | 184,520,054 | 63,849,054 |
| 2025 | 253,206,171 | 453,950,711 | 204,806,910 | 69,290,910 |

(full 2011–2025 table in the file; UniRef100/50 columns independently agree with the UniProt `relnotes` figures in §3, a useful cross-check.)
⚠️ The `uniprot` column **drops** 2015 → 2016 (89,998,523 → 60,268,458) — a TrEMBL proteome-redundancy purge, not an error. Same caution as §3: growth is not monotonic.

**`Results/unsupervised_results/PG_spearman_correlations_allyears213.csv`** — 213 assays × 14 yearly models of already-computed Spearman correlations, plus a `spearman_random` control column. Also `PG_spearman_correlations213.csv`, and semisupervised ridge results under `Results/semisupervised_results/` (one-hot, chunked, modulo splits). `Data/DMS_substitutions.csv` is their pinned copy of the ProteinGym reference file.

Their pipeline: `Analysis/get_AMPLIFY_logits_and_embeddings.py` (loops `range(2011,2025)` with `revision=f"AMPLIFY_120M_{year}"`), `ProteinGym_AMPLIFY_unsupervised_performance.ipynb`, `uniref_parser.ipynb`, and three `semisupervised_*.py` split variants. `requirements.txt` present. Note they use **213** assays vs ProteinGym's 217.

⚠️ **NOVELTY WARNING (flagged independently by another agent, consistent with this repo):** Spinner et al. already stratify the year-trend by MSA depth (Neff/L) in their Fig. S1 and report the interaction — deeper-MSA proteins improve with later training years, shallow-MSA ones flatten or decline. Check `Results/Figures/` before claiming that observation as novel.

⚠️ **Confound baked into the yearly sweep:** training steps are held constant at 250K while the corpus grows ~33× (11.7M → 391M UniRef100 clusters). The 2011 model therefore sees far more epochs over its data than the 2024 model. This is not fixable post-hoc and must be stated explicitly in any writeup.

---

## 3. Historical UniRef — report only

**See BLOCKER B1 for the structural problem.** Coverage itself is good:

- Use `https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/` (richest). EBI's mirror is yearly-only before 2021; **`ftp.expasy.org` has no archive at all**.
- Releases present: `release1.0`–`release15.0` (2005–2009), then `release-2010_01`, `2011_01`, `2012_01`, `2013_01`, `2014_01` (**January only**), then near-monthly from `2015_01` through `2021_04`, then ~5–6/year through `2026_01`. Current release is `2026_02` (only under `current_release/`).
- **`uniref/` is present in every release directory checked** (2009 through 2026_01) — the worry that older dirs are knowledgebase-only is **false**; `release-2010_01/` contains only `knowledgebase/` and `uniref/`.
- Net: continuous UniRef coverage **2009 → 2026_01**, sub-yearly from 2015, annual 2010–2014.

Whole-release tarball sizes (UniRef50+90+100 combined, decimal GB from Content-Length):

| Release | Size | Release | Size |
|---|---:|---|---:|
| 2010_01 | 6.1 GB | 2019_01 | 105.6 GB |
| 2011_01 | 7.4 GB | 2020_01 | 129.8 GB |
| 2012_01 | 10.0 GB | 2021_01 | 157.5 GB |
| 2013_01 | 13.4 GB | 2022_01 | 179.2 GB |
| 2014_01 | 22.4 GB | 2023_01 | 207.0 GB |
| 2015_01 | 35.3 GB | 2024_01 | 233.8 GB |
| 2016_01 | 45.4 GB | 2025_01 | 266.7 GB |
| 2017_01 | 58.2 GB | 2026_01 | ~268 GB (listing only) |
| 2018_01 | 83.8 GB | | |

`current_release` (2026_02) is the **only** place FASTA exists: `uniref100.fasta.gz` 63.1 GB, `uniref90.fasta.gz` 32.1 GB, `uniref50.fasta.gz` 8.8 GB.

Cluster counts per release — **these come free in `relnotes` and may let you skip the terabyte download entirely**:

| Release | UniRef100 | UniRef50 |
|---|---:|---:|
| 2012_01 | 15,688,962 | 4,606,913 |
| 2014_01 | 33,613,081 | 9,370,012 |
| 2016_01 | 72,946,704 | 16,038,089 |
| 2018_01 | 133,853,533 | 30,071,646 |
| 2020_01 | 216,491,817 | 39,232,797 |
| 2022_01 | 297,827,854 | 51,333,317 |
| 2024_01 | 390,790,959 | 63,849,054 |
| 2025_01 | 453,950,711 | 69,290,910 |
| 2026_01 | 475,217,233 | 60,315,044 |
| 2026_02 | 220,919,788 | 38,794,121 |

⚠️ **Two anomalies that will bite a naive "entry year" analysis:** UniRef50 *dropped* 69.3M → 60.3M between 2025_01 and 2026_01, and UniRef100 collapsed 475M → 221M between 2026_01 and 2026_02 (major redundancy/proteome pruning). Across 2025→2026 sequences **leave** the corpus as well as enter it. Monotonic-accumulation assumptions break here.

**License:** CC BY 4.0 (confirmed verbatim in `previous_releases/LICENSE`). ⚠️ Caveat: archived per-release metalinks for older releases declare **CC BY-ND 3.0** (verified in `release-2016_01/uniref/RELEASE.metalink`). If you redistribute derived historical UniRef, the archived files carry the more restrictive metadata.

Also archived per release: `RELEASE.metalink` (contains MD5s — use these to verify multi-hundred-GB transfers), `relnotes.txt`, `changes.html`.

---

## 4. SCOPe / ASTRAL40 — DOWNLOADED

**SCOPe 2.08 is the current release** (2.09 → HTTP 404). All in `data/scope/` (71 MB):

| File | Bytes |
|---|---:|
| `astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa` | 4,299,207 |
| `astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa` | 9,838,324 |
| `dir.cla.scope.2.08-stable.txt` | 34,697,841 |
| `dir.des.scope.2.08-stable.txt` | 15,971,286 |
| `dir.hie.scope.2.08-stable.txt` | 8,735,563 |

```bash
cd data/scope
curl -O https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
curl -O https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt
curl -O https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.08-stable.txt
curl -O https://scop.berkeley.edu/downloads/parse/dir.hie.scope.2.08-stable.txt
```
`cla` = classification (domain → sunid + full `cl/cf/sf/fa/dm/sp/px` lineage), `des` = node descriptions, `hie` = parent/child tree. Together these give the full SCOPe hierarchy PLM-GUARD keys on.

Not pulled (1.04 GB, needed only to run PLM-GUARD's structure experiments):
`https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz`

License: SCOPe is free for academic/non-commercial use; check `scop.berkeley.edu` terms before redistribution.

---

## 5. Coevolution / MSA corpus

### 5.1 GREMLIN PDB_EXP — NOT OBTAINABLE (BLOCKER B2)
Verified dead: `gremlin.bakerlab.org` → HTTP 404, `gremlin2.bakerlab.org` → HTTP 000, `files.ipd.uw.edu/pub/` → HTTP 403. `sokrypton/GREMLIN_CPP` is alive (HTTP 200) but README lists no data URLs.

Surviving artifact — the sequence set, in the paper's own repo:
- `repos/pLMs-interpretability/data/full_seq_dict.json` — 2,170 PDB chains (`{"2GWGA": "MIIDIHGHY…", …}`)
- `repos/pLMs-interpretability/data/selected_protein.json` — 1,431 chains after the paper's filtering
- `repos/pLMs-interpretability/data/ss_dict.json` — secondary-structure assignments

Alignments must be regenerated if needed.

### 5.2 Pfam — alive and well, via InterPro
Pfam is fully retrievable. Two routes, both verified:

**Per-family (InterPro API)** — gzipped Stockholm, `annotation=alignment:{full,seed,uniprot}`:
```bash
curl -o PF00072_full.aln.gz \
  "https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00072/?annotation=alignment:full&download"
```
Downloaded to `data/pfam/` (all gzip-integrity verified):

| Family | full | seed |
|---|---:|---:|
| PF00072 (Response_reg) | 89,662,705 B | 6,046 B |
| PF00076 (RRM_1) | 78,209,297 B | 5,754 B |
| PF00013 (KH_1) | 15,015,273 B | 49,787 B |

**Bulk (EBI FTP, the old Pfam FTP layout, still maintained):**
`https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/`

| File | Size | Status |
|---|---:|---|
| `Pfam-A.seed.gz` (all families, curated seeds) | 194 MB | **DOWNLOADED** |
| `Pfam-A.clans.tsv.gz` | 554 KB | **DOWNLOADED** |
| `Pfam.version.gz`, `relnotes.txt` | tiny | **DOWNLOADED** |
| `Pfam-A.hmm.gz` | 418 MB | available |
| `Pfam-A.full.gz` (all full alignments) | **24.0 GB** | >20 GB — not pulled |

**Pfam release 38.2** (dated 2026-01, built on UniProtKB 2025_03), 30,134 families — from the downloaded `Pfam.version.gz`. License is **CC0 1.0** ("no copyright"), confirmed verbatim in §12 of the downloaded `relnotes.txt`.

⚠️ Note the dependency for temporal work: Pfam 38.2 is built on **UniProtKB 2025_03**, so these alignments already encode 2025 sequence data. They are not a clean "historical" corpus for any pre-2025 year.

### 5.3 Recommended substitute corpus (already on disk)
ProteinGym's `DMS_msa_files` gives 217 deep MSAs already matched to the DMS assays, with per-assay `MSA_N_eff` / `MSA_Neff_L` / `MSA_num_seqs` in the reference file and redundancy weights in `DMS_msa_weights` (45 MB). For coevolution baselines tied to fitness data this is a better fit than GREMLIN anyway.

Format: **`.a2m`**, named `<UniProt>_full_<date>_b<bitscore>.a2m` (e.g. `YAP1_HUMAN_full_11-26-2021_b02.a2m`), some with a `theta0.99` infix. These filenames are exactly the `MSA_filename` column of `DMS_substitutions.csv`, so joining MSAs to assays is a direct lookup. `jac/utils.py::parse_fasta(..., a3m=True)` strips lowercase insertion columns and works on these directly.

---

## 6. Zhang & Ovchinnikov code — cloned, method is portable, script is not

`repos/pLMs-interpretability/` (36 MB). Paper: *Protein language models learn evolutionary statistics of interacting sequence motifs*, PNAS 2024, **DOI 10.1073/pnas.2406285121**.

**Contents:** `jac/` (8 items — Jacobian calculation, P@L contact metrics, pairwise-potential analysis, same-MSA comparisons), `contact_recovery/` (24 items — masking experiments), `data/` (7 JSON files, §5.1). **Mostly Jupyter notebooks; only 5 `.py` files.** The h5 outputs are deliberately not included.

**The categorical Jacobian itself** — `jac/02_get_jac_batch.py::get_categorical_jacobian`, ~20 lines:
- For each position `n` in `[0,L)`: tile the sequence 20×, set position `n` to each of the 20 amino acids, one batched forward pass, keep `logits[..., 1:L+1, 4:24]`.
- Returns `fx_h - fx`, shape `(L,20,L,20)`.
- Post-processing in `get_data`: center over all 4 axes, symmetrize `(jac + jac.transpose(2,3,0,1))/2`, then `utils.get_contacts` (Frobenius norm over the aa dims + APC).
- Cost: **L forward passes at batch 20**. Uses **unmasked** input (not masked-marginals) — worth noting, since `get_masked_logits` is a separate function in the same file.

`jac/utils.py` also provides the MRF/Gaussian comparator: `inv_cov()` (shrinkage inverse covariance on a one-hot MSA, `4.5/sqrt(N)` ridge) plus `_do_apc`/`get_contacts`, and a JAX variant.

**Runnable? Not as-is — but the fixes are trivial:**
1. **Broken data path**: reads `../esm_gap_distance/contact-recovery_github/data/full_seq_dict.json`, which does not exist in the repo. Real path is `../data/full_seq_dict.json`.
2. **Hardcoded output dirs** `/ssd/zhidian/jac/8M/` and `/ssd/zhidian/jac_contact/8M/`.
3. **Hardcoded** `os.environ["CUDA_VISIBLE_DEVICES"]='3'` (and `'1'` in `contact_recovery/*.py`).
4. **Undeclared deps**: needs `fair-esm` (calls `esm.pretrained.esm2_t6_8M_UR50D`), torch, **jax+jaxlib** (`utils.py` imports jax at module level), scipy, tqdm.
5. **No LICENSE, no requirements.txt, no setup.py** → BLOCKER B3.

Recommendation: reimplement `get_categorical_jacobian` (~20 lines) against whichever model you use — it is model-agnostic apart from the ESM alphabet offsets (`4:24`) and the BOS offset (`1:L+1`), which need remapping for AMPLIFY.

---

## 7. PLM-GUARD

**Code** — `repos/PLMGuard/` (1.2 MB), cloned from `github.com/batmen-lab/PLMGuard`. **Apache-2.0.** Layout: `src/sanity_check/` (6 experiment scripts + `overall_performance.py`), `src/PLMs_cmds/` (12 runner shell scripts + `.env`), `src/utils/` (8 helpers incl. `run_mut_rosetta.py`, `metric_utils.py`). Six experiments: bio-evolution, bio-mut-structure, perturbation-doubling, perturbation-truncation, permutation-data, permutation-model. Needs Python 3.11 + per-tool conda envs (DCTdomain, DHR, PLMSearch, TM-Vec) and a compiled `TMscore` binary.

**Zenodo** `10.5281/zenodo.19795993` (queried via API): title "PLMGuard", published **2026-04-26**, license **CC-BY-4.0**, **one file: `PLMGuard_data.tar.gz`, 8,820.3 MB (8.82 GB)**. Contains the reference databases, modified PLM library source, and Rosetta-relaxed mutant structures. Downloading to `data/plmguard_zenodo/`.
```bash
curl -L -o PLMGuard_data.tar.gz \
  "https://zenodo.org/records/19795993/files/PLMGuard_data.tar.gz?download=1"
```
**Status: still transferring at time of writing** (Zenodo throttles to roughly 30–60 MB/min, so the full 8.82 GB takes a couple of hours). The partial file is at `data/plmguard_zenodo/PLMGuard_data.tar.gz`. **Resume rather than restart:**
```bash
cd /scratch/author/icbinb/data/plmguard_zenodo
curl -L -C - -o PLMGuard_data.tar.gz \
  "https://zenodo.org/records/19795993/files/PLMGuard_data.tar.gz?download=1"
# then verify before extracting:
gzip -t PLMGuard_data.tar.gz && tar tzf PLMGuard_data.tar.gz | head
```
Expected final size: **9,248,018,432 bytes**. Nothing else depends on it — every other asset in this report is complete.

**HF dataset** `Hanhanhanhaner/PLMGuard` — ungated, **79 files, 49.08 GB total**. These are the precomputed parsed search results that let you reproduce all paper figures without running the search tools. Largest members are the shuffled-vs-original sweeps (`result_dhr_…_shuf_ori_hit30354` 6.33 GB, `result_blastp_…_shuf_ori` 2.91 GB, `result_dctdomain_…_shuf_ori` 2.81 GB). **>20 GB → reported only, not downloaded.** Fetch selectively per experiment rather than whole.

PLM-GUARD also needs SCOPe `pdbstyle-sel-gs-bib-40-2.08.tgz` (1.04 GB, §4) and ASTRAL40 (already downloaded).

---

## 8. ESM weights

> ⚠️ **The gating premise in the brief is out of date.** `EvolutionaryScale/*` now **HTTP 307-redirects to `biohub/*`** — EvolutionaryScale is Chan Zuckerberg Biohub. `https://raw.githubusercontent.com/evolutionaryscale/esm/main/LICENSE.md` now reads **"License (MIT) — Copyright 2026 Chan Zuckerberg Biohub, Inc."**

### ESM-2 (`facebook/*`, MIT, ungated — all verified)
| Repo | Weights | Size |
|---|---|---:|
| `facebook/esm2_t12_35M_UR50D` | safetensors | 136.0 MB |
| `facebook/esm2_t30_150M_UR50D` | safetensors | 595.3 MB |
| **`facebook/esm2_t33_650M_UR50D`** | safetensors | **2,609,506,392 B = 2.61 GB** |
| **`facebook/esm2_t36_3B_UR50D`** | 2× `.bin` shards | **11,367,082,474 B = 11.37 GB** |
| `facebook/esm2_t48_15B_UR50D` | 7× `.bin` shards | ~60.5 GB |

⚠️ **Gotchas:** the 3B and 15B have **no `model.safetensors` — only `pytorch_model-*.bin`**. `use_safetensors=True` will fail on 3B where it succeeds on 650M. And the 650M repo carries safetensors + `.bin` + `tf_model.h5`, so a naive `git clone` pulls ~7.8 GB instead of 2.6 GB — use `allow_patterns`.

### ESM-1v ensemble
`facebook/esm1v_t33_650M_UR90S_{1,2,3,4,5}` — ungated, `pytorch_model.bin` = 2,609,603,341 B each, **13.05 GB for the ensemble**.
⚠️ **No declared license**: these repos have no README, no `cardData`, no `license:` tag. Upstream `facebookresearch/esm` is MIT but that is an inference. Flag in any artifact statement. (Not needed anyway — ESM-1v single and ensemble scores are already precomputed in ProteinGym, §1.4.)

### ESM-3 — **ungated, MIT**
`biohub/esm3-sm-open-v1` — `gated: false`, no `extra_gated_*` fields, no click-through, no token. Card states "This repository is under a MIT license."
Sizes: `esm3_sm_open_v1.pth` **2.80 GB**, function decoder 1.30 GB, structure decoder 1.24 GB, structure encoder 62.3 MB; **full repo ≈ 5.5 GB**.
`EvolutionaryScale/esm3-open` does **not** exist (404) — use the `biohub` id.
Note: the card documents training-data safety filtering (virus-related data and USDA Select Agent organisms removed) — relevant if you make corpus-coverage claims.

### ESM-C
`biohub/ESMC-300M` (1.33 GB), `ESMC-600M` (2.30 GB), `ESMC-6B` (25.41 GB) — all ungated. License tags are **`mit` + `other`**, meaning some component carries non-MIT third-party terms; read `THIRD_PARTY_NOTICE.md` before commercial use.

### Bonus — directly relevant to a temporal-contamination study
The `biohub` org hosts assets that map onto this project unusually well:
- **`biohub/ESMFold2-Experimental-Cutoff2025`** and `-Experimental-Fast-Cutoff2025` — explicit **training-data-cutoff variants**, close to the temporal control the UniRef-by-year analysis is trying to build.
- **Training-checkpoint ladders**: `ESMC-{300M,600M}-step{250k,500k,750k,1000k,1500k}` — lets you measure *when during training* homolog knowledge is acquired, complementing AMPLIFY's *which corpus year*.
- SAE interpretability probes: `ESMC-{300M,600M,6B}-sae-k64-codebook16384`.

---

## 9. Environment notes (things that will waste your time otherwise)

**SSL: `marks.hms.harvard.edu` fails certificate verification.** The server omits the `InCommon RSA OV SSL CA 3` intermediate, so the system CA bundle alone fails with "unable to get local issuer certificate". A patched bundle is already built at `/scratch/author/icbinb/.cache/ca-bundle-plus.crt`:
```bash
export CURL_CA_BUNDLE=/scratch/author/icbinb/.cache/ca-bundle-plus.crt
```
To rebuild:
```bash
curl -sk http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt -o i.crt
openssl x509 -inform DER -in i.crt -out i.pem
cat /etc/pki/tls/certs/ca-bundle.crt i.pem > ca-bundle-plus.crt
```
Do **not** use `curl -k` for actual data downloads — use the patched bundle.

**HF cache** is pinned out of `$HOME`: `export HF_HOME=/scratch/author/icbinb/.cache/huggingface`.

**Disk:** 555 TB free on `/scratch`. Not a constraint; bandwidth and the B1 XML conversion are.

---

## 10. Verification status — what was actually tested

**Verified by direct inspection on disk:**
- ProteinGym: 217 substitution CSVs present; zero-shot assay list matches the DMS assay list exactly (`comm`, zero diff both directions); per-assay row counts identical between DMS and zero-shot files; 95 baseline columns enumerated from the header; populated baseline scores confirmed on a k=7 mutant.
- The entire k-distribution (§1.2) was computed from the downloaded CSVs, not copied from the reference file's `DMS_number_multiple_mutants`. The two agree.
- AMPLIFY: all 17 local `model.safetensors` at the expected byte sizes (16 × 473,126,988 + 350M at 1,416,062,764); sha256 of the yearly checkpoints matches values computed independently from HF metadata; `config.json` confirms `hidden_size=640`, `num_hidden_layers=24`, `torch_dtype=float32`, `auto_map` → `amplify.AMPLIFYConfig`/`amplify.AMPLIFY`.
- **Blocker B5 confirmed locally**: `diff AMPLIFY_120M_2011/rmsnorm.py main/rmsnorm.py` → differ (995 B vs 1172 B).
- Pfam: all downloads pass `gzip -t`; release/license read from the downloaded files themselves.
- Licenses read from downloaded LICENSE files: ProteinGym MIT, PLM-GUARD Apache-2.0, Pfam CC0. `pLMs-interpretability` confirmed to have **no** LICENSE.
- Dead GREMLIN endpoints confirmed by observed HTTP status codes (404 / 000 / 403).

**NOT yet verified — do this once the modern Python env is ready:**
- **No AMPLIFY checkpoint has been loaded and run.** System Python here is 3.6.8, too old for current `transformers`. Files are byte-correct, but a real `AutoModel.from_pretrained(..., revision=..., trust_remote_code=True)` forward pass is still pending. **Do this before building on the yearly sweep**, and specifically confirm fp32 vs bf16 behaviour given B5.
- The categorical Jacobian script has not been executed (needs `fair-esm` + jax).
- Historical UniRef tarball structure (§3) is from HTTP range-request header parsing, not a full download.
- The ESM/UniRef figures in §3 and §8 come from HF/UniProt API metadata, not from downloaded files.

## Layout on disk

```
/scratch/author/icbinb/
├── data/
│   ├── ASSETS.md                      <- this file
│   ├── MULTI_MUTANT_INVENTORY.md      <- full 69-assay k-breakdown
│   ├── k_distribution.py / .json      <- script + raw per-assay k counts
│   ├── proteingym/                    ~14 GB  (incl. 4.9 GB MSAs, 4.9 GB zero-shot scores)
│   ├── amplify/                       8.4 GB  (14 yearly + base + main + 350M)
│   ├── pfam/                          361 MB
│   ├── scope/                         71 MB
│   └── plmguard_zenodo/               8.82 GB when complete (in flight)
├── repos/
│   ├── ProteinGym/                    182 MB
│   ├── data-saturation-and-scaling/   24 MB   (Spinner et al. — yearly results CSVs)
│   ├── pLMs-interpretability/         36 MB
│   └── PLMGuard/                      1.2 MB
└── .cache/
    ├── ca-bundle-plus.crt             <- SSL fix
    └── huggingface/                   <- HF_HOME
```

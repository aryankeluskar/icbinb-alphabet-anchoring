# Perturbation Axes Design

**Date:** 2026-08-13
**Purpose:** Formalize the protein selection, relabeling regimes, and biological rationale for the alphabet-anchoring experiment. This document serves as both the design record and the source text for the paper's methodology section.

---

## 1. Overview

The experiment applies a single intervention — a global bijective relabeling of the amino-acid alphabet — and measures whether a model's contact prediction survives it. The intervention provably preserves every alignment column statistic (entropies, mutual information, the true contact map), so any performance drop is attributable to the model's representation, not to lost information.

Three axes of variation structure the design:

| Axis | Symbol | Levels | What it varies |
|---|---|---|---|
| **Dose** | $m$ | 2, 6, 10, 20 | Number of residue types relabeled |
| **Regime** | — | random, In-class, Cross-class | Which residue types are swapped |
| **Model** | — | ESM-2 (5 sizes), MSA-T, AMPLIFY, Potts | Architecture, capacity, training data |

The dose axis controls the magnitude of the intervention. The regime axis controls its *kind*: In-class swaps preserve biochemistry, Cross-class swaps violate it, and random swaps are indifferent. Comparing the three at matched $m$ separates "the input got harder" from "the model is keyed on identity." The model axis tests whether the failure is specific to one lab's tokenizer or is a property of the pretraining recipe.

---

## 2. Protein Selection

### 2.1 Selection Criteria

Proteins were selected to satisfy three constraints simultaneously:

1. **Structural ground truth available.** A deposited PDB chain with $\geq 90\%$ sequence identity to the query, so that experimental $C_\beta$–$C_\beta$ contact maps can be computed and precision is meaningful.

2. **MSA depth sufficient for Potts.** A multiple sequence alignment with $\geq 1000$ sequences (for the crossing experiment, C10), so the classical baseline is not a strawman.

3. **Sequence length in a workable range.** $L \in [51, 164]$, so that $L \times 20$ forward passes for the categorical Jacobian are computationally tractable and the top-$L/2$ contact set is large enough to give a non-trivial overlap measurement.

### 2.2 Selected Proteins

| Protein | Length | PDB (chain) | Identity | MSA depth | Role |
|---|---|---|---|---|---|
| Ubiquitin | 76 | 1UBQ (A) | 100% | 16,228 | Headline target: 14.4× chance precision, deepest MSA |
| Protein G B1 | 56 | 1PGB (A) | 100% | 3,109 | Second crossing target: 6.3× chance, shallower MSA |
| Cytochrome c | 104 | 1HRC (A) | 88.5% | — | C6 matched-Hamming control (precision excluded) |
| T4 Lysozyme | 164 | 2LZM (A) | 98.8% | — | Longest sequence, stress-tests the overlap metric |
| PF00018\|BOI2_YEAST | 89 | — | — | — | MSA-T / Potts comparison family |
| PF00027\|P73234_SYNY3 | 77 | — | — | — | MSA-T / Potts comparison family |
| PF00072\|AGMR_PSEAE | 67 | — | — | — | MSA-T / Potts comparison family |
| PF00076\|LA_HUMAN | 100 | — | — | — | MSA-T / Potts comparison family |
| PF00013\|B3RHY5_TRIAD | 51 | — | — | — | Shortest sequence in the sweep |

### 2.3 Biological Rationale for Why Relabeling Should Not Change Structure

The key biological insight: **a global bijective relabeling of the amino-acid alphabet is not a mutation.** It does not change any physical property of the protein. It renames the letters in the sequence while preserving every statistical property of the alignment from which structure is inferred.

More formally, let $X \in \mathcal{A}^{N \times L}$ be an MSA and $\pi: \mathcal{A} \to \mathcal{A}$ a bijection. Then:

- **Column frequencies are permuted, not changed:** $f_i^\pi(\pi(a)) = f_i(a)$. The histogram is the same histogram with its bins renamed.
- **Mutual information is invariant:** $MI^\pi(i,j) = MI(i,j)$, because MI is a function of the joint and marginal distributions, which are permuted consistently.
- **The true contact map is unchanged by construction:** contacts are a physical property of the folded structure, and the structure is determined by the physics of the residues, not by what we call them. If residue 45 is a valine and we call it isoleucine instead, the protein does not refold.

This is what makes the intervention different from a mutation. A real substitution at one position (e.g., V→D at site 45) can destabilize a hydrophobic core and change the contact map. A *global relabeling* that maps V→I everywhere simultaneously changes no physical interaction — it is the same protein wearing a different name tag.

### 2.4 Why Certain Substitutions Are Biologically Silent (In-class Regime)

The In-class regime swaps residues only within BLOSUM62-similar groups. These groups capture the biochemical properties that govern folding:

| Group | Property | Rationale |
|---|---|---|
| I, V, L, M | Aliphatic hydrophobic | Core packing: side chains are sterically interchangeable in hydrophobic cores. I↔V is the most In-class substitution in BLOSUM62 (+3). |
| K, R | Positively charged | Salt bridges: both form electrostatic interactions with D/E. Side-chain length differs but charge is preserved. |
| D, E | Negatively charged | Salt bridges: both form electrostatic interactions with K/R/H. |
| N, Q | Amide | Polar, uncharged: H-bond donors/acceptors with similar geometry. |
| S, T | Small hydroxyl | Polar, small: H-bond capacity with minimal steric cost. |
| F, Y, W | Aromatic | π-stacking, hydrophobic ring: bulky aromatic side chains with planar ring structures. |

A In-class relabeling that swaps I→V at every position simultaneously preserves the hydrophobic core, the salt bridges, and the hydrogen bond network. The protein would fold identically — and indeed, nature performs these substitutions across homologs without changing the fold.

### 2.5 Why In-class Relabeling Is a Stronger Test Than It Looks

The In-class regime is not a "soft" intervention. It changes the identity of up to 15 of 20 residue types — every position in the sequence that holds one of those 15 residues is rewritten. At $m=15$ (the maximum), a 104-residue protein like cytochrome c has 76 of 104 positions changed. That is a Hamming distance of 73%.

What makes it In-class is not the *number* of changes but their *kind*: every substitution stays within a biochemical group. A model that has learned biochemical regularities (hydrophobic cores, salt bridges, aromatic stacking) should recognize the relabeled sequence as the same physical structure. A model that has memorized "this motif with these exact letters → this contact pattern" should fail — even though the biochemistry is intact.

This is exactly what we observe: ESM-2 650M retains 0.655 overlap under In-class $m=20$ (realized 15) but drops to 0.015 under random $m=20$, where the biochemistry is scrambled. The model knows biochemistry; it does not know the alphabet-abstract computation.

### 2.6 Residues Excluded from In-class Relabeling

Five residues — A, C, G, H, P — have no BLOSUM62-similar partner and are fixed points in every In-class permutation:

| Residue | Property | Why no partner |
|---|---|---|
| A (Ala) | Tiny, helix-favoring | No other residue is both tiny and helix-favoring. S is close but loses the helix preference. |
| C (Cys) | Disulfide-forming | Unique thiol chemistry; no other residue forms disulfide bonds. |
| G (Gly) | No side chain | Unique: no $C_\beta$, backbone flexibility. No substitute. |
| H (His) | Imidazole, pH-sensitive | Dual protonation state is unique. N/Q share polarity but not the pH switch. |
| P (Pro) | Cyclic, helix-breaking | Backbone rigidity is unique; no other residue restricts $\phi$ angle. |

This is a fact about amino-acid biochemistry, not a limitation of the experiment. The In-class ceiling of $m=15$ is reported as a realized-$m$ axis, not hidden.

---

## 3. The Three Relabeling Regimes

### 3.1 Random Regime

**Definition:** Sample $m$ residue types uniformly at random from the 20 standard amino acids, then apply a random derangement (no fixed points) over those $m$ types. The remaining $20-m$ types are fixed.

**What it preserves:** Column frequencies, mutual information, the true contact map — all alignment statistics are unchanged by construction.

**What it destroys:** The mapping from residue identity to biochemical class. With $m=20$, every residue is relabeled to a different one, and the biochemical group structure is scrambled: a hydrophobic residue might be mapped to a charged one, an aromatic to a tiny one.

**What it tests:** Whether the model's contact prediction is a function of alignment column statistics (invariant) or of literal residue identity (not invariant).

**Seed formula:** `rng = np.random.default_rng(1000 * seed + m)`, so every run is reproducible.

### 3.2 In-class Regime

**Definition:** Swap residues only within the six BLOSUM62-similar groups listed in §2.4. The groups are processed in a random order, and within each group, a random derangement is applied. The number of residue types touched is capped at $m$.

**What it preserves:** In addition to all alignment statistics, the biochemical class of every residue. A hydrophobic position is still hydrophobic, a charged position is still charged (with the same sign).

**What it destroys:** Literal residue identity. I becomes V, K becomes R, D becomes E — but the physical interactions are preserved.

**What it tests:** Whether the model has learned biochemical regularities (in which case In-class relabeling should be cheap) or has memorized identity-keyed lookups (in which case it should fail regardless).

**Maximum realized $m$:** 15, because A, C, G, H, P have no partner (§2.6).

### 3.3 Cross-class Regime

**Definition:** Swap residues across biochemical classes using a pre-defined list of 10 cross-class pairings:

| Pair | Class violation |
|---|---|
| K↔D | Positive ↔ negative charge |
| R↔E | Positive ↔ negative charge |
| I↔E | Hydrophobic ↔ negative charge |
| V↔K | Hydrophobic ↔ positive charge |
| F↔D | Aromatic ↔ negative charge |
| W↔N | Bulky aromatic ↔ small polar |
| L↔S | Hydrophobic ↔ small hydroxyl |
| M↔Q | Hydrophobic ↔ amide |
| Y↔G | Aromatic ↔ no side chain |
| A↔P | Helix-favoring ↔ helix-breaking |

**What it preserves:** All alignment statistics (same as the other regimes — this is the point of the design).

**What it destroys:** Both literal identity *and* biochemical class. Every swap maximally violates the BLOSUM62 substitution matrix.

**What it tests:** Whether the model degrades with *biochemical* severity rather than with edit distance. The key prediction: Cross-class should fall *faster* than random at intermediate $m$, because a deliberately cross-class derangement preserves biochemical class by accident less often than a uniform random one does. This is what "biochemistry is encoded" predicts, and it is what we observe.

### 3.4 Matched-Hamming Control (C6)

For each (protein, regime, $m$, seed) cell, three sequences are constructed at the **identical Hamming distance** from wild-type:

1. **Permuted:** the bijective relabeling.
2. **Uniform:** random amino-acid substitutions at the same positions, same Hamming distance.
3. **Composition-matched:** substitutions drawn from a distribution matching the wild-type composition at those positions.

All three arms change the same number of residues. They differ only in *which* residues they install. If the permuted sequence retains more contacts than either matched control, the effect is not edit distance — it is the structure of the relabeling.

---

## 4. The Dose-Response Design ($m$-sweep)

The $m$ axis is a dose-response curve. At $m=2$, only two residue types are relabeled — roughly 10% of positions change. At $m=20$, every position is rewritten (random regime). The sweep tests whether the failure is categorical (present at all $m > 0$) or graded (emerges only at high $m$).

**Result:** The failure is graded in $m$ but categorical in regime. At $m=2$, no regime damages the model. At $m=6$, the random and Cross-class arms diverge from In-class. At $m=20$, random and Cross-class sit at chance while In-class retains $0.655$. The crossover happens between $m=2$ and $m=6$.

---

## 5. The Scale Axis

Five ESM-2 sizes span a $375\times$ parameter range:

| Model | Parameters | Random $m=20$ | In-class $m=20$ | Gap |
|---|---|---|---|---|
| t6_8M | 8M | 0.042 (1.8× chance) | 0.211 | +0.170 |
| t12_35M | 35M | 0.027 (1.2×) | 0.367 | +0.340 |
| t30_150M | 150M | 0.028 (1.2×) | 0.623 | +0.596 |
| t33_650M | 650M | 0.017 (0.7×) | 0.501 | +0.484 |
| t36_3B | 3B | 0.035 (1.5×) | 0.526 | +0.491 |

The random arm is pinned at chance at every scale. The In-class arm climbs roughly threefold. The gap grows from +0.170 to +0.491. Capacity buys sensitivity to biochemical similarity and zero alphabet abstraction.

---

## 6. The Crossing (C10)

On the identical restricted candidate set (same $k$, same pair universe), ESM-2 and Potts precisions are directly comparable:

| Protein | ESM-2 identity | ESM-2 $m=20$ | Potts (any) | Ratio before | Ratio after |
|---|---|---|---|---|---|
| Ubiquitin | 0.737 (16.2×) | 0.053 (1.2×) | 0.316 (6.9×) | 2.33× | **0.17×** |
| Protein G | 0.500 (6.3×) | 0.107 (1.3×) | 0.179 (2.2×) | 2.80× | **0.60×** |

ESM-2 starts well above the classical baseline and ends below it under a transformation the classical model does not notice. The crossing margin tracks how good the classical baseline is (deeper MSA → stronger Potts → wider crossing).

---

## 7. Summary of What Each Axis Tests

| Axis | Question | Answer |
|---|---|---|
| $m$ (dose) | Is the failure categorical or graded? | Graded in $m$; crossover at $m \approx 6$ |
| Regime (random vs In-class vs Cross-class) | Is the controlling variable edit distance or biochemical class? | Biochemical class. In-class $\gg$ random at matched $m$; Cross-class $\leq$ random |
| Model size (8M–3B) | Does scale fix it? | No. Random arm flat at chance; In-class arm climbs. Two different axes |
| Architecture (ESM-2, MSA-T, AMPLIFY) | Is it one lab's tokenizer? | No. Three architectures, two tokenizers, same failure |
| Readout (contact map, variant effect) | Is it one readout? | No. Same dissociation on 15 ProteinGym assays |
| Crossing (C10) | Does the neural model become the worse choice? | Yes. 2.33× → 0.17× on ubiquitin |

---

## 8. Protein Sequences

The exact sequences used in the experiments:

**Ubiquitin** (76 aa):
```
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

**Protein G B1** (56 aa):
```
MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```

**Cytochrome c** (104 aa, human):
```
GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE
```

**T4 Lysozyme** (164 aa, pseudo-WT):
```
MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL
```

---

## 9. Example Permutations (seed 0, $m=20$)

The actual relabelings applied to `ACDEFGHIKLMNPQRSTVWY`:

| Regime | Image | Cycle structure | Realized $m$ |
|---|---|---|---|
| Domain | `ACDEFGHIKLMNPQRSTVWY` | identity | 0 |
| Random | `LKYGAFDNMTESQWICPRHV` | one 20-cycle | 20 |
| In-class | `ACEDWGHLRMVQPNKTSIYF` | (DE)(FWY)(ILMV)(KR)(NQ)(ST) | 15 |
| Cross-class | `PCFIDYHEVSQWAMRLTKNG` | (AP)(DF)(EI)(GY)(KV)(LS)(MQ)(NW) | 16 |

# ProteinGym v1.3 — multi-mutant (k>=2) inventory

Computed directly from the 217 downloaded per-assay CSVs by counting `:` separators in the `mutant` column.

Source: `data/proteingym/DMS_ProteinGym_substitutions/`  |  script: `data/k_distribution.py`  |  raw: `data/k_distribution.json`

## Headline

| | mutants | share |
|---|---:|---:|
| k=1 (singles) | 696311 | 28.24% |
| **k>=2 (multi)** | **1769456** | **71.76%** |
| TOTAL | 2465767 | 100% |

**The substitutions benchmark is 72% multi-mutant by row count.** 69 of 217 assays contain k>=2. Max observed k = 44.

## Global k distribution

| k | mutants | share |
|---:|---:|---:|
| 1 | 696311 | 28.239% |
| 2 | 826245 | 33.509% |
| 3 | 84134 | 3.412% |
| 4 | 187850 | 7.618% |
| 5 | 92213 | 3.740% |
| 6 | 130395 | 5.288% |
| 7 | 149318 | 6.056% |
| 8 | 134593 | 5.458% |
| 9 | 90108 | 3.654% |
| 10 | 45816 | 1.858% |
| 11 | 17983 | 0.729% |
| 12 | 5907 | 0.240% |
| 13 | 1673 | 0.068% |
| 14 | 573 | 0.023% |
| 15 | 339 | 0.014% |
| 16 | 282 | 0.011% |
| 17 | 237 | 0.010% |
| 18 | 198 | 0.008% |
| 19 | 158 | 0.006% |
| 20 | 120 | 0.005% |
| 21 | 96 | 0.004% |
| 22 | 74 | 0.003% |
| 23 | 59 | 0.002% |
| 24 | 47 | 0.002% |
| 25 | 27 | 0.001% |
| 26 | 24 | 0.001% |
| 27 | 12 | 0.000% |
| 28 | 6 | 0.000% |
| 29 | 1 | 0.000% |
| 31 | 3 | 0.000% |
| 32 | 8 | 0.000% |
| 33 | 17 | 0.001% |
| 34 | 39 | 0.002% |
| 35 | 80 | 0.003% |
| 36 | 100 | 0.004% |
| 37 | 146 | 0.006% |
| 38 | 140 | 0.006% |
| 39 | 124 | 0.005% |
| 40 | 98 | 0.004% |
| 41 | 81 | 0.003% |
| 42 | 65 | 0.003% |
| 43 | 55 | 0.002% |
| 44 | 12 | 0.000% |

## Per-assay breakdown — all 69 assays with k>=2

| DMS_id | UniProt | taxon | len | total | k=1 | k=2 | k=3 | k=4 | k>=5 | max k | MSA Neff/L |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPG1_STRSG_Olson_2014 | SPG1_STRSG | Prokaryote | 448 | 536962 | 1045 | 535917 | 0 | 0 | 0 | 2 | 0 |
| HIS7_YEAST_Pokusaeva_2019 | HIS7_YEAST | Eukaryote | 220 | 496137 | 168 | 1475 | 7627 | 25927 | 460940 | 28 | 23.97 |
| PHOT_CHLRE_Chen_2023 | PHOT_CHLRE | Eukaryote | 118 | 167529 | 2122 | 176 | 978 | 3565 | 160688 | 15 | 5215.65 |
| SPG1_STRSG_Wu_2016 | SPG1_STRSG | Prokaryote | 448 | 149360 | 76 | 2091 | 26019 | 121174 | 0 | 4 | 1.34 |
| GRB2_HUMAN_Faure_2021 | GRB2_HUMAN | Human | 217 | 63366 | 1034 | 62332 | 0 | 0 | 0 | 2 | 6.78 |
| GFP_AEQVI_Sarkisyan_2016 | GFP_AEQVI | Eukaryote | 238 | 51714 | 1084 | 12777 | 12336 | 9387 | 16130 | 15 | 0.06 |
| CAPSD_AAV2S_Sinai_2021 | CAPSD_AAV2S | Virus | 735 | 42328 | 532 | 10833 | 6906 | 6646 | 17411 | 28 | 0.26 |
| PABP_YEAST_Melamed_2013 | PABP_YEAST | Eukaryote | 577 | 37708 | 1187 | 36521 | 0 | 0 | 0 | 2 | 1.42 |
| Q8WTC7_9CNID_Somermeyer_2022 | Q8WTC7_9CNID | Eukaryote | 238 | 33510 | 1201 | 11260 | 11641 | 5762 | 3646 | 43 | 0.51 |
| Q6WV12_9MAXI_Somermeyer_2022 | Q6WV12_9MAXI | Eukaryote | 222 | 31401 | 1141 | 15992 | 8762 | 3575 | 1931 | 13 | 0.44 |
| D7PM05_CLYGR_Somermeyer_2022 | D7PM05_CLYGR | Eukaryote | 235 | 24515 | 1169 | 10148 | 6315 | 3417 | 3466 | 23 | 0.6 |
| RASK_HUMAN_Weng_2022_abundance | RASK_HUMAN | Human | 188 | 26012 | 3066 | 22946 | 0 | 0 | 0 | 2 | 146.37 |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | RASK_HUMAN | Human | 188 | 24873 | 3084 | 21789 | 0 | 0 | 0 | 2 | 146.37 |
| A4_HUMAN_Seuma_2022 | A4_HUMAN | Human | 770 | 14811 | 796 | 14015 | 0 | 0 | 0 | 2 | 0.08 |
| YAP1_HUMAN_Araya_2012 | YAP1_HUMAN | Human | 504 | 10075 | 362 | 9713 | 0 | 0 | 0 | 2 | 0.26 |
| F7YBW8_MESOW_Aakre_2015 | F7YBW8_MESOW | Prokaryote | 93 | 9192 | 37 | 499 | 2798 | 5858 | 0 | 4 | 176.23 |
| F7YBW8_MESOW_Ding_2023 | F7YBW8_MESOW | Prokaryote | 93 | 7922 | 80 | 86 | 603 | 2159 | 4994 | 10 | 176.23 |
| DLG4_HUMAN_Faure_2021 | DLG4_HUMAN | Human | 724 | 6976 | 1280 | 5696 | 0 | 0 | 0 | 2 | 0.46 |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | HECD1_HUMAN | Human | 72 | 5586 | 1244 | 4342 | 0 | 0 | 0 | 2 | 17.22 |
| POLG_PESV_Tsuboyama_2023_2MXD | POLG_PESV | Virus | 53 | 5130 | 995 | 4135 | 0 | 0 | 0 | 2 | 262.74 |
| GCN4_YEAST_Staller_2018 | GCN4_YEAST | Eukaryote | 281 | 2638 | 33 | 55 | 149 | 380 | 2021 | 44 | 0.66 |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | UBE4B_HUMAN | Human | 69 | 3622 | 1118 | 2504 | 0 | 0 | 0 | 2 | 506.27 |
| MYO3_YEAST_Tsuboyama_2023_2BTT | MYO3_YEAST | Eukaryote | 61 | 3297 | 947 | 2350 | 0 | 0 | 0 | 2 | 208.4 |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | AMFR_HUMAN | Human | 47 | 2972 | 820 | 2152 | 0 | 0 | 0 | 2 | 26.51 |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | SPTN1_CHICK | Eukaryote | 60 | 3201 | 1051 | 2150 | 0 | 0 | 0 | 2 | 285.61 |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | OBSCN_HUMAN | Human | 65 | 3197 | 1213 | 1984 | 0 | 0 | 0 | 2 | 378.54 |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | CSN4_MOUSE | Eukaryote | 72 | 3295 | 1353 | 1942 | 0 | 0 | 0 | 2 | 48.23 |
| SDA_BACSU_Tsuboyama_2023_1PV0 | SDA_BACSU | Prokaryote | 44 | 2770 | 834 | 1936 | 0 | 0 | 0 | 2 | 20.05 |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | DOCK1_MOUSE | Eukaryote | 66 | 2915 | 1213 | 1702 | 0 | 0 | 0 | 2 | 368.28 |
| YNZC_BACSU_Tsuboyama_2023_2JVD | YNZC_BACSU | Prokaryote | 39 | 2300 | 714 | 1586 | 0 | 0 | 0 | 2 | 40.99 |
| VILI_CHICK_Tsuboyama_2023_1YU5 | VILI_CHICK | Eukaryote | 65 | 2568 | 1202 | 1366 | 0 | 0 | 0 | 2 | 761.62 |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | CBX4_HUMAN | Human | 50 | 2282 | 917 | 1365 | 0 | 0 | 0 | 2 | 269.65 |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | NKX31_HUMAN | Human | 61 | 2482 | 1149 | 1333 | 0 | 0 | 0 | 2 | 147.96 |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | RCRO_LAMBD | Virus | 63 | 2278 | 1195 | 1083 | 0 | 0 | 0 | 2 | 803.73 |
| SPA_STAAU_Tsuboyama_2023_1LP1 | SPA_STAAU | Prokaryote | 55 | 2105 | 1035 | 1070 | 0 | 0 | 0 | 2 | 37.48 |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | DNJA1_HUMAN | Human | 65 | 2264 | 1216 | 1048 | 0 | 0 | 0 | 2 | 547.68 |
| ISDH_STAAW_Tsuboyama_2023_2LHR | ISDH_STAAW | Prokaryote | 55 | 1944 | 940 | 1004 | 0 | 0 | 0 | 2 | 713.67 |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | BBC1_YEAST | Eukaryote | 64 | 2069 | 1084 | 985 | 0 | 0 | 0 | 2 | 271.76 |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | YAIA_ECOLI | Prokaryote | 52 | 1890 | 928 | 962 | 0 | 0 | 0 | 2 | 16.25 |
| MBD11_ARATH_Tsuboyama_2023_6ACV | MBD11_ARATH | Eukaryote | 66 | 2116 | 1155 | 961 | 0 | 0 | 0 | 2 | 24.91 |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | PITX2_HUMAN | Human | 52 | 1824 | 938 | 886 | 0 | 0 | 0 | 2 | 191.76 |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | PR40A_HUMAN | Human | 63 | 2033 | 1163 | 870 | 0 | 0 | 0 | 2 | 56.72 |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | NUSA_ECOLI | Prokaryote | 69 | 2028 | 1306 | 722 | 0 | 0 | 0 | 2 | 598.86 |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | EPHB2_HUMAN | Human | 66 | 1960 | 1239 | 721 | 0 | 0 | 0 | 2 | 126.33 |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | CBPA2_HUMAN | Human | 72 | 2068 | 1357 | 711 | 0 | 0 | 0 | 2 | 43.64 |
| SR43C_ARATH_Tsuboyama_2023_2N88 | SR43C_ARATH | Eukaryote | 48 | 1583 | 889 | 694 | 0 | 0 | 0 | 2 | 253.55 |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | BCHB_CHLTE | Prokaryote | 52 | 1572 | 890 | 682 | 0 | 0 | 0 | 2 | 51.95 |
| FECA_ECOLI_Tsuboyama_2023_2D1U | FECA_ECOLI | Prokaryote | 72 | 1886 | 1219 | 667 | 0 | 0 | 0 | 2 | 141.6 |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | MAFG_MOUSE | Eukaryote | 41 | 1429 | 762 | 667 | 0 | 0 | 0 | 2 | 3.89 |
| CUE1_YEAST_Tsuboyama_2023_2MYX | CUE1_YEAST | Eukaryote | 52 | 1580 | 955 | 625 | 0 | 0 | 0 | 2 | 7.64 |
| THO1_YEAST_Tsuboyama_2023_2WQG | THO1_YEAST | Eukaryote | 41 | 1279 | 656 | 623 | 0 | 0 | 0 | 2 | 206.87 |
| CATR_CHLRE_Tsuboyama_2023_2AMI | CATR_CHLRE | Eukaryote | 72 | 1903 | 1340 | 563 | 0 | 0 | 0 | 2 | 1107.51 |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | ODP2_GEOSE | Prokaryote | 44 | 1134 | 669 | 465 | 0 | 0 | 0 | 2 | 397.86 |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | TCRG1_MOUSE | Eukaryote | 37 | 1058 | 621 | 437 | 0 | 0 | 0 | 2 | 79.56 |
| SPG2_STRSG_Tsuboyama_2023_5UBS | SPG2_STRSG | Prokaryote | 56 | 1451 | 1029 | 422 | 0 | 0 | 0 | 2 | 56.55 |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | NUSG_MYCTU | Prokaryote | 55 | 1380 | 1019 | 361 | 0 | 0 | 0 | 2 | 320.33 |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | TNKS2_HUMAN | Human | 59 | 1479 | 1118 | 361 | 0 | 0 | 0 | 2 | 195.89 |
| PSAE_PICP2_Tsuboyama_2023_1PSE | PSAE_PICP2 | Prokaryote | 68 | 1579 | 1219 | 360 | 0 | 0 | 0 | 2 | 1.85 |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | UBR5_HUMAN | Human | 58 | 1453 | 1094 | 359 | 0 | 0 | 0 | 2 | 17.55 |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | RBP1_HUMAN | Human | 52 | 1332 | 975 | 357 | 0 | 0 | 0 | 2 | 976.49 |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | RFAH_ECOLI | Prokaryote | 55 | 1326 | 969 | 357 | 0 | 0 | 0 | 2 | 258.65 |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | SRBS1_HUMAN | Human | 67 | 1556 | 1211 | 345 | 0 | 0 | 0 | 2 | 363.68 |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | RL20_AQUAE | Prokaryote | 59 | 1461 | 1121 | 340 | 0 | 0 | 0 | 2 | 1816.95 |
| RPC1_BP434_Tsuboyama_2023_1R69 | RPC1_BP434 | Virus | 61 | 1459 | 1124 | 335 | 0 | 0 | 0 | 2 | 9091.45 |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | SAV1_MOUSE | Eukaryote | 43 | 965 | 679 | 286 | 0 | 0 | 0 | 2 | 105.1 |
| RCD1_ARATH_Tsuboyama_2023_5OAO | RCD1_ARATH | Eukaryote | 57 | 1261 | 988 | 273 | 0 | 0 | 0 | 2 | 28.23 |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | RD23A_HUMAN | Human | 44 | 1019 | 798 | 221 | 0 | 0 | 0 | 2 | 182.36 |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | RAD_ANTMA | Eukaryote | 54 | 912 | 774 | 138 | 0 | 0 | 0 | 2 | 777.81 |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | PIN1_HUMAN | Human | 39 | 802 | 686 | 116 | 0 | 0 | 0 | 2 | 287.66 |

## Assays with DEEP combinatorial data (k>=5 present)

| DMS_id | max k | n(k>=5) | total |
|---|---:|---:|---:|
| GCN4_YEAST_Staller_2018 | 44 | 2021 | 2638 |
| Q8WTC7_9CNID_Somermeyer_2022 | 43 | 3646 | 33510 |
| HIS7_YEAST_Pokusaeva_2019 | 28 | 460940 | 496137 |
| CAPSD_AAV2S_Sinai_2021 | 28 | 17411 | 42328 |
| D7PM05_CLYGR_Somermeyer_2022 | 23 | 3466 | 24515 |
| PHOT_CHLRE_Chen_2023 | 15 | 160688 | 167529 |
| GFP_AEQVI_Sarkisyan_2016 | 15 | 16130 | 51714 |
| Q6WV12_9MAXI_Somermeyer_2022 | 13 | 1931 | 31401 |
| F7YBW8_MESOW_Ding_2023 | 10 | 4994 | 7922 |

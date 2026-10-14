# Audit: dimensiunile matricelor și rangul numeric pe experimente

Verificat direct din artefactele din `data/` cu NumPy 2.5.3, Python 3.13.9
(Apple M5). **Nu toate experimentele folosesc aceeași matrice** — tabelul de
mai jos e sursa adevărului.

## Matricele de intrare

| Matrice | Shape | Rang numeric (convenția NumPy) | σ₁ | σ_min | Proveniență |
|---|---|---|---|---|---|
| `data/A.npy` | 28380×406 | **387** | 333.90 | 2.74e-13 | `src/build_matrix.py` pe `nyc-rolling-sales.csv` filtrat (min_price=10000, 895 rânduri eliminate) |
| `data/A_numeric.npy` | 28380×6 | **6** (rang plin) | 291.73 | 0.352 | aceleași rânduri, doar cele 6 coloane numerice din `meta.json` |
| `data/y.npy` | (28380,) | — | — | — | prețuri de vânzare (ținta regresiei) |
| sintetică (`scripts/test_synthetic_rank.py`) | 100×5 | **3** (by construction) | — | — | `rng.default_rng(42)`, coloanele 3–4 dependente liniar |

Notă: `A` completă are rang 387 < 406 — există 19 de direcții aproape nule
(one-hot-uri cu suprapuneri de categorii). Convenția de prag din regresie
(`tol=1e-10` relativ la σ₁, `src/regression.py:29-30`) tăie mai agresiv decât
`np.linalg.matrix_rank`; rangul folosit efectiv de regresie e cel scris în
`results/regression_metrics*.csv`.

## Pe experimente

| Experiment | Script | Matrice | Shape | Rang |
|---|---|---|---|---|
| Validare SVD custom vs NumPy | `src/validate_svd.py` | `A.npy` | 28380×406 | 387 (raportat în `results/svd_validation.csv`) |
| Validare pe sintetic | `scripts/test_synthetic_rank.py` | sintetică | 100×5 | 3 (așteptat), comparat custom vs NumPy |
| Reproducibilitate | `scripts/verify_reproducibility.py` | `A.npy` | 28380×406 | 387 |
| Rank-k, matrice completă | `src/rank_k.py --A data/A.npy` | `A.npy` | 28380×406 | 387; k90 = 319, k95 = 343 |
| Rank-k, numeric | `src/rank_k.py --A data/A_numeric.npy` | `A_numeric.npy` | 28380×6 | 6; k90 = 4, k95 = 4 |
| Anomalii SVD (custom) | `src/anomaly_detectors.py --methods svd` | `A.npy` (implicit) | 28380×406 | 387 |
| Anomalii K-Means / Isolation Forest | `src/anomaly_detectors.py --methods kmeans iforest` | `A.npy` | 28380×406 | — |
| Anomalii, variantă numerică | `... --A data/A_numeric.npy --tag _numeric` | `A_numeric.npy` | 28380×6 | — |
| Acord (Jaccard) | `src/agreement.py [--tag _numeric]` | (citește CSV-uri) | — | — |
| Corelație scor vs normă | `src/corelatie_norma.py [A.npy\|A_numeric.npy]` | ambele, pe rând | 28380×406 / 28380×6 | 387 / 6 |
| Regresie LS + ridge (brut și log) | `src/regression.py [--log-target]` | `A.npy` + `y.npy` | 28380×406 | cu prag tol=1e-10·σ₁ (vezi CSV) |

Concluzii de reținut pentru revizuire:

- Experimentele pe varianta „numerică” (6 coloane) au rang plin și un spectru
  complet diferit — nu le putem trata ca pe un caz particular al matricei complete.
- Rangul raportat diferă după convenția de prag (NumPy default vs tol=1e-10 în
  regresie); documentăm mereu care convenție a fost folosită.

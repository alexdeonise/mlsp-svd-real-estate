# Baseline: snapshot al rezultatelor inițiale

Conținutul acestui director este copia exactă a tuturor CSV-urilor din
`results/` de la crearea branch-ului `revision/custom-svd-pipeline` (înainte de
orice revizuire a pipeline-ului SVD custom). **Nu se suprascrie**; orice
re-rulare comparabilă se salvează într-un director nou, de forma
`results/<config>/`.

## Configurația asociată (provenienta completă în `docs/mediu-reproductibilitate.md`)

- Date de intrare: `data/A.npy` (28380×406), `data/A_numeric.npy` (28380×6),
  `data/y.npy`, generate de `src/build_matrix.py` cu `robust=false`,
  `drop_top_pct=0.0`, `min_price=10000` (`data/meta.json`).
- Seed-uri: 42 pentru K-Means / Isolation Forest (`src/anomaly_detectors.py`).
- Mediu: Python 3.13.9, NumPy 2.5.3, scikit-learn 1.9.0, Apple M5.
- Rang numeric: A = 387, A_numeric = 6 (`docs/audit-matrice-experimente.md`).

## Corespondența fișier → experiment → matrice

| Fișier | Experiment | Matrice |
|---|---|---|
| `svd_validation.csv` | validare SVD custom vs NumPy | A (28380×406) |
| `rank_k_full.csv` | rank-k, curba erorii | A |
| `rank_k_numeric.csv` | rank-k, curba erorii | A_numeric |
| `anomalies_svd.csv` | anomalii, scor SVD custom (k=5) | A |
| `anomalies_svd_numeric.csv` | anomalii, scor SVD custom | A_numeric |
| `anomalies_kmeans.csv` / `anomalies_iforest.csv` | anomalii, k=4 / contamination=0.05 | A |
| `anomalies_*_numeric.csv` | idem | A_numeric |
| `agreement.csv` | Jaccard între detectoare | A |
| `agreement_numeric.csv` | Jaccard între detectoare | A_numeric |
| `corelatie_norma.csv` | corelație scor vs norma rândului | A |
| `corelatie_norma_numeric.csv` | idem | A_numeric |
| `regression_metrics.csv` | least squares + ridge α∈{100,1000,10000,100000}, preț brut | A + y |
| `regression_metrics_log.csv` | idem, ținta log1p(preț) | A + y |
| `regression_neighborhood_concentration*.csv` | concentrarea reziduurilor pe cartiere | A + y |
| `least_squares_top50_*.csv` / `ridge_alpha_*_top50_*.csv` | top 50 supra/subevaluate | A + y |

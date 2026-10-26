# mlsp-svd-real-estate

Lucrarea 3 din grupul MLSP: un pipeline SVD implementat de la zero, aplicat pe ~29.000 de tranzacții
imobiliare din New York. Întrebarea de paper: poate detecta proprietăți greșit evaluate la fel de bine
ca metodele de bibliotecă, și cât de mult sunt de acord SVD, K-Means și Isolation Forest când marchează
aceleași proprietăți drept anomalii?

Echipa: Maria (nucleul SVD și modelarea prețului), David (detectoarele de anomalii, acordul dintre ele,
sensibilitate, robustețe, reproducibilitate).

Punctul de plecare e proiectul Mariei: https://github.com/mariatimbus/NYC_RealEstate (SVD-ul custom e
în `src/custom_svd.py` acolo). Datele: https://www.kaggle.com/datasets/new-york-city/nyc-property-sales

## Planul tău, pas cu pas

- [Maria](docs/plan-maria.md)
- [David](docs/plan-david.md)
- [Reproductibilitatea numerică a SVD-ului custom](docs/reproductibilitate-numerica.md) — matricea Gram, conditioning la pătrat, criterii de convergență, praguri de rang (M4)

## Setup (5 minute)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pune nyc-rolling-sales.csv de pe Kaggle in data/
python src/build_matrix.py data/nyc-rolling-sales.csv     # -> data/A.npy, data/y.npy, data/meta.json
```

Nu ai nevoie de GPU. Totul rulează pe laptop.

## Structura

```
src/build_matrix.py      curatare + one-hot + standardizare, exact ca in raportul Mariei -> A (m x n)
src/validate_svd.py      compara un SVD custom cu numpy: reconstructie, ortogonalitate, timp      (Maria)
src/rank_k.py            energia retinuta la k = 5/10/20/50, pragurile 90% / 95%, custom vs numpy  (Maria)
src/regression.py        least squares + ridge cu evaluare 80/20 train/test, validare pentru alpha (Maria)
src/anomaly_detectors.py trei scoruri: SVD reconstruction, K-Means distance, Isolation Forest    (David)
src/agreement.py         Jaccard intre liste, cate proprietati sunt marcate de toate metodele    (David)
scripts/compare_solvers.py  validarea M1: aceleasi tolerante, custom vs numpy pe ambele matrici
results/                 CSV-uri si liste de proprietati
figures/                 figurile, generate din scripturi
```

## Tutorial 1 (Maria, până vineri): validarea SVD-ului tău pe matricea reală

Copiază `custom_svd.py` din repo-ul tău în `src/`, apoi:

```bash
python src/validate_svd.py --custom src/custom_svd.py --func SVD
```

Scriptul încarcă `data/A.npy` (28380×406), rulează SVD-ul tău și `numpy.linalg.svd`, și scrie în
`results/svd_validation.csv`: eroarea relativă de reconstrucție ‖A − UΣVᵀ‖/‖A‖, ‖UᵀU − I‖, ‖VᵀV − I‖,
diferența maximă între valorile singulare, timpul fiecăruia. În raportul tău ai făcut asta pe o matrice
de test de 180×24; acum o facem pe cea reală. Cifrele intră în abstract.

Următorul pas (luni): `python src/rank_k.py`, curba erorii în funcție de k și k-ul la care păstrezi
90% și 95% din informație. Apoi regresia (joi), într-un fișier nou `src/regression.py`.

## Tutorial 2 (David, până vineri): două detectoare fără SVD

```bash
python src/anomaly_detectors.py --methods kmeans iforest --pct 95
```

Rulează K-Means (k=4, scor = distanța la centroidul propriu) și Isolation Forest pe `A`, marchează ca
anomalii proprietățile peste percentila 95 a fiecărui scor, și scrie `results/anomalies_<metoda>.csv`
(indexul rândului, scorul, marcat/nu). Asta e livrabilul tău de vineri și nu depinde de Maria.

De luni adaugi `--methods svd`, care folosește SVD-ul Mariei (eroarea de reconstrucție la rang k).
Apoi:

```bash
python src/agreement.py          # Jaccard pe fiecare pereche, cate sunt marcate de toate trei
python src/anomaly_detectors.py --methods svd kmeans iforest --pct 90    # sensibilitate la prag
python src/anomaly_detectors.py --methods svd --k 3                      # sensibilitate la rang
```

Robustețea (duminică): rulezi tot fără top 1% cele mai scumpe tranzacții (`--drop-top-pct 1`) și cu
standardizare robustă (`build_matrix.py --robust`), și vezi cât se schimbă listele.

De citit: Isolation Forest, Liu et al. 2008, https://ieeexplore.ieee.org/document/4781136

## Rezultatele

Un experiment e gata când scriptul îl reproduce de la zero și CSV-ul sau figura e comisă în `results/`
sau `figures/`. Nu „merge la mine în notebook".

## Rezultate actuale

Setul final de date conține 28.380 de tranzacții și 406 caracteristici.
Tranzacțiile cu prețul de vânzare sub 10.000 USD sunt eliminate în mod
consistent înainte de construirea matricelor folosite în toate experimentele.

- Rangul numeric al matricei complete: 387 (numpy) / 393 (custom — cele 6
  direcții în plus sunt zgomot de la formarea AᵀA, vezi
  `docs/reproductibilitate-numerica.md`)
- Rank-k, matrice completă: k90 = 319, k95 = 343; energia la k=5 este
  **3.6%**, la k=50 este 22.6% (ambele solver-e coincid până la ~1e-15)
- Rank-k, doar variabile numerice: k90 = k95 = 4; energia la k=5 este 100%
- Regresie cu evaluare corectă train/test (80/20, seed 42, standardizare
  fitată doar pe train, α Ridge ales pe validare):
  - preț brut, least squares: test R² ≈ 0.459 (in-sample era 0.727 — gap
    de generalizare, nu de calcul)
  - log(1+preț), Ridge α=1000 (ales pe validare): test R² ≈ 0.933,
    RMSE ≈ 8.06M USD

### Solver custom vs numpy (M1)

`compute_svd` din `src/svd_common.py` folosește implicit solverul custom,
cu același prag de rang și aceleași toleranțe pentru ambele implementări.
Validarea completă (valori singulare, reconstrucție, ortogonalitate,
predicții de regresie) este în `results/solver_comparison.csv`, generată
cu `python scripts/compare_solvers.py`.

### De ce este R² catastrofal de mic pentru log(1+preț) fără regularizare?

Least squares pe ținta logaritmică, evaluat corect pe test, dă R² ≈ −379.
Cauza nu e calculul, ci transformarea inversă: erori mici în spațiul log
devin predicții exponențiale. Diagnosticul din
`results/log_target_diagnostics.csv`:

- predicția maximă pe test: **47.8 miliarde USD** (maximul real: 2.2 miliarde);
- 9 predicții peste 100M USD, una peste 1 miliarde;
- top 5 cele mai mari reziduuri contribuie cu 99.996% la SS_res — R² e
  distrus de un număr de puncte care se numără pe degete.

Ridge comprimă exact aceste direcții explozive: la α=1000 predicția maximă
scade la 1.9 miliarde USD și R² pe test ajunge la 0.933. Figura
`figures/regression_log_ls_investigation.png` arată predicția de 47.8
miliarde și, în același timp, că reziduurile centrale 99% sunt strânse
în jurul zero — modelul e bun tipic, dar expm1 amplifică catastrofal
cozile. Valoarea veche R² = −1.98 era aceeași poveste, măsurată in-sample.

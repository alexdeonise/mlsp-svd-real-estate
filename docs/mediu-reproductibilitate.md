# Mediu, versiuni, seed-uri, hardware

Starea mediului la momentul snapshot-ului baseline (octombrie 2026), pe
branch-ul `revision/custom-svd-pipeline`. Orice re-rulare care schimbă una din
valorile de mai jos trebuie să producă un director de rezultate nou, nu să
suprascrie `results/baseline/`.

## Versiuni

| Componentă | Versiune |
|---|---|
| Python | 3.13.9 (arm64) |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| SciPy | 1.18.1 |
| matplotlib | 3.11.1 |

Mediu virtual: `.venv/` din rădăcina repo-ului, instalat din `requirements.txt`.

## Hardware

- CPU: Apple M5 (ARM64, single machine, fără GPU)
- RAM: 24 GB (25769803776 bytes)
- OS: macOS

## Seed-uri

| Unde | Seed | Observații |
|---|---|---|
| `src/anomaly_detectors.py:20-25` | 42 (default) | K-Means k=4, `n_init=10`; Isolation Forest `contamination=0.05` |
| `scripts/verify_reproducibility.py:297,318` | 42 | K-Means și Isolation Forest în verificarea de reproducibilitate |
| `scripts/test_synthetic_rank.py:10` | 42 | `np.random.default_rng(42)` pentru matricea sintetică 100×5 |
| `scripts/check_seed_stability.py:80` | 42, 7, 13, 123 | stabilitatea listelor de anomalii la alte seed-uri |

SVD-ul (custom și NumPy) este determinist și nu folosește seed. `custom_svd.SVD`
primește `TOL=1e-14`, `RTOL=1e-6` ca parametri numerici, nu seed.

## Configurația datelor (din `data/meta.json`)

- `robust: false`, `drop_top_pct: 0.0` (standardizare clasică, fără eliminarea top-ului)
- `min_price: 10000` — 895 de tranzacții sub prag eliminate
- Rezultat: m = 28380, n_full = 406, n_numeric = 6

Configurații alternative menționate în README (robust, drop-top-pct) **nu** sunt
cele cu care a fost generat baseline-ul; dacă se rulează, rezultatele se
salvează într-un director nou cu eticheta configurației.

# Inventarul apelurilor `np.linalg` din pipeline

Scop: fiecare experiment care folosește rutine de algebră liniară din NumPy trebuie
identificat explicit, ca înlocuirea lor cu SVD-ul custom (`src/custom_svd.py:425`)
să fie exhaustivă. Audit făcut pe branch-ul `revision/custom-svd-pipeline`.

## Rezumat

| Rutină | Nr. apeluri | Fișiere |
|---|---|---|
| `np.linalg.svd` | 5 | `src/rank_k.py`, `src/validate_svd.py`, `src/regression.py`, `scripts/test_synthetic_rank.py`, `scripts/verify_reproducibility.py` |
| `np.linalg.matrix_rank` | 3 | `src/rank_k.py`, `src/validate_svd.py`, `scripts/test_synthetic_rank.py` |
| `np.linalg.pinv` | 0 | — nu este folosit; pseudo-inversa e calculată manual din factorii SVD |
| `np.linalg.lstsq` | 0 | — nu este folosit; least squares / ridge se rezolvă manual din factorii SVD |

## Detaliu pe apel

### `np.linalg.svd`

1. **`src/validate_svd.py:54`** — referința NumPy pentru validarea SVD-ului custom.
   - Intrare: `data/A.npy` (28380×406, implicit; suprascriibil prin `--A`).
   - `full_matrices=False`; valorile singulare `sn` și `Vtn` comparate cu
     rezultatul custom (eroare de reconstrucție, ortogonalitate, diferența de
     valori singulare).
   - Acest apel **trebuie păstrat** în orice revizuire: este etalonul de comparație.

2. **`src/rank_k.py:31`** (`compute_uv=False`) — curba de eroare de reconstrucție
   în funcție de k și pragurile k90/k95.
   - Intrare: matrice dată prin `--A` (în practică `data/A.npy` sau
     `data/A_numeric.npy`; nu presupune aceeași matrice — verifică
     `results/rank_k_full.csv` vs `results/rank_k_numeric.csv`).

3. **`src/regression.py:24`** (`compute_svd`, `full_matrices=False`) — baza
   regresiei least squares și ridge.
   - Intrare: matricea din `--A` (implicit `data/A.npy`, 28380×406) și `y.npy`.
   - Rangul propriu-zis se calculează cu pragul `tol=1e-10` relativ la σ₁
     (`src/regression.py:29-30`), **nu** cu `np.linalg.matrix_rank` — cele două
     convenții pot da rezultate diferite.

4. **`scripts/test_synthetic_rank.py:23`** (`compute_uv=False`) — test pe
   matrice sintetică 100×5 de rang exact 3 (`rng.default_rng(42)`), compară
   valorile singulare custom vs NumPy.

5. **`scripts/verify_reproducibility.py:427`** — duplicat al validării custom vs
   NumPy din `src/validate_svd.py`, folosit în verificarea de reproducibilitate.
   - Atenție: logica e duplicată între cele două fișiere; orice modificare a
     convenției de validare trebuie aplicată în ambele.

### `np.linalg.matrix_rank`

1. **`src/validate_svd.py:58`** — rangul numeric raportat în `results/svd_validation.csv`.
2. **`src/rank_k.py:62`** — rangul numeric tipărit la rularea rank-k.
3. **`scripts/test_synthetic_rank.py:24`** — rangul de referință pe matricea sintetică.

Convenția implicită NumPy: `s > max(m,n) * eps * s[0]` (eps float64 ≈ 2.22e-16).

## Ce NU folosește NumPy (de verificat la revizuire)

- **Pseudo-inversă / least squares**: `src/regression.py:35-51` calculează
  coeficienții direct din `U, s, Vt` (fără `pinv`/`lstsq`).
- **Detectorul SVD de anomalii**: `src/anomaly_detectors.py:15` folosește deja
  `SVD()` custom, nu `np.linalg.svd`.

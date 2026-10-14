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

1. **`src/svd_common.py`** (`compute_svd`) — **unicul punct de intrare** pentru
   SVD în pipeline. Apelează `np.linalg.svd(A, full_matrices=False)` pentru
   `solver="numpy"` sau `custom_svd.SVD` pentru `solver="custom"` și normalizează
   ambele rezultate la formatul comun `U, s, Vt` (s vector, ordonat
   descrescător, rang numeric comun, info de convergență). Scripturile
   `src/rank_k.py`, `src/regression.py`, `src/anomaly_detectors.py` nu mai
   apelează direct `np.linalg.svd`; o fac prin `compute_svd`, cu `--solver
   custom|numpy` în linia de comandă.

2. **`src/validate_svd.py:54`** — referința NumPy pentru validarea brută a
   SVD-ului custom (rămas nemodificat intenționat: etalonul de comparație).
   - Intrare: `data/A.npy` (28380×406, implicit; suprascriibil prin `--A`).
   - `full_matrices=False`; valorile singulare `sn` și `Vtn` comparate cu
     rezultatul custom (eroare de reconstrucție, ortogonalitate, diferența de
     valori singulare).
   - Acest apel **trebuie păstrat** în orice revizuire: este etalonul de comparație.

3. **`scripts/verify_reproducibility.py:427`** — duplicat al validării custom vs
   NumPy din `src/validate_svd.py`, folosit în verificarea de reproducibilitate.
   - Atenție: logica e duplicată între cele două fișiere; orice modificare a
     convenției de validare trebuie aplicată în ambele.

### `np.linalg.matrix_rank`

1. **`src/validate_svd.py:58`** — rangul numeric raportat în `results/svd_validation.csv`.
2. **`scripts/test_synthetic_rank.py`** — înlocuit cu criteriul comun din
   `svd_common` (testul compară acum ambii solveri prin `compute_svd`).
3. ~~`src/rank_k.py:62`~~ — eliminat; rank_k folosește `info["rank"]` din
   `compute_svd` (aceeași convenție NumPy implicit, deci aceleași valori).

Convenția implicită NumPy: `s > max(m,n) * eps * s[0]` (eps float64 ≈ 2.22e-16).
Criteriul comun din `svd_common.compute_svd` coincide cu ea când `rtol`/`atol`
sunt `None`; cu toleranțe explicite rangul poate diferi — vezi docstring-ul
modulului.

## Ce NU folosește NumPy (de verificat la revizuire)

- **Pseudo-inversă / least squares**: `src/regression.py:35-51` calculează
  coeficienții direct din `U, s, Vt` (fără `pinv`/`lstsq`).
- **Detectorul SVD de anomalii**: `src/anomaly_detectors.py:15` folosește deja
  `SVD()` custom, nu `np.linalg.svd`.

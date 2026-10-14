# Organizarea datelor, rezultatelor și figurilor

Reguli de separație, ca fiecare artefact să poată fi asociat neambiguus cu
configurația care l-a produs:

## 1. Date de intrare — `data/` (sursă, nu se suprascrie)

| Fișier | Rol |
|---|---|
| `nyc-rolling-sales.csv` | setul brut de pe Kaggle (nu e versionat, se pune local) |
| `A.npy`, `A_numeric.npy`, `y.npy` | matricele construite de `src/build_matrix.py` |
| `meta.json` | **configurația exactă** a construcției (robust, drop_top_pct, min_price, m, n) |

Dacă `build_matrix.py` rulează cu altă configurație, matricele noi se duc într-un
director nou (`data/<config>/`), nu peste cele existente.

## 2. Rezultate — `results/<config>/`

- `results/baseline/` — snapshot-ul inițial, cu proveniența documentată în
  `results/baseline/MANIFEST.md`. Read-only.
- `results/*.csv` (rădăcina) — working copy, poate fi re-generată; comparația
  oficială se face împotriva lui `results/baseline/`.
- Convenția de nume: `<experiment>[_<varianta>].csv`, unde `<varianta>` e
  `_numeric` pentru matricea 28380×6, `_log` pentru ținta logaritmică.

## 3. Figuri — `figures/<config>/`

- `figures/baseline/` — copia figurilor de la snapshot-ul inițial.
- Scripturile din `scripts/plot_*.py` generează figurile din CSV-uri; numele
  figurii trebuie să indice experimentul (ex. `rank_k_comparison.png`).

## 4. Asocierea artefact → configurație

Fiecare set de rezultate e identificat prin tripletul:

1. **matricea** (A 28380×406 sau A_numeric 28380×6 — vezi `docs/audit-matrice-experimente.md`)
2. **configurația datelor** (`data/meta.json`: robust, drop_top_pct, min_price)
3. **mediul** (versiuni + seed — vezi `docs/mediu-reproductibilitate.md`)

Un experiment e gata când acest triplet e documentat împreună cu artefactele lui.

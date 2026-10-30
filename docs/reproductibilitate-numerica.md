# Reproductibilitatea numerică a SVD-ului custom

Acest document însoțește implementarea din `src/custom_svd.py` și explică
fiecare alegere numerică care afectează reproducibilitatea rezultatelor,
așa cum cere Reviewer 1. Toate constantele menționate sunt cele din cod,
la data rulărilor din `results/solver_comparison.csv` și
`results/rank_k_summary.csv`.

## 1. De ce matricea Gram (AᵀA)

SVD-ul custom calculează valorile singulare ca rădăcini pătrate ale
valorilor proprii ale matricei Gram **B = AᵀA** (n×n), nu prin
bidiagonalizarea lui A. Rațiunea:

- Matricea reală A are 28380×406; B are 406×406 — problema de valori
  proprii e independentă de m și e suficient de mică pentru o iterație QR
  explicită, scrisă de la zero.
- Vectorii singulari dreapta sunt vectorii proprii ai lui B, iar cei stânga
  se obțin apoi din uᵢ = Avᵢ/σᵢ — nicio descompunere suplimentară a lui A
  nu e necesară.

Limitarea este conditioning-ul la pătrat (secțiunea 2), care fixează
implicit pragul de încredere al întregii implementări.

## 2. Conditioning-ul la pătrat și consecința lui

Formarea lui B = AᵀA pătratează numărul de condiționare:
κ₂(B) = κ₂(A)². Eroarea relativă a unei valori proprii λᵢ a lui B este de
ordin eps·κ₂(B), deci eroarea relativă a valorii singulare
σᵢ = √λᵢ este de ordin **√eps · κ₂(A)** — pe float64, σᵢ devine
nedemonstrabil sub ~1e-6·σ₁ pentru matricea noastră.

Observație directă pe date: numpy raportează σ₃₈₈… = 0 (sub pragul
eps·max(m,n)·σ₁ ≈ 2.1e-9), în timp ce solverul custom, prin B, obține pentru
aceleași direcții valori de 1.8e-4 … 1.0e-6 — pur zgomot de rotunjire
ridicat la σ². Concret:

- pe spectrul semnificativ cele două implementări coincid:
  max|σ_custom − σ_numpy| peste primele 387 de valori = **2.35e-8**
  (relative ~7.5e-8 la σ₃₈₇ = 0.31), iar pe primele 50 relative 4.6e-13;
- sub ~1e-6·σ₁ valorile custom nu mai reprezintă semnal, ci podea de zgomot.

Consecință de protocol: orice prag de rang cu sens fizic trebuie să stea
**deasupra** etajului de zgomot, deci rtol ≥ 1e-6 pentru această matrice.
`compute_svd` aplică același prag, o singură dată, după rularea
solverului, identic pentru `custom` și `numpy` — astfel decizia de rang nu
depinde de implementare (M1).

## 3. Criteriile de convergență ale iterației QR

Etapă: `QR_iteration` pe matricea tridiagonală T = QᵀBQ (tridiagonalizare
Householder în prealabil, `Tridiag_Householder`).

- **Deflație**: un element subdiagonal T[k+1,k] este declarat zero când
  |T[k+1,k]| < **TOL = 1e-6** (convenție absolută; valorile proprii ale lui
  B sunt de ordin 10⁴, deci pragul e relativ ~1e-10 față de spectru).
  Blocul trailing se desprinde și se iterează separat ( QR cu shift
  Wilkinson, `_qr_step_tridiag_explicit`).
- **Bucla internă**: maximum **10 iterații QR per bloc**; la epuizare,
  elementul subdiagonal este forțat la zero (deflație forțată) și blocul
  se micșorează — garanție de terminare chiar dacă convergența e lentă.
  **Deflațiile forțate sunt numărate separat** (`info["n_forced_deflations"]`,
  cu `info["max_forced_deflation_residual"]`), ca să nu fie confundate cu
  deflațiile prin convergență.
- **Bucla externă**: maximum **5000 iterații totale** (`max_total_iters`);
  flagul `converged` raportat în `info` spune dacă s-a ajuns la matrice
  diagonală înainte de limită.
- **Certificat de convergență**: după bucle se măsoară
  `eigen_residual = ‖B − V D Vᵀ‖_F / ‖B‖_F` — reziduul efectiv al perechii
  (valori proprii, vectori proprii), în baza originală (nu în cea
  tridiagonală). Pe matricea reală: **0 deflații forțate**,
  `eigen_residual` = 7.1e-12 (A completă, 535 iterații) și 9.6e-12
  (A numerică) — convergență curată; calea de deflație forțată nu se
  activează pe datele noastre, dar e instrumentată și testată (pytest)
  pentru cazurile dificile.
- Acuratețea absolută a valorilor proprii e astfel ~1e-6 · |λ|max, ceea ce
  se propagă în σ ca ~1e-3·σ relativ — suficient pentru praguri de rang ≥1e-6,
  insuficient sub ele (vezi secțiunea 2).

## 4. Toleranțele valorilor singulare

- **TOL = 1e-14, RTOL = 0** — pragul intern cu care `SVD()` este apelat din
  `svd_common.compute_svd`. Rolul lui e strict de siguranță: exclude din
  construcția lui U direcțiile cu σ ≤ 1e-14 (la limita zgomotului de
  mașină, unde împărțirea u = Av/σ ar fi instabilă). Nu participă la decizia
  de rang.
- **cutoff = max(atol, rtol·σ₁)** — pragul real de rang, aplicat post-hoc
  în `compute_svd` pentru ambele solver-e. Cu `rtol = atol = None` se
  folosește convenția NumPy `eps·max(m,n)·σ₁` (documentat în `info`),
  pentru comparabilitate directă cu `np.linalg.matrix_rank`.

## 5. Decizia de rang

`rank = #{i : σᵢ ≥ cutoff}`. Toate experimentele din repo folosesc același
criteriu prin `compute_svd`:

- regresie (M2): rtol = 1e-6 (deasupra etajului de zgomot al metodei Gram);
  pe train 18163×406 → rang 367, pe train+val 22704×406 → rang 376;
- rank-k (M3): tot rtol = 1e-6 — prag uniform cu regresia, deasupra
  etajului de zgomot; pe A completă ambele solver-e dau **387/387**, iar
  convenția NumPy (rtol ≈ 6.3e-12) ar da 393 vs 387: cele 6 direcții în
  plus sunt exact zgomotul de la secțiunea 2. Energia cumulativă la
  k = 5, 10, 20, 50 nu depinde de prag și coincide între solver-e până
  la ~1e-15. Rangul și energia sunt raportate separat, cu toleranța
  folosită, în `results/rank_k_summary.csv`.

## 6. Valorile proprii mici negative

B = AᵀA este teoretic pozitiv semidefinită, dar iterația QR pe float64
produce uneori valori proprii ușor negative (până la −1e-8 pe matricea
reală). Tratament în `SVD`:

1. sortare descrescătoare a valorilor proprii;
2. `val[val < 0] = 0.0` — tăiere la zero, cu numărul exact de valori
   afectate raportat în `info["n_negative_clipped"]` (13 pe A completă);
3. σ = √val, deci nu apare niciodată σ imaginar sau NaN;
4. pragul intern 1e-14 (secțiunea 4) elimină direcțiile cu σ nul din
   construcția lui U; coloanele lipsă din U sunt completate ortonormat
   (`_complete_orthonormal_basis`), ca U să rămână ortogonal în orice caz.

Perechea (tăiere + completare ortonormală) e idempotentă și independentă
de solver: numpy taie prin aceeași regulă de cutoff aplicată post-hoc.

## 7. Ce măsurăm ca să demonstrăm echivalența (M1)

`scripts/compare_solvers.py`, cu același rtol/atol pentru ambele
implementări, raportează în `results/solver_comparison.csv`:

| măsurătoare | full (28380×406) | numeric (28380×6) |
|---|---|---|
| rang custom / numpy (rtol = 1e-6) | 387 / 387 | 6 / 6 |
| max \|Δσ\| pe rangul comun | 2.35e-8 | 1.77e-10 |
| eroare relativă de reconstrucție custom / numpy | 5.19e-11 / 4.12e-14 | 1.16e-11 / 7.86e-14 |
| \|UᵀU − I\| custom / numpy | 1.22e-13 / 4.25e-13 | 2.63e-15 / 1.48e-13 |
| \|VVᵀ − I\| custom / numpy | 8.84e-14 / 5.91e-14 | 1.79e-15 / 1.84e-15 |
| diferența maximă a predicțiilor LS (preț brut) | ~1e-5 USD pe 2.3e7 | — |
| corelația predicțiilor Ridge α=1000 (log) | 1.000000 | — |

Concluzia M1: pe spectrul semnificativ și pe orice predicție de regresie
cele două implementări sunt echivalente; diferențele apar doar în cele
~6 direcții de zgomot de la podea, unde metoda Gram nu are acuratețe prin
construcție — iar pragul comun de rang (rtol = 1e-6, deasupra podelei)
le exclude din toate calculele downstream.

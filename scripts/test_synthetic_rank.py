import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from svd_common import compute_svd, format_svd_info

rng = np.random.default_rng(42)

A = rng.normal(size=(100, 5))

# Facem doua coloane dependente liniar
A[:, 3] = A[:, 0] + 2 * A[:, 1]
A[:, 4] = 3 * A[:, 2]

print("shape:", A.shape)
print("expected rank:", 3)

for solver in ["custom", "numpy"]:
    U, s, Vt, info = compute_svd(A, solver=solver)

    print(format_svd_info(info))
    print(f"{solver:<7} singular values:", s)
    assert Vt.shape == (5, 5) and U.shape == (100, 5), \
        f"forme economice gresite: U{U.shape}, Vt{Vt.shape}"
    assert np.all(np.diff(s) <= 0), "valorile singulare nu sunt sortate descrescator"
    assert np.all(s >= 0), "valori singulare negative necuratate"

print("OK: ambele solvere produc U, s, Vt cu acelasi format")

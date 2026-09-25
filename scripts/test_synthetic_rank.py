import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from custom_svd import SVD

rng = np.random.default_rng(42)

A = rng.normal(size=(100, 5))

# Facem doua coloane dependente liniar
A[:, 3] = A[:, 0] + 2 * A[:, 1]
A[:, 4] = 3 * A[:, 2]

U, S, V = SVD(A)

s_custom = np.diag(S)
rank_custom = np.sum(s_custom > 0)

s_numpy = np.linalg.svd(A, compute_uv=False)
rank_numpy = np.linalg.matrix_rank(A)

print("shape:", A.shape)
print("expected rank:", 3)
print("custom rank:", rank_custom)
print("numpy rank:", rank_numpy)
print("custom singular values:", s_custom)
print("numpy singular values:", s_numpy)
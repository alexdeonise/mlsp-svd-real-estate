"""Pytest ruleaza din radacina repo-ului; sursele sunt in src/."""

import os
import sys

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(
    0,
    os.path.join(
        ROOT,
        "src",
    )
)

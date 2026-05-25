"""Pytest configuration for Harry's test package.

Injects ``src/`` into ``sys.path`` so the in-tree ``stml`` package is
importable from tests without requiring an editable install. Scoped to
Harry's tests (it lives under ``tests/harry/``) so the rest of the project
is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

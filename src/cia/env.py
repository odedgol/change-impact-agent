"""Load settings from a ``.env`` file into the environment. Standard library only.

    KEY=value          # comments and blank lines are skipped
    export KEY="value" # optional export and quotes

A variable already set in the shell wins, so ``OPENROUTER_API_KEY=... make bench-openrouter``
still overrides the file. ``.env`` is git-ignored; ``.env.example`` lists every setting.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, value = m.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()  # an inline comment after an unquoted value
        out[key] = value
    return out


def load_env(path: str | Path = ".env") -> list[str]:
    """Set each variable from ``path`` that the shell has not set. Returns the names it set."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for key, value in parse(p.read_text()).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded

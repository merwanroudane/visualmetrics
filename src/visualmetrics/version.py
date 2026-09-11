"""Single source of truth for the package version."""

from __future__ import annotations

__version__ = "0.1.2"

VERSION_INFO = tuple(int(part) for part in __version__.split(".")[:3])

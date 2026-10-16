"""Compatibility shim: week-1 code imports ``EvidenceTools`` from here.

The implementation moved to ``cia.sources.SyntheticSource``, behind the
``EvidenceSource`` interface.
"""
from __future__ import annotations

from .sources import EvidenceSource, SyntheticSource

EvidenceTools = SyntheticSource

__all__ = ["EvidenceSource", "EvidenceTools", "SyntheticSource"]

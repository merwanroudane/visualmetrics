"""Proof library modules.

Each module exports a single ``PROOF`` object.  Modules are imported lazily by
:class:`~visualmetrics.proofs.registry.ProofRegistry`, never eagerly here, so
adding a proof costs nothing at import time.
"""

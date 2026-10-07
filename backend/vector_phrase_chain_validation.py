"""Direct safety scanning for strict-vector diagnostic artifacts.

The former request-local stage seal cache proved object identity, byte
equality, and source/output replay before later stages could run.  Those are
internal accounting properties, so they no longer participate in product
admission.  Callers keep this small compatibility function because recursive
OCR/consumer safety remains a real strict-path guard.
"""

from __future__ import annotations

from typing import Any

from vector_artifact_safety import vector_artifact_safety_reasons


def request_local_artifact_safety_reasons(
    value: Any,
    *,
    path: str,
) -> list[str]:
    """Scan the current value directly; no identity or digest seal is cached."""
    return vector_artifact_safety_reasons(value, path=path)

"""Pure accessors for the controlled R92 directional step tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DIRECTIONS = ("negative", "positive")


def lookup_step_candidates(
    step_tables: Mapping[str, Any],
    *,
    direction: str,
    previous_shape_key: str,
) -> tuple[dict[str, Any], ...]:
    """Return stable eligible landing candidates for one previous glyph key."""

    if direction not in DIRECTIONS:
        raise ValueError("direction must be negative or positive")
    directions = step_tables.get("directions")
    if not isinstance(directions, Mapping):
        raise ValueError("step-table directions missing")
    table = directions.get(direction)
    if not isinstance(table, Mapping):
        raise ValueError(f"step-table direction missing: {direction}")
    lookup = table.get("lookup")
    if not isinstance(lookup, Mapping):
        raise ValueError(f"step-table lookup missing: {direction}")
    raw_rows = lookup.get(str(previous_shape_key), ())
    if not isinstance(raw_rows, (list, tuple)):
        raise ValueError("step-table lookup row must be an array")
    result: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("step-table candidate must be an object")
        if raw.get("eligible") is not True:
            continue
        entry_id = str(raw.get("entry_id") or "")
        if not entry_id:
            raise ValueError("eligible step candidate has no entry_id")
        step_pt = float(raw.get("step_pt"))
        if step_pt <= 0.0:
            raise ValueError("eligible step candidate has invalid step_pt")
        result.append(
            {
                "entry_id": entry_id,
                "step_pt": step_pt,
                "n": int(raw.get("n", 0)),
                "relative_dispersion": float(raw.get("relative_dispersion", 0.0)),
                "eligible": True,
                "low_confidence": bool(raw.get("low_confidence", False)),
                "next_shape_keys": tuple(
                    str(value) for value in raw.get("next_shape_keys", ())
                ),
            }
        )
    return tuple(sorted(result, key=lambda row: (row["step_pt"], row["entry_id"])))


__all__ = ["DIRECTIONS", "lookup_step_candidates"]

"""Server-owned authority for every product final-consume decision.

Request recognition settings are deliberately absent from this module.  They
may select recognition behavior, but they cannot mint consumer, display, or
release authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import Config


SCHEMA_VERSION = "final_consume_approval_v1"
SERVER_CONSUME_FLAGS = (
    "VECTOR_GLYPH_CONSUME_FINAL",
    "EVIDENCE_SHADOW_CONSUME_FINAL",
)

# Issue 16 owns the closed promotion gate and Issue 17 owns the human Go.  R33
# Issue 03 must not provide an environment-variable bypass for either gate.
_PROMOTION_GATE_OPEN = False


@dataclass(frozen=True)
class FinalConsumeApproval:
    schema_version: str
    policy_valid: bool
    enabled_flags: tuple[str, ...]
    explicit_approval: bool
    promotion_gate_open: bool
    consumer_allowed: bool
    display_allowed: bool
    release_allowed: bool
    reason_code: str

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_valid": self.policy_valid,
            "enabled_flags": list(self.enabled_flags),
            "explicit_approval": self.explicit_approval,
            "promotion_gate_open": self.promotion_gate_open,
            "consumer_allowed": self.consumer_allowed,
            "display_allowed": self.display_allowed,
            "release_allowed": self.release_allowed,
            "reason_code": self.reason_code,
        }


def resolve_final_consume_approval() -> FinalConsumeApproval:
    """Resolve authority from server configuration and the controlled gate."""
    enabled_flags = tuple(
        name for name in SERVER_CONSUME_FLAGS if bool(getattr(Config, name, False))
    )
    explicit_approval = bool(Config.PRODUCTION_FINAL_CONSUME_APPROVED)
    promotion_gate_open = bool(_PROMOTION_GATE_OPEN)

    if not enabled_flags and not explicit_approval:
        policy_valid = True
        reason_code = "final_consume_disabled"
    elif enabled_flags and not explicit_approval:
        policy_valid = False
        reason_code = "final_consume_approval_required"
    elif explicit_approval and not enabled_flags:
        policy_valid = False
        reason_code = "final_consume_approval_orphaned"
    elif not promotion_gate_open:
        policy_valid = False
        reason_code = "final_consume_promotion_gate_closed"
    else:
        policy_valid = True
        reason_code = "final_consume_approved"

    consumer_allowed = bool(
        policy_valid
        and enabled_flags
        and explicit_approval
        and promotion_gate_open
    )
    return FinalConsumeApproval(
        schema_version=SCHEMA_VERSION,
        policy_valid=policy_valid,
        enabled_flags=enabled_flags,
        explicit_approval=explicit_approval,
        promotion_gate_open=promotion_gate_open,
        consumer_allowed=consumer_allowed,
        display_allowed=consumer_allowed,
        release_allowed=False,
        reason_code=reason_code,
    )

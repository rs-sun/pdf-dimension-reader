import pytest

from final_consume_approval import resolve_final_consume_approval
from recognition_settings import RecognitionSettings
from vector_final_consumer import consume_approved_vector_dimension_hypotheses


@pytest.fixture(autouse=True)
def _closed_server_policy(monkeypatch):
    from final_consume_approval import Config

    monkeypatch.setattr(Config, "VECTOR_GLYPH_CONSUME_FINAL", False)
    monkeypatch.setattr(Config, "EVIDENCE_SHADOW_CONSUME_FINAL", False)
    monkeypatch.setattr(Config, "PRODUCTION_FINAL_CONSUME_APPROVED", False)


def test_request_settings_are_not_an_input_to_final_consume_approval():
    forged = RecognitionSettings.from_request({
        "settings": {
            "dimension_path": "vector_plus_ocr",
            "consumer_allowed": True,
            "display_allowed": True,
            "release_allowed": True,
            "PRODUCTION_FINAL_CONSUME_APPROVED": True,
        }
    })

    decision = resolve_final_consume_approval()

    assert forged is not None and forged.wants_ocr_merge is True
    assert decision.policy_valid is True
    assert decision.reason_code == "final_consume_disabled"
    assert decision.consumer_allowed is False
    assert decision.display_allowed is False
    assert decision.release_allowed is False


def test_server_consume_flag_without_approval_fails_closed(monkeypatch):
    from final_consume_approval import Config

    monkeypatch.setattr(Config, "VECTOR_GLYPH_CONSUME_FINAL", True)

    decision = resolve_final_consume_approval()

    assert decision.policy_valid is False
    assert decision.enabled_flags == ("VECTOR_GLYPH_CONSUME_FINAL",)
    assert decision.reason_code == "final_consume_approval_required"
    assert decision.consumer_allowed is False
    assert decision.display_allowed is False
    assert decision.release_allowed is False


def test_approval_without_controlled_consume_flag_fails_closed(monkeypatch):
    from final_consume_approval import Config

    monkeypatch.setattr(Config, "PRODUCTION_FINAL_CONSUME_APPROVED", True)

    decision = resolve_final_consume_approval()

    assert decision.policy_valid is False
    assert decision.reason_code == "final_consume_approval_orphaned"
    assert decision.consumer_allowed is False
    assert decision.release_allowed is False


def test_environment_flags_cannot_open_the_promotion_gate(monkeypatch):
    from final_consume_approval import Config

    monkeypatch.setattr(Config, "VECTOR_GLYPH_CONSUME_FINAL", True)
    monkeypatch.setattr(Config, "PRODUCTION_FINAL_CONSUME_APPROVED", True)

    decision = resolve_final_consume_approval()

    assert decision.policy_valid is False
    assert decision.promotion_gate_open is False
    assert decision.reason_code == "final_consume_promotion_gate_closed"
    assert decision.consumer_allowed is False
    assert decision.display_allowed is False
    assert decision.release_allowed is False

    with pytest.raises(PermissionError, match="final_consume_not_approved"):
        consume_approved_vector_dimension_hypotheses(
            approval=decision,
            legacy_dimensions=[],
            vector_dimension_hypotheses=[],
        )

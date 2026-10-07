from types import SimpleNamespace

import app as app_module


def _ok_check(name):
    return {"name": name, "ok": True, "reason_code": None}


def test_r33_template_readiness_fails_closed_when_path_is_unconfigured(monkeypatch):
    monkeypatch.setattr(app_module.Config, "R33_M1_TEMPLATE_LIBRARY_PATH", "")

    check = app_module._r33_m1_template_runtime_check()

    assert check == {
        "name": "r33_m1_template_runtime",
        "ok": False,
        "reason_code": "template_library_path_unconfigured",
        "template_version": None,
        "content_sha256": None,
    }


def test_r33_template_readiness_uses_pipeline_loader_without_leaking_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module.Config,
        "R33_M1_TEMPLATE_LIBRARY_PATH",
        "/controlled/runtime/templates.json",
    )
    monkeypatch.setattr(
        app_module,
        "load_r33_m1_phrase_runtime",
        lambda path: calls.append(path) or SimpleNamespace(template_identity={
            "schema_version": "r33_m1_template_identity_v1",
            "template_version": "vector_glyph_fixed_template_library_v2",
            "content_sha256": "a" * 64,
        }),
    )

    check = app_module._r33_m1_template_runtime_check()

    assert calls == ["/controlled/runtime/templates.json"]
    assert check == {
        "name": "r33_m1_template_runtime",
        "ok": True,
        "reason_code": None,
        "template_version": "vector_glyph_fixed_template_library_v2",
        "content_sha256": "a" * 64,
    }
    assert all("path" not in key for key in check)


def test_r33_template_readiness_contains_unexpected_loader_failure(monkeypatch):
    monkeypatch.setattr(
        app_module.Config,
        "R33_M1_TEMPLATE_LIBRARY_PATH",
        "/controlled/runtime/templates.json",
    )

    def fail_loader(_path):
        raise RuntimeError("sensitive internal loader detail")

    monkeypatch.setattr(app_module, "load_r33_m1_phrase_runtime", fail_loader)

    check = app_module._r33_m1_template_runtime_check()

    assert check == {
        "name": "r33_m1_template_runtime",
        "ok": False,
        "reason_code": "template_runtime_unavailable",
        "template_version": None,
        "content_sha256": None,
    }
    assert "sensitive" not in str(check)


def test_readyz_includes_template_failure_in_service_readiness(monkeypatch):
    monkeypatch.setattr(app_module.Config, "R33_M1_TEMPLATE_LIBRARY_PATH", "")
    monkeypatch.setattr(app_module, "_feedback_log_check", lambda: _ok_check("feedback_log_writable"))
    monkeypatch.setattr(
        app_module,
        "_final_consume_check",
        lambda: _ok_check("final_consume_requires_explicit_approval"),
    )
    monkeypatch.setattr(app_module, "_strict_runtime_assets_check", lambda: _ok_check("strict_runtime_assets"))
    monkeypatch.setattr(app_module, "_strict_forbidden_runtime_check", lambda: _ok_check("strict_forbidden_runtime"))
    monkeypatch.setattr(app_module, "_eager_model_check", lambda: _ok_check("eager_model_load"))

    payload = app_module._readiness_payload()

    assert payload["status"] == "not_ready"
    assert payload["reason_codes"] == ["template_library_path_unconfigured"]
    template_check = next(
        check for check in payload["checks"]
        if check["name"] == "r33_m1_template_runtime"
    )
    assert template_check["ok"] is False

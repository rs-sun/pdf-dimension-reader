import posixpath
import re
from pathlib import Path

import pytest

import app as app_module


def _client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/index.html",
        "/styles.css",
        "/app.js",
        "/legacy_pdf_transport.js",
        "/review_candidates_v1.js",
        "/app_sidebar/list.js",
        "/libs/pdf.min.js",
        "/libs/pdf.worker.min.js",
        "/libs/pdf-lib.min.js",
        "/libs/jszip.min.js",
    ],
)
def test_product_static_allowlist_serves_required_assets(path):
    response = _client().get(path)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/backend/app.py",
        "/backend/tests/test_api_validation.py",
        "/docs/PRODUCTION_READINESS.md",
        "/frontend_tests/recognition_settings.test.mjs",
        "/.git/HEAD",
        "/.scratch/private/plan.md",
        "/runs/private.json",
        "/test/private.pdf",
        "/AGENTS.md",
        "/CLAUDE.md",
        "/README.md",
        "/build_test_checklist.py",
        "/%2e%2e/backend/app.py",
        "/%252e%252e/backend/app.py",
        "/backend%2fapp.py",
        "/backend%252fapp.py",
        "/backend%5capp.py",
        "/..%5cbackend%5capp.py",
    ],
)
def test_non_product_and_encoded_static_paths_are_not_served(path):
    response = _client().get(path)

    assert response.status_code == 404


def test_allowlisted_symlink_cannot_escape_frontend_root(monkeypatch, tmp_path):
    frontend_root = tmp_path / "frontend"
    frontend_root.mkdir()
    outside = tmp_path / "outside.js"
    outside.write_text("globalThis.leaked = true;", encoding="utf-8")
    (frontend_root / "app.js").symlink_to(outside)
    monkeypatch.setattr(app_module.Config, "FRONTEND_DIR", str(frontend_root))

    response = _client().get("/app.js")

    assert response.status_code == 404
    assert b"leaked" not in response.data


def test_allowlisted_symlink_cannot_alias_blocked_file_inside_frontend_root(
    monkeypatch,
    tmp_path,
):
    frontend_root = tmp_path / "frontend"
    blocked = frontend_root / "backend" / "app.py"
    blocked.parent.mkdir(parents=True)
    blocked.write_text("SECRET = True", encoding="utf-8")
    (frontend_root / "app.js").symlink_to(blocked)
    monkeypatch.setattr(app_module.Config, "FRONTEND_DIR", str(frontend_root))

    response = _client().get("/app.js")

    assert response.status_code == 404
    assert b"SECRET" not in response.data


def test_allowlisted_asset_must_be_a_regular_file(monkeypatch, tmp_path):
    frontend_root = tmp_path / "frontend"
    (frontend_root / "app.js").mkdir(parents=True)
    monkeypatch.setattr(app_module.Config, "FRONTEND_DIR", str(frontend_root))

    response = _client().get("/app.js")

    assert response.status_code == 404


def test_transactional_review_modules_are_in_product_allowlist():
    assert {
        "history_snapshot.js",
        "page_coordinates.js",
        "reanalysis_transaction.js",
        "review_candidates_v1.js",
    }.issubset(app_module._PRODUCT_STATIC_ASSETS)


def test_product_import_graph_is_fully_covered_by_static_allowlist():
    frontend_root = Path(app_module.Config.FRONTEND_DIR)
    dependencies = set()

    html = (frontend_root / "index.html").read_text(encoding="utf-8")
    for reference in re.findall(r'(?:src|href)=["\']([^"\'#]+)', html):
        dependencies.add(reference.split("?", 1)[0].removeprefix("./"))

    for logical_name in app_module._PRODUCT_STATIC_ASSETS:
        if not logical_name.endswith(".js"):
            continue
        path = frontend_root / logical_name
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        references = re.findall(r'\bfrom\s+["\']([^"\']+)', source)
        references += re.findall(r'\bimport\s+["\']([^"\']+)', source)
        for reference in references:
            if not reference.startswith("."):
                continue
            dependencies.add(posixpath.normpath(posixpath.join(
                posixpath.dirname(logical_name),
                reference,
            )))

    assert dependencies <= app_module._PRODUCT_STATIC_ASSETS

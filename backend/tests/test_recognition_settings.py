import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recognition_settings import (
    RecognitionSettings,
    DIMENSION_PATHS,
    OCR_ENGINES,
    VIEW_SEG_BACKENDS,
    GDT_BACKENDS,
)


def test_from_request_none_when_no_settings_key():
    assert RecognitionSettings.from_request({"page": 1}) is None


def test_from_request_none_when_settings_not_dict():
    assert RecognitionSettings.from_request({"settings": "vector_only"}) is None


def test_from_request_parses_valid_dimension_path():
    s = RecognitionSettings.from_request({"settings": {"dimension_path": "vector_plus_ocr"}})
    assert s is not None
    assert s.dimension_path == "vector_plus_ocr"
    assert s.invalid_fields == {}


def test_from_request_falls_back_and_flags_invalid_value():
    s = RecognitionSettings.from_request({"settings": {"dimension_path": "telepathy"}})
    assert s is not None
    assert s.dimension_path == "ocr_only"  # safe legacy fallback, not vector
    assert s.invalid_fields == {"dimension_path": "telepathy"}


def test_whitelist_exact():
    assert DIMENSION_PATHS == ("vector_only", "vector_plus_ocr", "ocr_only")


def test_path_properties():
    vo = RecognitionSettings(dimension_path="vector_only")
    assert vo.is_vector_only is True and vo.wants_ocr_merge is False
    vp = RecognitionSettings(dimension_path="vector_plus_ocr")
    assert vp.is_vector_only is False and vp.wants_ocr_merge is True
    oo = RecognitionSettings(dimension_path="ocr_only")
    assert oo.is_vector_only is False and oo.wants_ocr_merge is False


def test_ocr_engines_whitelist_exact():
    assert OCR_ENGINES == ("rapidocr", "paddle", "yolo_char")


def test_ocr_engine_defaults_none_when_absent():
    s = RecognitionSettings.from_request({"settings": {"dimension_path": "ocr_only"}})
    assert s is not None
    assert s.ocr_engine is None
    assert s.invalid_fields == {}


def test_ocr_engine_parses_valid():
    s = RecognitionSettings.from_request(
        {"settings": {"dimension_path": "vector_plus_ocr", "ocr_engine": "paddle"}}
    )
    assert s is not None
    assert s.ocr_engine == "paddle"


def test_ocr_engine_invalid_falls_back_none_and_flags():
    s = RecognitionSettings.from_request(
        {"settings": {"dimension_path": "ocr_only", "ocr_engine": "magic"}}
    )
    assert s is not None
    assert s.ocr_engine is None
    assert s.invalid_fields == {"ocr_engine": "magic"}


def test_effective_ocr_engine_maps_paddle_to_paddleocr():
    s = RecognitionSettings(dimension_path="ocr_only", ocr_engine="paddle")
    assert s.effective_ocr_engine() == "paddleocr"
    assert RecognitionSettings(ocr_engine=None).effective_ocr_engine() is None
    assert RecognitionSettings(ocr_engine="yolo_char").effective_ocr_engine() is None


def test_view_seg_whitelist_exact():
    assert VIEW_SEG_BACKENDS == ("yolo", "watershed", "minesweeper")


def test_view_seg_defaults_none_when_absent():
    s = RecognitionSettings.from_request({"settings": {"dimension_path": "ocr_only"}})
    assert s is not None
    assert s.view_seg is None


def test_view_seg_parses_valid():
    s = RecognitionSettings.from_request({"settings": {"view_seg": "watershed"}})
    assert s is not None
    assert s.view_seg == "watershed"


def test_view_seg_invalid_falls_back_none_and_flags():
    s = RecognitionSettings.from_request({"settings": {"view_seg": "magic"}})
    assert s is not None
    assert s.view_seg is None
    assert s.invalid_fields == {"view_seg": "magic"}


def test_gdt_whitelist_exact():
    assert GDT_BACKENDS == ("hybrid", "vector", "yolo")


def test_gdt_defaults_none_when_absent():
    s = RecognitionSettings.from_request({"settings": {"dimension_path": "ocr_only"}})
    assert s is not None
    assert s.gdt_backend is None


def test_gdt_parses_valid():
    s = RecognitionSettings.from_request({"settings": {"gdt_backend": "vector"}})
    assert s is not None
    assert s.gdt_backend == "vector"


def test_gdt_invalid_falls_back_none_and_flags():
    s = RecognitionSettings.from_request({"settings": {"gdt_backend": "magic"}})
    assert s is not None
    assert s.gdt_backend is None
    assert s.invalid_fields == {"gdt_backend": "magic"}

from dimension_syntax import (
    DIMENSION_SYNTAX_LABELS,
    classify_dimension_syntax,
)


def test_dimension_syntax_examples_are_frozen():
    assert classify_dimension_syntax("12.34.5") == "incomplete_separator"
    assert classify_dimension_syntax("7X23.125±0.037") == "ok"
    assert classify_dimension_syntax(
        "2X3.5",
        drops=[{"reason": "unrecognized_slot"}],
    ) == "unread_glyph_present"


def test_dimension_syntax_has_exact_stable_label_set():
    assert DIMENSION_SYNTAX_LABELS == {
        "ok",
        "incomplete_separator",
        "unread_glyph_present",
        "invalid_dimension_syntax",
    }


def test_dimension_syntax_does_not_treat_accounting_drop_as_unread_ink():
    assert classify_dimension_syntax(
        "12.5",
        drops=[{"reason": "primitive_partition_invalid"}],
    ) == "ok"


def test_dimension_syntax_uses_generic_label_for_other_invalid_text():
    assert classify_dimension_syntax("R") == "invalid_dimension_syntax"

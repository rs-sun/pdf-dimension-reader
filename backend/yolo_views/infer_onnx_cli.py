#!/usr/bin/env python3
"""Run strict YOLO-A ONNX view inference for one PDF page.

The parent process consumes stdout as JSON.  Runtime diagnostics emitted by the
existing detector are therefore redirected to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import pdfplumber

try:  # Package import in tests; direct import when executed as a script.
    from .infer import detect_views_yolo
except ImportError:  # pragma: no cover - exercised by the subprocess entrypoint
    from infer import detect_views_yolo


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--page", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    if args.model.suffix.lower() != ".onnx":
        parser.error("--model must be an .onnx file")

    with contextlib.redirect_stdout(sys.stderr):
        with pdfplumber.open(str(args.pdf)) as pdf:
            if args.page < 1 or args.page > len(pdf.pages):
                parser.error(f"--page {args.page} is out of range")
            views = detect_views_yolo(
                pdf.pages[args.page - 1],
                model_path=args.model,
            )

    json.dump(
        views,
        sys.stdout,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

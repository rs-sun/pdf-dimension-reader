#!/usr/bin/env python3
"""Subprocess entrypoint for YOLO-view .pt inference.

The main OCR process must not import ultralytics/torch when PaddleOCR is also
loaded.  This CLI keeps the .pt runtime isolated and returns view boxes as JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import pdfplumber

from infer import detect_views_yolo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--page", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args(argv)

    with pdfplumber.open(str(args.pdf)) as pdf:
        if args.page < 1 or args.page > len(pdf.pages):
            raise SystemExit(f"page {args.page} out of range")
        page = pdf.pages[args.page - 1]
        with contextlib.redirect_stdout(sys.stderr):
            views = detect_views_yolo(page, model_path=args.model)

    print(json.dumps(views, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

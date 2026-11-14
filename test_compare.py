"""Smoke tests for image comparison logic (run: .venv/bin/python test_compare.py)."""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from compare_core import run_comparison
from report_builder import build_printable_document, educational_match_percent


def _solid(w: int, h: int, rgb: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return Image.fromarray(arr)


def _noise(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_identical() -> None:
    a = _solid(200, 150, (120, 80, 60))
    result = run_comparison(a, a, 0.5)
    _assert(result.ssim > 0.99, f"expected high SSIM, got {result.ssim}")
    _assert(result.match_percent >= 85, f"expected high match %, got {result.match_percent}")
    _assert(result.pil_a is not None, "outputs missing")
    print("  OK identical images")


def test_different() -> None:
    a = _solid(180, 180, (255, 0, 0))
    b = _solid(180, 180, (0, 0, 255))
    result = run_comparison(a, b, 0.5)
    _assert(result.ssim < 0.99, "different colors should lower SSIM")
    print("  OK different images")


def test_textured_orb() -> None:
    a = _noise(320, 240, 1)
    b = _noise(320, 240, 2)
    result = run_comparison(a, b, 0.5)
    _assert(result.pil_matches is not None, "ORB match image should render for textured input")
    print("  OK textured + ORB")


def test_tiny_image() -> None:
    a = _solid(4, 4, (10, 10, 10))
    b = _solid(4, 4, (20, 20, 20))
    result = run_comparison(a, b, 0.5)
    _assert(result.pil_a is not None and result.pil_b is not None, "tiny images should produce outputs")
    print("  OK tiny images")


def test_report_html() -> None:
    a = _solid(100, 100, (1, 2, 3))
    b = _solid(100, 100, (4, 5, 6))
    doc = build_printable_document(
        run_comparison(a, b, 0.5).report_inputs(),
        expert_name="Dr. Aryan",
        project_author="Aryan",
    )
    _assert("Forensic examination conclusion" in doc and "Forensic expert examiner" in doc, "report sections")
    _assert("Dr. Aryan" in doc, "expert name in report")
    print("  OK printable report")


def test_match_percent_formula() -> None:
    pct = educational_match_percent(1.0, 1.0, 1.0)
    _assert(pct == 100.0, pct)
    print("  OK match percent formula")


def main() -> int:
    print("Running compare_images tests...")
    for t in (
        test_identical,
        test_different,
        test_textured_orb,
        test_tiny_image,
        test_report_html,
        test_match_percent_formula,
    ):
        t()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
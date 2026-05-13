#!/usr/bin/env python3
"""Deprecated compatibility entrypoint for :mod:`evaluation.evaluate_results`."""

try:
    from .evaluate_results import main
except ImportError:
    from evaluate_results import main


if __name__ == "__main__":
    print("[INFO] evaluate_results_enhanced.py is deprecated; forwarding to evaluate_results.py")
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ACC1 wrapper for the shared M1 scalp save-model pipeline.

This keeps ACC2 frozen while letting ACC1 train with its own config/artifact names.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import acc2_scalp_m1_save_model as base  # type: ignore


base.CONFIG = REPO / "configs" / "live_acc1_scalp_m1.yaml"


if __name__ == "__main__":
    base.main()

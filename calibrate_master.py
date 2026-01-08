#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is a wrapper to execute the calibration script from the repository root
when called from within the phase5/project_patch directory.  It simply
imports the top-level calibrate_master module and re-exports its main
function.

Usage:
    python calibrate_master.py --data metrics/calibration_trades.jsonl --out calibration.json

It behaves identically to running the top-level calibrate_master script.
"""

from __future__ import annotations

import os
import sys

# Modify sys.path so that the repository root (one level up) is importable.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from calibrate_master import main as _calibrate_master_main


if __name__ == '__main__':
    _calibrate_master_main()

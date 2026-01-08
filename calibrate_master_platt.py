# -*- coding: utf-8 -*-
"""Calibrate master confidence using Platt scaling.

Reads metrics/calibration_trades.jsonl and fits:

    p_cal = sigmoid(a * logit(p_raw) + b)

Outputs calibration.json with bounded parameters to avoid score saturation.

Usage:
    py -3.12 calibrate_master_platt.py

This script has no third-party dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "metrics" / "calibration_trades.jsonl"
OUT = ROOT / "calibration.json"


def _sigmoid(z: float) -> float:
    # clamp to avoid overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _load_xy(path: Path) -> Tuple[List[float], List[int]]:
    xs: List[float] = []
    ys: List[int] = []
    if not path.exists():
        return xs, ys

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            p_raw = r.get("master_raw")
            if p_raw is None:
                p_raw = r.get("master_conf_before")
            if p_raw is None:
                continue
            try:
                p_raw = float(p_raw)
            except Exception:
                continue

            y = r.get("win")
            if y is None:
                # backward-compat: derive from pnl
                pnl = r.get("realized_pnl")
                if pnl is None:
                    continue
                try:
                    y = 1 if float(pnl) > 0 else 0
                except Exception:
                    continue
            y = 1 if bool(y) else 0

            xs.append(_logit(p_raw))
            ys.append(y)

    return xs, ys


def _fit_platt(xs: List[float], ys: List[int]) -> Tuple[float, float]:
    """Fit a and b by Newton-Raphson on logistic loss."""
    if len(xs) < 50:
        return 1.0, 0.0

    a = 1.0
    b = 0.0
    lam = 1e-2  # L2 damping

    for _ in range(40):
        # gradients
        ga = 0.0
        gb = 0.0
        haa = lam
        hbb = lam
        hab = 0.0

        for x, y in zip(xs, ys):
            z = a * x + b
            p = _sigmoid(z)
            err = p - y
            ga += err * x
            gb += err
            w = p * (1.0 - p)
            haa += w * x * x
            hab += w * x
            hbb += w

        # solve 2x2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-9:
            break

        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det

        step = max(0.1, min(1.0, 1.0 / (1.0 + abs(da) + abs(db))))
        a -= step * da
        b -= step * db

        # early stop
        if abs(da) < 1e-6 and abs(db) < 1e-6:
            break

    # bounds to avoid saturation
    a = max(0.25, min(4.0, float(a)))
    b = max(-2.0, min(2.0, float(b)))
    return a, b


def main() -> None:
    xs, ys = _load_xy(DATA)
    a, b = _fit_platt(xs, ys)

    payload = {
        "type": "logistic",
        "a": a,
        "b": b,
        "trained_on": len(xs),
        "source": "metrics/calibration_trades.jsonl",
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} -> a={a:.4f}, b={b:.4f} (n={len(xs)})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""healthcheck.py

Phase-1 operational healthcheck.

Usage:
  py -3.12 healthcheck.py

This script validates critical configuration and prints actionable diagnostics.
It is safe to run without network access.
"""

from __future__ import annotations

import json
from pathlib import Path

from settings import require_env, env_bool


def main() -> int:
    strict = env_bool("STRICT_ENV", True)
    if strict:
        require_env(["OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"], context="OKX")

    cfg_path = Path("config.json")
    if cfg_path.exists():
        try:
            json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"[HEALTHCHECK] config.json bozuk: {e}")
    else:
        print("[HEALTHCHECK][WARN] config.json bulunamadı (varsayılanlar kullanılacak)")

    metrics_dir = Path("metrics")
    if not metrics_dir.exists():
        print("[HEALTHCHECK][WARN] metrics/ klasörü bulunamadı. Bot ilk çalıştırmada oluşturacaktır.")

    print("[HEALTHCHECK] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

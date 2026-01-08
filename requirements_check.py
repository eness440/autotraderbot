# requirements_check.py
# -*- coding: utf-8 -*-
"""
Basit bağımlılık ve Python sürüm denetleyici.
Eksikler sadece loglanır; auto install yapmaz (istem dışı değişiklik olmasın).
"""

import sys, importlib

REQS = ["ccxt", "pandas", "numpy", "requests"]

def main():
    print(f"[PY] {sys.version}")
    for m in REQS:
        try:
            importlib.import_module(m)
            print(f"[PKG] {m}: OK")
        except Exception as e:
            print(f"[PKG] {m}: MISSING ({e})")

if __name__ == "__main__":
    main()

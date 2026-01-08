# -*- coding: utf-8 -*-
import logging
import logging.handlers
import os
from datetime import datetime
import settings

_LOGGER = None

def get_logger(name: str = "autotrader") -> logging.Logger:
    global _LOGGER
    if _LOGGER:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Konsol
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    # Dosya (dönen)
    log_path = os.path.join(settings.LOG_DIR, f"app_{datetime.now():%Y%m}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    _LOGGER = logger
    return logger

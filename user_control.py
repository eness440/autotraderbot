"""
user_control.py
---------------

This module provides a simple interface for external user control of the
trading bot via messaging platforms such as Telegram or Discord.  It
allows an operator to send commands (e.g. pause/resume trading, set
risk level, blacklist certain symbols) and have the bot respond in real
time.  The implementation here is a skeleton; a production system
should handle authentication, concurrency, and error handling more
robustly.

Two example implementations are provided:

* ``TelegramController``: Uses the Telegram Bot API to poll for new
  messages and respond to commands.  Requires that the ``TELEGRAM_BOT_TOKEN``
  environment variable is set.  Commands are simple text strings such as
  ``/pause``, ``/resume``, ``/risk 0.5`` and ``/blacklist BTC,ETH``.
* ``DiscordController``: Placeholder for a Discord bot implementation.

Usage::

    from user_control import TelegramController
    ctrl = TelegramController(handle_command)
    ctrl.start()

The ``handle_command`` callback will be invoked with the command name
and arguments whenever a recognised command is received.  It is up to
the caller to implement the logic for each command.
"""

from __future__ import annotations

import os
import threading
import time
import logging
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramController:
    """Simple Telegram bot controller using long polling."""

    def __init__(self, command_handler: Callable[[str, str], None], poll_interval: float = 2.0) -> None:
        """
        :param command_handler: callback taking (command, args)
        :param poll_interval: how frequently to poll Telegram API for messages
        """
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        self.poll_interval = poll_interval
        self.offset: Optional[int] = None
        self.command_handler = command_handler
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        base_url = f"https://api.telegram.org/bot{self.token}"
        while not self._stop_event.is_set():
            try:
                params = {"timeout": int(self.poll_interval * 1000), "offset": self.offset}
                resp = requests.get(f"{base_url}/getUpdates", params=params, timeout=self.poll_interval + 1)
                data = resp.json()
                if data.get("ok"):
                    for update in data.get("result", []):
                        self.offset = update.get("update_id", 0) + 1
                        message = update.get("message", {})
                        text = message.get("text", "").strip()
                        if not text:
                            continue
                        if text.startswith("/"):
                            parts = text[1:].split(maxsplit=1)
                            cmd = parts[0].lower()
                            args = parts[1] if len(parts) > 1 else ""
                            try:
                                self.command_handler(cmd, args)
                            except Exception as exc:
                                logger.warning("Command handler error: %s", exc)
                time.sleep(self.poll_interval)
            except Exception as exc:
                logger.warning("Telegram polling error: %s", exc)
                time.sleep(self.poll_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, name="TelegramController", daemon=True)
        self._thread.start()
        logger.info("TelegramController started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("TelegramController stopped")


class DiscordController:
    """Placeholder for a Discord bot controller.  Not implemented."""
    def __init__(self, command_handler: Callable[[str, str], None]) -> None:
        self.command_handler = command_handler
        logger.warning("DiscordController is not implemented yet.")

    def start(self) -> None:
        raise NotImplementedError("DiscordController not implemented")

    def stop(self) -> None:
        pass
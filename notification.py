"""
notification.py
-----------------

This module centralises all notification mechanisms for the trading bot.
When important events occur (for example, a trade is executed, a
kill‑switch activates, or a model finishes training) you may wish to
receive a message on Telegram, Discord or via e‑mail.  The functions
below abstract away the details of each channel and automatically pick
the first configured option.

To enable notifications, set one or more of the following environment
variables before running the bot:

* ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` – send messages via the
  Telegram bot API.
* ``DISCORD_WEBHOOK_URL`` – send messages via a Discord webhook.
* ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASS`` and
  ``SMTP_TO`` – send e‑mails via an SMTP server.

If none of these variables are defined, calls to ``send_notification``
will silently fail and return ``False``.  All network errors are
caught and logged rather than propagating exceptions to the caller.

Example usage::

    from notification import send_notification
    send_notification("📈 New trade executed on BTC/USDT with +12% profit!")

The functions in this module depend only on the standard library and the
``requests`` package.
"""

from __future__ import annotations

import os
import logging
import smtplib
from email.message import EmailMessage
from typing import Optional

try:
    import requests  # type: ignore
except ImportError:
    # Fallback: define a minimal stub so that tests can import this module
    class requests:  # type: ignore
        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("requests is required for HTTP notifications")


logger = logging.getLogger(__name__)


def _send_telegram(message: str) -> bool:
    """Send a plain text message to a Telegram chat via the Bot API.

    Returns True on success, False on failure.  No exception is raised.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)
        return False


def _send_discord(message: str) -> bool:
    """Send a message via a Discord webhook.

    The webhook URL must be stored in the ``DISCORD_WEBHOOK_URL``
    environment variable.
    """
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        return False
    data = {"content": message}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(webhook, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Discord notification failed: %s", e)
        return False


def _send_email(subject: str, body: str) -> bool:
    """Send an e‑mail via SMTP.

    Requires ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASS`` and
    ``SMTP_TO`` environment variables to be defined.  Returns True on
    success and False on failure.  In case of error the exception is logged.
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    recipient = os.getenv("SMTP_TO")
    if not (host and port and user and password and recipient):
        return False
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.warning("Email notification failed: %s", e)
        return False


def send_notification(message: str, subject: Optional[str] = None) -> bool:
    """Send a notification message via the first available channel.

    Tries Telegram, then Discord, then e‑mail.  Returns True if at least
    one channel was successful, otherwise False.  If subject is not
    provided it will be derived from the first 40 characters of the
    message.
    """
    subject = subject or message[:40]
    # Attempt Telegram
    if _send_telegram(message):
        return True
    # Attempt Discord
    if _send_discord(message):
        return True
    # Attempt Email
    if _send_email(subject, message):
        return True
    logger.info("No notification channels configured; message not sent.")
    return False

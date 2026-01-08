# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
import settings
from logger import get_logger
from trade_logger import get_daily_realized_pnl

log = get_logger(__name__)


def is_paused() -> bool:
    return os.path.exists(settings.PAUSE_FLAG)


def request_pause():
    path = settings.PAUSE_FLAG
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    except Exception as e:
        log.warning(f"PAUSE flag klasörü oluşturulamadı: {e}")
    open(path, "w", encoding="utf-8").write("pause")


def clear_pause():
    if os.path.exists(settings.PAUSE_FLAG):
        os.remove(settings.PAUSE_FLAG)


def is_killed() -> bool:
    return os.path.exists(settings.KILL_FLAG)


def request_kill():
    path = settings.KILL_FLAG
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    except Exception as e:
        log.warning(f"KILL flag klasörü oluşturulamadı: {e}")
    open(path, "w", encoding="utf-8").write("kill")


def clear_kill():
    if os.path.exists(settings.KILL_FLAG):
        os.remove(settings.KILL_FLAG)


def pause_loop_sleep(sec: int = 10):
    """
    pause modunda aralıklı bekler, pause kalkınca döner.
    """
    while is_paused():
        # Ayar: durdurma/duraklatma dosyaları runtime klasöründe tutuluyor.
        # Kullanıcı arayüzü de aynı dosya adını kullanıyor. Bu mesajda
        # kullanıcıya doğru yolu belirtelim.
        log.info("PAUSE modunda. Devam için 'runtime/PAUSE' dosyasını silin.")
        time.sleep(sec)


class DailyKillSwitch:
    """
    Günlük zarar limiti aşıldığında kill arası verir.
    daily_limit_pct: negatif yüzde (ör. -0.10 → %10 günlük zarar limiti)
    Gerçekleşmiş (realized) günlük PnL, trade_log.json içinden çekilir.
    """
    def __init__(self, daily_limit_pct: float = -0.10, cooldown_hours: int = 24):
        self.daily_limit_pct = daily_limit_pct
        self.cooldown_hours = cooldown_hours
        self.daily_pnl = 0.0
        self.last_day = datetime.now().day
        self.cooldown_until = None

    def on_new_day(self):
        self.daily_pnl = 0.0
        self.last_day = datetime.now().day

    def add_pnl(self, pnl: float):
        """
        Eski kullanım için bırakıldı. Güncel tasarımda günlük PnL trade_log.json'dan
        okunduğu için bu metod opsiyonel bir ek katkı olarak kalır.
        """
        self.daily_pnl += pnl

    def _refresh_daily_pnl_from_log(self):
        """
        Bugünün gerçekleşmiş PnL değerini trade_log.json dosyasından okur.
        Okuma başarısız olursa mevcut daily_pnl değeri korunur.
        """
        try:
            self.daily_pnl = float(get_daily_realized_pnl())
        except Exception as e:
            log.warning(f"DailyKillSwitch: günlük PnL trade_log'dan okunamadı: {e}")

    def check_switch(self, balance: float) -> bool:
        """
        Kill-switch kontrolü:
        - Günlük gerçekleşmiş PnL, trade_log.json içinden okunur.
        - daily_limit_pct (negatif) ile balance çarpılarak mutlak limit hesaplanır.
        - Günlük PnL bu limitin altına (daha negatif) düşerse cooldown başlatılır.
        """
        now = datetime.now()
        if self.last_day != now.day:
            self.on_new_day()

        # Güncel realized PnL'i log'dan çek
        self._refresh_daily_pnl_from_log()

        if self.cooldown_until and now < self.cooldown_until:
            return True  # kill aktif

        # Günlük zarar limiti: balance referans alınarak
        try:
            limit_abs = float(balance) * float(self.daily_limit_pct)
        except Exception:
            # Hata durumunda limit_abs 0 alınır, kill tetiklenmez
            limit_abs = 0.0

        # daily_limit_pct negatif olduğu için, örneğin:
        # balance=2000, daily_limit_pct=-0.10 → limit_abs=-200
        # daily_pnl <= -200 olduğunda kill tetiklenir.
        if self.daily_pnl <= limit_abs:
            self.cooldown_until = now + timedelta(hours=self.cooldown_hours)
            log.error(
                "DAILY KILL-SWITCH tetiklendi! Günlük PnL=%.4f, limit=%.4f. %s kadar duracak.",
                self.daily_pnl,
                limit_abs,
                self.cooldown_until.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return True

        return False

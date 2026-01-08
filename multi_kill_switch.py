"""
multi_kill_switch.py

Çok kademeli bir kill‑switch sistemi. Günlük gerçekleşmiş PnL oranına
dayalı olarak üç farklı eşik tanımlar ve her eşiğin aşılıp aşılmadığına
göre işlem davranışını belirler. Örneğin:

  - Günlük PnL, bakiyenin -%3'üne eşit veya daha kötüyse → yeni girişler
    azaltılır (yarıya düşürülür).
  - Günlük PnL, bakiyenin -%5'ine eşit veya daha kötüyse → yeni işlem
    açma tamamen durdurulur, mevcut pozisyonlar yönetilir.
  - Günlük PnL, bakiyenin -%7'sine eşit veya daha kötüyse → bot tamamen
    durur, kill‑switch tetiklenir.

Bu sınıf, günlük PnL'i trade_log.json dosyasından okuyarak mevcut
balance ile karşılaştırır. Her seviye için ayrı bir cooldown
süresi tanımlanabilir. Cooldown süresi boyunca ilgili aksiyon geçerli
kalır. Cooldown süreleri saat cinsinden verilir.

Kullanım:
    kill = MultiLevelKillSwitch([
        (-0.03, "reduce"), (-0.05, "halt"), (-0.07, "stop")
    ], {"reduce": 6, "halt": 12, "stop": 24})
    action = kill.check(balance, realized_pnl)
    if action == "reduce":
        # risk azalt
    elif action == "halt":
        # yeni giriş yok
    elif action == "stop":
        # kill
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from trade_logger import get_daily_realized_pnl

class MultiLevelKillSwitch:
    def __init__(self, limits: List[Tuple[float, str]], cooldown_hours: Dict[str, int]):
        """
        Args:
            limits: [(pct, action)] listesi. pct negatif bir sayı olup
                balance * pct = limit. Günlük gerçekleşmiş PnL bu limiti
                aştığında ilgili action tetiklenir.
            cooldown_hours: action→saat. Her action tetiklendiğinde bu
                kadar süre geçerli olur.
        """
        # Limitleri büyükten küçüğe sırala (örneğin -0.03, -0.05, -0.07)
        self.limits = sorted(limits, key=lambda x: float(x[0]))
        self.cooldown_hours = cooldown_hours
        self.cooldown_until: Dict[str, datetime] = {}
        self.last_day = datetime.utcnow().day

    def _refresh_daily_pnl(self) -> float:
        try:
            pnl = float(get_daily_realized_pnl())
            return pnl
        except Exception:
            return 0.0

    def _reset_daily(self):
        self.cooldown_until = {}

    def check(self, balance: float) -> str:
        """
        Günlük PnL'e göre aksiyon döndürür. Mümkün aksiyonlar:
          - "normal": Her şey normal, işlem yapmaya devam
          - "reduce": Yeni girişler risk azaltılarak yapılmalı
          - "halt": Yeni girişler yapılmamalı, açık pozisyonlar
                     yönetilmeye devam eder
          - "stop": Bot tamamen durmalı (kill‑switch)
        """
        now = datetime.utcnow()
        # Yeni günse resetle
        if now.day != self.last_day:
            self._reset_daily()
            self.last_day = now.day
        # Önce etkin cooldown'lar var mı kontrol et
        for action, until in list(self.cooldown_until.items()):
            if until and now < until:
                return action
            else:
                # Süresi dolmuş
                self.cooldown_until.pop(action, None)
        # Günlük PnL'i güncelle
        pnl = self._refresh_daily_pnl()
        for pct, action in self.limits:
            # pct negatif olduğu için balance*pct negatif bir limit
            try:
                limit_abs = float(balance) * float(pct)
            except Exception:
                continue
            # Eğer PnL bu limitin altına düştüyse (daha negatif), aksiyon
            if pnl <= limit_abs:
                # Cooldown başlat
                hours = self.cooldown_hours.get(action, 24)
                self.cooldown_until[action] = now + timedelta(hours=hours)
                return action
        return "normal"
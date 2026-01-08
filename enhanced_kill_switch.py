"""
enhanced_kill_switch.py
-----------------------

Bu modül, mevcut ``multi_kill_switch.py`` sınıfını genişleterek anlık
intraday çöküşler, aşırı slipaj ve büyük volatilite durumları için ek
koruma sağlar. Günlük kill‑switch mekanizması çoğu durumda yeterli
olsa da, bazı senaryolarda dakikalar içinde büyük kayıplar yaşanabilir.

**Özellikler:**

1. **Slipaj Koruması:** Her yeni işlem kapatıldığında, gerçekleşen
   slipaj (gerçek giriş fiyatı ile planlanan fiyat arasındaki fark)
   ölçülür. Bu fark yüzde olarak ``slippage_threshold`` değerini
   aşarsa, ``halt`` veya ``stop`` eylemleri tetiklenir.

2. **Volatilite İzleme:** Kısa süreli (örneğin 5 dakika) fiyat
   değişiminin belirli bir yüzdesi aşması durumunda bot
   yavaşlatılır veya durdurulur. Bu özellik, harici bir fiyat akışına
   abone olunması durumunda kullanılabilir.

3. **Bildirim:** Kill‑switch tetiklendiğinde veya slipaj/volatilite
   eşiği aşıldığında ``notification.send_notification`` fonksiyonu
   aracılığıyla Telegram/Discord/e‑posta mesajı gönderilebilir.

Bu sınıf tek başına çalıştırılmaya uygun değildir; risk yönetimi
bileşenleri tarafından kullanılmalıdır. Slipaj ve volatilite ölçümleri
geliştiricinin mevcut ticaret altyapısından beslenmelidir.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from multi_kill_switch import MultiLevelKillSwitch

try:
    # Bildirim sistemi opsiyoneldir. Mevcut değilse uyarı vermez.
    from notification import send_notification
except Exception:
    def send_notification(message: str, channel: str = "telegram") -> bool:
        return False


class EnhancedKillSwitch(MultiLevelKillSwitch):
    """Genişletilmiş kill‑switch sınıfı.

    Args:
        limits: Günlük PnL yüzdeleri ve aksiyonlar (bkz. MultiLevelKillSwitch).
        cooldown_hours: Her aksiyon için cooldown süresi.
        slippage_threshold: Her işlemde izin verilen maksimum slipaj yüzdesi
            (örneğin 0.02, yani %2). Slipaj bu eşiği aşarsa ``halt``
            veya ``stop`` eylemi tetiklenebilir.
        vol_window: Volatilite hesaplamasında kullanılacak pencere (dakika).
        vol_threshold: Belirtilen pencere içindeki fiyat değişiminin
            aşırı volatil sayılacağı yüzdelik değer (örn. 0.05).
    """

    def __init__(
        self,
        limits: List[Tuple[float, str]],
        cooldown_hours: Dict[str, int],
        slippage_threshold: float = 0.02,
        vol_window: int = 5,
        vol_threshold: float = 0.05,
        ma_window: int = 20,
        ma_drawdown_pct: float = 0.03,
    ):
        super().__init__(limits, cooldown_hours)
        self.slippage_threshold = slippage_threshold
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.ma_window = max(3, int(ma_window))
        self.ma_drawdown_pct = max(0.0, float(ma_drawdown_pct))
        self._balance_history: List[float] = []
        # Basit fiyat geçmişi; gerçek projede fiyat akışı için farklı
        # çözüm kullanılmalıdır.
        self._price_history: List[Tuple[datetime, float]] = []

    def _record_balance(self, balance: float) -> None:
        """Record balance history for MA-based kill switch."""
        try:
            b = float(balance)
        except Exception:
            return
        self._balance_history.append(b)
        # Keep only last ma_window samples
        if len(self._balance_history) > self.ma_window:
            self._balance_history = self._balance_history[-self.ma_window:]

    def _check_balance_ma(self, balance: float) -> Optional[str]:
        """If balance drops below moving average by configured drawdown, trigger halt/stop."""
        if self.ma_drawdown_pct <= 0:
            return None
        if len(self._balance_history) < max(3, self.ma_window // 2):
            return None
        try:
            ma = sum(self._balance_history) / float(len(self._balance_history))
            if ma <= 0:
                return None
            b = float(balance)
        except Exception:
            return None
        # Halt if below MA by drawdown; Stop if below by 2x drawdown
        if b < ma * (1.0 - 2.0 * self.ma_drawdown_pct):
            return 'stop'
        if b < ma * (1.0 - self.ma_drawdown_pct):
            return 'halt'
        return None

    def record_price(self, price: float) -> None:
        """Fiyat geçmişine yeni bir kayıt ekle ve eski kayıtları sil."""
        now = datetime.utcnow()
        self._price_history.append((now, float(price)))
        # Pencereyi koru (vol_window dakika)
        cutoff = now - timedelta(minutes=self.vol_window)
        self._price_history = [p for p in self._price_history if p[0] >= cutoff]

    def check_slippage(self, expected: float, actual: float) -> Optional[str]:
        """
        Slipajı kontrol et. Gerçek giriş fiyatı ``actual`` planlanan fiyat
        ``expected``'den belirli bir yüzde kadar saparsa, uygun eylemi
        döndürür. ``None`` döndürürse normal işleyişe devam edilir.
        """
        try:
            expected = float(expected)
            actual = float(actual)
            if expected == 0:
                return None
            slip_pct = abs(actual - expected) / expected
        except Exception:
            return None
        if slip_pct >= self.slippage_threshold * 2:
            # Büyük slipaj → stop bot
            send_notification(
                f"Slipaj {slip_pct:.2%} ile %100 limitin üzerinde, bot durduruluyor.",
                channel="telegram",
            )
            return "stop"
        elif slip_pct >= self.slippage_threshold:
            send_notification(
                f"Slipaj {slip_pct:.2%} limitin üzerinde, yeni girişler durduruluyor.",
                channel="telegram",
            )
            return "halt"
        return None

    def check_volatility(self) -> Optional[str]:
        """
        Kısa dönem volatiliteyi kontrol et. Kayıtlı fiyat geçmişindeki
        maksimum ve minimum fiyat arasındaki fark, mevcut fiyatın belirli
        bir yüzdesini aşıyorsa aksiyon döndürür.
        """
        if len(self._price_history) < 2:
            return None
        prices = [p[1] for p in self._price_history]
        max_price = max(prices)
        min_price = min(prices)
        # Aşırı volatilite kriteri
        if min_price == 0:
            return None
        range_pct = (max_price - min_price) / min_price
        if range_pct >= self.vol_threshold * 2:
            send_notification(
                f"Volatilite % {range_pct:.2%} ile kritik seviyenin üzerinde, bot durduruluyor.",
                channel="telegram",
            )
            return "stop"
        elif range_pct >= self.vol_threshold:
            send_notification(
                f"Volatilite % {range_pct:.2%} limitin üzerinde, yeni girişler durduruluyor.",
                channel="telegram",
            )
            return "halt"
        return None

    def check(self, balance: float) -> str:
        """
        Günlük PnL sınırları ve intraday kontrolleri birleştirerek eylemi
        belirler. Önce slipaj ve volatilite kontrolü yapılır, ardından
        temel kill‑switch mantığı çalıştırılır.
        """
        # Record balance for MA-based kill switch
        self._record_balance(balance)
        ma_action = self._check_balance_ma(balance)
        if ma_action:
            send_notification(f"Kill-switch (MA) tetiklendi: {ma_action}", channel="telegram")
            return ma_action

        # Öncelik slipaj ve volatilite kontrollerinde
        vol_action = self.check_volatility()
        if vol_action:
            return vol_action
        # Günlük PnL kontrolü
        action = super().check(balance)
        if action != "normal":
            # Bildirim gönder
            send_notification(f"Kill‑switch tetiklendi: {action}", channel="telegram")
        return action
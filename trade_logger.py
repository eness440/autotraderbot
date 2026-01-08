# -*- coding: utf-8 -*-
"""
trade_logger.py
----------------

Bu modül, açık ve kapanan işlemlerin detaylarını tutmak ve kalıcı olarak
``trade_log.json`` dosyasına yazmak için yardımcı fonksiyonlar sağlar.

İlgili her trade için hem giriş (input) hem de çıkış (output) bilgilerini
kaydetmek, daha sonra bilimsel kalibrasyon ve modelleme adımlarında
kullanılacak bir veri seti oluşturmak açısından kritiktir. Bu modül
eşzamanlı işlemlere karşı basit bir bellek içi sözlük ile açık pozisyonları
takip eder ve pozisyon kapandığında gerekli metrikleri hesaplar.

Trade giriş kayıtlarında şu alanlar bulunur:
    - timestamp_open: ISO8601 formatında giriş zamanı (UTC)
    - symbol: işlem yapılan sembol (ör. BTC/USDT)
    - side: 'long' veya 'short'
    - entry_price: pozisyonun açıldığı fiyat
    - size: pozisyon büyüklüğü (coin adedi)
    - ai_score: ChatGPT/DeepSeek çıkışlarının kalibre edilmiş skoru
    - tech_score: teknik analiz skorunun değeri
    - sent_score: sentiment skorunun değeri
    - master_confidence: birleşik güven skoru (0..1 arası)
    - leverage: kullanılan kaldıraç değeri
    - atr: işlem anındaki ATR değeri
    - fgi: Fear & Greed Index (0..100)
    - adx: ADX göstergesinin değeri
    - rsi: RSI göstergesinin değeri
    - ema_fast / ema_slow: EMA kısa ve uzun değerleri

Trade kapanış kayıtlarında ek olarak şu alanlar eklenir:
    - timestamp_close: kapanış zamanı (ISO8601 UTC)
    - exit_price: pozisyonun kapandığı (veya TP/SL emrinin vurduğu) fiyat
    - pnl_abs: USDT cinsinden gerçekleşen kar/zarar
    - pnl_pct: pozisyon değerine göre yüzde kar/zarar
    - max_drawdown_pct: pozisyon açıkken görülen en kötü düşüş yüzdesi (tahmini)
    - holding_time_minutes: pozisyonun açık kaldığı süre (dakika)

Kayıtlar ``trade_log.json`` dosyasına JSON formatında satır satır eklenir.
Dosya, önceki kayıtları bozmadan yeni kayıtlar ekleyecek şekilde bir liste
formatında tutulur. Aynı sembol için birden fazla işlem yapılmadığı
varsayımıyla (``main_bot_async`` içinde ``open_trades`` kontrolü var) bu
modül sembolü bir anahtar olarak kullanır. Eğer aynı sembol için birden
fazla işlem açılması durumunda, ``symbol`` anahtarının yanı sıra ``timestamp_open``
değeri ayırt edici olacaktır.

Bu modül bağımsız çalışacak şekilde tasarlanmıştır ve ``safe_submit_entry_plan``
ve ``safe_submit_exit_plan`` fonksiyonları tarafından kullanılır. Asenkron
çalışma modelinde olmasına rağmen, file IO işlemleri basit ve bloklayıcıdır.
Gerektiğinde geliştirilerek lock mekanizması eklenebilir.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, Optional

# Circuit breaker integration: update circuit breaker state on trade close
try:
    from .circuit_breaker import update_outcome as _circuit_update_outcome  # type: ignore
except Exception:
    def _circuit_update_outcome(pnl_frac: float) -> None:  # type: ignore
        return

# Sinyal veri setindeki kayıtların etiketlerini güncellemek için
try:
    from .signal_logger import update_label as _update_signal_label
except Exception:
    # Eğer modül yoksa dummy fonksiyon
    def _update_signal_label(signal_id: str, label: int) -> None:
        return

# Trade log dosyasının yolu. Proje kök dizininde tutulur.
ROOT_DIR = Path(__file__).resolve().parent
TRADE_LOG_FILE = ROOT_DIR / "trade_log.json"
# AI tahmin logu (performans etiketi için)
AI_PRED_FILE = ROOT_DIR / "metrics" / "ai_predictions.json"

# Bellekte açık pozisyonların izini sürmek için sözlük. Anahtar: sembol.
_open_trades: Dict[str, Dict[str, any]] = {}

# LOGGING SWITCH
# ---------------
# Ticaret kayıtlarını dosyaya yazma davranışı ``ENABLE_TRADE_LOG`` adlı
# ortam değişkeni veya .env ayarı üzerinden kontrol edilir.  Kapanış
# doğrulaması düzeltildiği için varsayılan değer True olarak set edildi.
# Bu sayede trade_log.json güncellenir ve günlük gerçekleşen PnL
# hesaplamaları, kill-switch ve performans raporları anlamlı hale gelir.
try:
    # settings.env_bool anahtarını import etmeye çalış.  Bu helper
    # .env dosyasından veya çevre değişkenlerinden bool okur.
    from .settings import env_bool  # type: ignore
    # ENABLE_TRADE_LOG çevre değişkeni '0', 'false' vb. ise loglama kapanır.
    # Varsayılan: True (loglama açık)
    DO_LOG_TRADES: bool = env_bool("ENABLE_TRADE_LOG", True)
except Exception:
    # settings import edilemezse veya env yardımcıları bulunamazsa, loglama açık
    DO_LOG_TRADES: bool = True


def _read_trade_log() -> list:
    """trade_log.json dosyasından mevcut kayıtları okur."""
    if not TRADE_LOG_FILE.exists():
        return []
    try:
        text = TRADE_LOG_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return []
        obj = json.loads(text)
        # Hem eski list hem de {'rows': [...]} formatını destekle
        if isinstance(obj, dict):
            return obj.get("rows", [])
        elif isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []


def _write_trade_log(rows: list) -> None:
    """
    trade_log.json dosyasına kayıt listesi yazar.  Yazım sırasında
    eşzamanlı erişimlerden kaynaklanan bozulmaları önlemek için
    atomik yazım helper'ı kullanır.  Loglama kapalıysa (ENABLE_TRADE_LOG
    çevre değişkeni false ise) hiçbir şey yazmaz.
    """
    # Eğer loglama kapalıysa dosyaya yazma
    if not DO_LOG_TRADES:
        return
    try:
        from .atomic_io import safe_write_json as _safe_write_json  # type: ignore
    except Exception:
        def _safe_write_json(path, data):
            try:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass
    try:
        # We persist as a simple list; if future schema changes require
        # wrapping under a {'rows': [...]} envelope, adapt here.
        _safe_write_json(TRADE_LOG_FILE, rows)
    except Exception:
        pass


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _tag_ai_prediction_with_trade(record: Dict[str, any]) -> None:
    """
    Kapanan trade'e karşılık gelen son AI tahmin kaydını (metrics/ai_predictions.json)
    outcome/pnl ve sinyal etiketleriyle günceller.

    Etiketler:
      - outcome: 'win' | 'loss' | 'flat'
      - pnl: realize edilmiş getiri (pnl_pct / 100, yani oransal)
      - direction_ok: True/False
      - ai_effective / tech_effective / sent_effective: bool
    """
    try:
        if not AI_PRED_FILE.exists():
            return
        txt = AI_PRED_FILE.read_text(encoding="utf-8").strip()
        if not txt:
            return
        data = json.loads(txt)
        if not isinstance(data, list) or not data:
            return
    except Exception:
        return

    sym = record.get("symbol")
    if not sym:
        return

    pnl_abs = _safe_float(record.get("pnl_abs"))
    pnl_pct = _safe_float(record.get("pnl_pct"))

    if pnl_abs is None and pnl_pct is None:
        return

    # Oransal getiri (pnl_pct zaten % cinsinden)
    pnl_frac = None
    if pnl_pct is not None:
        pnl_frac = pnl_pct / 100.0
    else:
        entry_price = _safe_float(record.get("entry_price"))
        size = _safe_float(record.get("size"))
        if entry_price and size and pnl_abs is not None:
            notional = entry_price * size
            if notional:
                pnl_frac = pnl_abs / notional

    direction_ok = None
    if pnl_abs is not None:
        if abs(pnl_abs) < 1e-12:
            direction_ok = False
        else:
            direction_ok = pnl_abs > 0.0

    # Sinyal skorları
    ai_score = _safe_float(record.get("ai_score"))
    tech_score = _safe_float(record.get("tech_score"))
    sent_score = _safe_float(record.get("sent_score"))
    thr = 0.55

    ai_eff = bool(direction_ok and ai_score is not None and ai_score >= thr)
    tech_eff = bool(direction_ok and tech_score is not None and tech_score >= thr)
    sent_eff = bool(direction_ok and sent_score is not None and sent_score >= thr)

    # Bu trade'e en yakın AI tahmini: aynı sembol için outcome/pnl henüz set edilmemiş son kayıt
    idx_to_update = None
    for idx in range(len(data) - 1, -1, -1):
        row = data[idx]
        if str(row.get("symbol")) != sym:
            continue
        # outcome veya pnl zaten set edilmişse bu kaydı atla
        if row.get("outcome") is not None or row.get("pnl") is not None:
            continue
        idx_to_update = idx
        break

    if idx_to_update is None:
        return

    row = data[idx_to_update]

    # outcome etiketi
    if pnl_abs is None or abs(pnl_abs) < 1e-12:
        outcome_str = "flat"
    elif pnl_abs > 0:
        outcome_str = "win"
    else:
        outcome_str = "loss"

    row["outcome"] = outcome_str
    if pnl_frac is not None:
        row["pnl"] = float(pnl_frac)
    row["direction_ok"] = bool(direction_ok)

    # Sinyal bazlı etiketler
    row["ai_effective"] = ai_eff
    row["tech_effective"] = tech_eff
    row["sent_effective"] = sent_eff

    if ai_score is not None:
        row["ai_score"] = ai_score
    if tech_score is not None:
        row["tech_score"] = tech_score
    if sent_score is not None:
        row["sent_score"] = sent_score

    try:
        AI_PRED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Tahmin dosyasını güncelleyemezsek trade loglamayı bozmamak için sessiz geç
        pass


# ---------------------------------------------------------------------------
# Kalibrasyon Trades Logger
# ---------------------------------------------------------------------------

# Kalibrasyon dosyasının yolu (metrics klasörü altında). JSON Lines formatında
# tutulur; her satır ayrı bir trade kaydıdır.
METRICS_DIR = ROOT_DIR / "metrics"
CALIB_TRADES_FILE = METRICS_DIR / "calibration_trades.jsonl"


def _ensure_metrics_dir() -> None:
    """metrics klasörünün varlığını garanti eder."""
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _write_calibration_trade(record: Dict[str, any]) -> None:
    """
    Kapanan trade'den kalibrasyon için gerekli alanları ``calibration_trades.jsonl``
    dosyasına yazar. Dosya JSON Lines formatındadır. Her satır bir trade
    kaydıdır ve sahadaki başarı/başarısızlık oranlarını kalibre etmek için
    kullanılır.

    Yazılan alanlar:
        - timestamp: açılış zamanı (timestamp_open)
        - symbol: işlem sembolü
        - side: long/short
        - master_conf_before: pozisyon açılırkenki master confidence
        - ai_score, tech_score, sent_score, rl_score
        - leverage, wallet_allocation_percent, risk_usd
        - pnl_abs (USD), pnl_pct (%), r_multiple, max_drawdown_pct
        - label: 1=kazan, 0=kaybet
    """
    if not record:
        return
    # Gerekli alanları çıkar
    try:
        ts = record.get("timestamp_open")
        symbol = record.get("symbol")
        side = record.get("side")
        master_conf = record.get("master_confidence")
        ai = record.get("ai_score")
        tech = record.get("tech_score")
        sent = record.get("sent_score")
        rl = record.get("rl_score")
        lev = record.get("leverage")
        alloc = record.get("wallet_allocation_percent")
        risk_val = record.get("risk_usd")
        pnl_abs = record.get("pnl_abs")
        pnl_pct = record.get("pnl_pct")
        r_multiple = record.get("r_multiple")
        dd_pct = record.get("max_drawdown_pct")
        label = None
        try:
            # Label: pnl_pct > 0 -> 1, aksi halde 0
            if pnl_pct is not None:
                label = 1 if float(pnl_pct) > 0.0 else 0
        except Exception:
            label = None
        # Hesaplanan r_multiple: pnl_abs / risk_usd
        if r_multiple is None:
            try:
                if risk_val is not None and float(risk_val) > 0 and pnl_abs is not None:
                    r_mult = float(pnl_abs) / float(risk_val)
                    # write back to record for completeness
                    record["r_multiple"] = r_mult
                    r_multiple = r_mult
            except Exception:
                r_multiple = None
        # r_multiple negative values allowed
        # Güncel max drawdown zaten record'a yazıldı
        # Build calibration entry
        entry = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "master_conf_before": master_conf,
            "raw_score": record.get("raw_score"),
            "vol_category": record.get("vol_category"),
            "provider_flags": record.get("provider_flags"),
            "ai_score": ai,
            "tech_score": tech,
            "sent_score": sent,
            "rl_score": rl,
            "leverage": lev,
            "wallet_allocation_percent": alloc,
            "risk_usd": risk_val,
            "pnl_usd": pnl_abs,
            "pnl_pct": pnl_pct,
            "r_multiple": r_multiple,
            "max_drawdown_pct": dd_pct,
            "label": label,
        }
    except Exception:
        return
    # Dosyaya ekle
    try:
        _ensure_metrics_dir()
        # JSONL append (lock'lu)
        from atomic_io import safe_append_jsonl
        safe_append_jsonl(CALIB_TRADES_FILE, entry)
        # Otomatik kalibrasyon: yeterli kayıt varsa calibrate_confidence.py'yi çağır
        try:
            # Sadece satır sayısını saymak için dosyayı tekrar açma
            line_count = 0
            with CALIB_TRADES_FILE.open("r", encoding="utf-8") as fcount:
                for _ in fcount:
                    line_count += 1
            # Header yok; doğrudan satır sayısı kontrolü
            if line_count >= 50:
                from .calibrate_confidence import calibrate_confidence as _calibrate_confidence
                root = ROOT_DIR
                data_path = root / "metrics" / "calibration_trades.jsonl"
                calibration_path = root / "calibration.json"
                weights_path = root / "logistic_weights.json"
                schedule_path = root / "risk_schedule.json"
                _calibrate_confidence(data_path, calibration_path, weights_path, schedule_path)
        except Exception:
            pass
    except Exception:
        pass


def log_trade_open(
    symbol: str,
    side: str,
    entry_price: float,
    size: float,
    ai_score: Optional[float],
    tech_score: Optional[float],
    sent_score: Optional[float],
    master_confidence: float,
    leverage: int,
    atr: Optional[float] = None,
    fgi: Optional[float] = None,
    adx: Optional[float] = None,
    rsi: Optional[float] = None,
    ema_fast: Optional[float] = None,
    ema_slow: Optional[float] = None,
    # Yeni parametreler: RL skoru, cüzdan kullanım yüzdesi ve risk (USD)
    rl_score: Optional[float] = None,
    wallet_allocation_percent: Optional[float] = None,
    risk_usd: Optional[float] = None,
    tf: Optional[str] = None,
    base_decision: Optional[str] = None,
    timestamp_open: Optional[str] = None,
    session_name: Optional[str] = None,
    regime: Optional[str] = None,
    # Ek kalibrasyon parametreleri: ham skor, sağlayıcı durumları ve volatilite kategorisi
    raw_score: Optional[float] = None,
    provider_flags: Optional[dict] = None,
    vol_category: Optional[str] = None,
    **extra_fields,
) -> None:
    """
    Yeni bir işlem açıldığında giriş bilgilerini kaydeder. Aynı sembol için
    açık bir kayıt varsa üzerine yazılır.

    Parametreler:
        symbol: İşlem yapılan sembol (ör. 'BTC/USDT')
        side: 'long' veya 'short'
        entry_price: Pozisyonun açıldığı fiyat
        size: Pozisyon büyüklüğü (coin adedi)
        ai_score: AI (ChatGPT/DeepSeek) skorunun değeri
        tech_score: Teknik skor
        sent_score: Sentiment skor
        master_confidence: Birleşik güven skoru (0..1)
        leverage: Kullanılan kaldıraç
        atr: ATR değeri
        fgi: Fear & Greed Index (0..100)
        adx: ADX göstergesi
        rsi: RSI göstergesi
        ema_fast/ema_slow: EMA değerleri
        timestamp_open: ISO8601 tarih stringi; verilmezse UTC şimdi kullanılır
    """
    if timestamp_open is None:
        timestamp_open = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    try:
        entry_price_f = float(entry_price)
        size_f = float(size)
    except Exception:
        # Geçersiz sayı varsa loglama yapma
        return

    # Eğer risk_usd parametresi verilmediyse, ATR veya fallback ile tahmini hesapla.
    # Bu değer, stop-loss mesafesine göre pozisyon büyüklüğü ile çarpılarak elde edilir.
    computed_risk = None
    if risk_usd is None:
        try:
            # risk_manager'dan compute_stop_loss fonksiyonunu içe aktar
            from .risk_manager import compute_stop_loss
            # ATR kullanarak veya fallback ile SL fiyatını hesapla
            sl_price = compute_stop_loss(
                side=side,
                entry_price=entry_price_f,
                atr=float(atr) if atr is not None else None,
                # atr_mult varsayılan 1.0, percent_fallback 0.01
                atr_mult=1.0,
                tick_size=None,
                percent_fallback=0.01,
                last_price=None,
            )
            risk_per_unit = abs(entry_price_f - float(sl_price))
            computed_risk = risk_per_unit * size_f
        except Exception:
            computed_risk = None
    # atanan risk_usd parametresini tercih et, yoksa hesaplananı kullan
    final_risk_usd = risk_usd if risk_usd is not None else computed_risk
    record = {
        "timestamp_open": timestamp_open,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price_f,
        "size": size_f,
        "ai_score": float(ai_score) if ai_score is not None else None,
        "tech_score": float(tech_score) if tech_score is not None else None,
        "sent_score": float(sent_score) if sent_score is not None else None,
        "master_confidence": float(master_confidence),
        "leverage": int(leverage),
        "atr": float(atr) if atr is not None else None,
        "fgi": float(fgi) if fgi is not None else None,
        "adx": float(adx) if adx is not None else None,
        "rsi": float(rsi) if rsi is not None else None,
        "ema_fast": float(ema_fast) if ema_fast is not None else None,
        "ema_slow": float(ema_slow) if ema_slow is not None else None,
        # RL skoru, cüzdan kullanım yüzdesi ve risk değeri
        "rl_score": float(rl_score) if rl_score is not None else None,
        "wallet_allocation_percent": float(wallet_allocation_percent) if wallet_allocation_percent is not None else None,
        "risk_usd": float(final_risk_usd) if final_risk_usd is not None else None,
        "tf": str(tf) if tf is not None else None,
        "base_decision": str(base_decision) if base_decision is not None else None,
        # output tarafı boş bırakılır
        "timestamp_close": None,
        "exit_price": None,
        "pnl_abs": None,
        "pnl_pct": None,
        "max_drawdown_pct": None,
        "holding_time_minutes": None,
        "r_multiple": None,
        "label": None,
        # Ek alanlar: seans ve piyasa rejimi
        "session": str(session_name) if session_name is not None else None,
        "regime": str(regime) if regime is not None else None,
        # Kalibrasyon veri seti için ham skor, sağlayıcı durumları ve vol kategorisi
        "raw_score": float(raw_score) if raw_score is not None else None,
        "provider_flags": provider_flags if provider_flags is not None else None,
        "vol_category": str(vol_category) if vol_category is not None else None,
    }

    # Merge any additional keyword arguments into the record.  This allows
    # callers to persist arbitrary extra fields (e.g. ai_components,
    # tech_signals_detail, final_weights, risk_veto) without modifying
    # the function signature for each new requirement.
    try:
        for k, v in extra_fields.items():
            record[k] = v
    except Exception:
        pass
    # Bellekte sakla
    _open_trades[symbol] = record


def log_trade_close(symbol: str, exit_price: float, fee: Optional[float] = None, slippage_pct: Optional[float] = None) -> None:
    """
    Mevcut açık trade'i kapatarak trade_log.json'a yazar.  Çıkış fiyatı,
    işlem ücreti (``fee``) ve slipaj yüzdesi (``slippage_pct``) dahil
    edilerek PnL hesaplanır ve kayıt güncellenir.

    Args:
        symbol: kapanan pozisyonun sembolü (ör. "BTC/USDT").
        exit_price: borsadan alınan ham çıkış fiyatı.
        fee: isteğe bağlı toplam işlem ücreti (USDT cinsinden).  Eğer
            sağlanmışsa, PnL hesaplamasında doğrudan düşülür.
        slippage_pct: isteğe bağlı bağıl slipaj yüzdesi (0.0–1.0 arası).  
            Long pozisyonlarda çıkış fiyatından düşülür, short
            pozisyonlarda eklenir.  Eğer None ise slipaj uygulanmaz.

    Notlar:
        Bu fonksiyon, ``_open_trades`` sözlüğündeki kayıtları çıkarır ve
        ``trade_log.json`` dosyasına kalıcı olarak ekler.  Aynı sembol
        için açık bir trade yoksa fonksiyon sessizce döner.
    """
    # Eğer sembol için açık trade yoksa, sessizce dön
    if symbol not in _open_trades:
        return
    record = _open_trades.pop(symbol)
    # Geçerli çıkış fiyatını float'a çevir; dönüş hatasında kayıt yapma
    try:
        exit_price_f = float(exit_price)
    except Exception:
        return
    # Zaman damgası ve ham çıkış fiyatı
    now_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    record["timestamp_close"] = now_iso
    record["exit_price"] = exit_price_f
    # Kaydet işlem ücreti ve slipaj yüzdesi
    record["fee"] = float(fee) if fee is not None else None
    record["slippage_pct"] = float(slippage_pct) if slippage_pct is not None else None

    entry_price = record.get("entry_price")
    size = record.get("size")
    side = record.get("side")
    # Slipaj uygulanmış gerçek çıkış fiyatını hesapla
    net_exit = exit_price_f
    if slippage_pct is not None and entry_price and size:
        try:
            slip = float(slippage_pct)
            # Long: slipaj çıkış fiyatını azaltır; Short: çıkış fiyatını artırır
            if side == "long":
                net_exit = exit_price_f * (1.0 - slip)
            else:
                net_exit = exit_price_f * (1.0 + slip)
        except Exception:
            net_exit = exit_price_f
    # Kar/zarar hesabı
    if entry_price and size:
        try:
            if side == "long":
                pnl_abs = (net_exit - entry_price) * size
            else:
                pnl_abs = (entry_price - net_exit) * size
            # Ücreti düş (her zaman maliyet olarak kabul edilir)
            if fee is not None:
                try:
                    pnl_abs -= float(fee)
                except Exception:
                    pass
            # Pozisyonun notional değerine göre yüzde hesapla
            position_value = entry_price * size
            pnl_pct = (pnl_abs / position_value) * 100.0 if position_value else None
            record["pnl_abs"] = round(pnl_abs, 8) if pnl_abs is not None else None
            record["pnl_pct"] = round(pnl_pct, 6) if pnl_pct is not None else None
        except Exception:
            record["pnl_abs"] = None
            record["pnl_pct"] = None
    # Basit max drawdown tahmini: ATR oranı ve gerçekleşen PnL’den negatif olanı seç
    atr = record.get("atr")
    dd_atr_pct = None
    if atr and entry_price:
        try:
            dd_atr_pct = - (float(atr) / entry_price) * 100.0
        except Exception:
            dd_atr_pct = None
    # PnL yüzde hesaplandıysa onu da kullan
    dd_pnl_pct = None
    try:
        dd_pnl_pct = float(record.get("pnl_pct")) if record.get("pnl_pct") is not None else None
        if dd_pnl_pct is not None:
            dd_pnl_pct = min(0.0, dd_pnl_pct)
    except Exception:
        dd_pnl_pct = None

    dd_candidates = [v for v in [dd_atr_pct, dd_pnl_pct] if v is not None]
    if dd_candidates:
        record["max_drawdown_pct"] = round(min(dd_candidates), 6)
    else:
        record["max_drawdown_pct"] = None

    # Pozisyon süresi
    try:
        ts_open_str = record.get("timestamp_open")
        ts_open = datetime.fromisoformat(ts_open_str) if ts_open_str else None
        ts_close = datetime.fromisoformat(now_iso)
        if ts_open and ts_close:
            delta = ts_close - ts_open
            record["holding_time_minutes"] = round(delta.total_seconds() / 60.0, 3)
    except Exception:
        record["holding_time_minutes"] = None

    # --- Sinyal etiket güncelleme ---
    # trade open timestamp + symbol → signal_id
    try:
        ts_open_str = record.get("timestamp_open")
        if ts_open_str:
            # Use the globally imported datetime instead of re-importing to
            # avoid shadowing the module.  Remove tzinfo and microseconds for
            # stable signal ID construction.
            try:
                ts_open_dt = datetime.fromisoformat(ts_open_str)
            except Exception:
                ts_open_dt = None
            if ts_open_dt is not None:
                ts_no_micro = ts_open_dt.replace(microsecond=0, tzinfo=None)
                signal_id = f"{ts_no_micro.isoformat()}_{symbol}"
                # Etiket: pnl_pct > 0 ise 1, aksi halde 0
                pnl_pct_val = record.get("pnl_pct")
                label: Optional[int] = None
                if pnl_pct_val is not None:
                    try:
                        label = 1 if float(pnl_pct_val) > 0.0 else 0
                    except Exception:
                        label = None
                if label is not None:
                    _update_signal_label(signal_id, label)
    except Exception:
        # update hatası sessizce yutulur
        pass

    # PnL ve risk bilgisine göre r_multiple ve label hesapla
    try:
        risk_val = record.get("risk_usd")
        pnl_abs_val = record.get("pnl_abs")
        if risk_val is not None and pnl_abs_val is not None:
            try:
                risk_float = float(risk_val)
                pnl_float = float(pnl_abs_val)
                if risk_float > 0:
                    record["r_multiple"] = round(pnl_float / risk_float, 6)
            except Exception:
                pass
        # Label: pnl_pct > 0 -> 1; pnl_pct <= 0 -> 0
        pnl_pct_val = record.get("pnl_pct")
        if pnl_pct_val is not None:
            try:
                record["label"] = 1 if float(pnl_pct_val) > 0.0 else 0
            except Exception:
                record["label"] = None
    except Exception:
        pass

    # Eğer loglama devre dışıysa, kayıt dosyalarını güncelleme ve
    # kalibrasyon/AI güncelleme işlemlerini atla. Yalnızca bellek içi
    # _open_trades sözlüğündeki girdiyi kaldır ve çık.
    if not DO_LOG_TRADES:
        return

    # --- Kalibrasyon verisi güncelle ---
    try:
        _write_calibration_trade(record)
    except Exception:
        # calibration kaydı yazılamazsa sessizce yut
        pass

    # trade_log dosyasını güncelle
    rows = _read_trade_log()
    rows.append(record)
    _write_trade_log(rows)

    # Circuit breaker: update with realised PnL fraction
    try:
        pnl_pct_str = record.get("pnl_pct")
        if pnl_pct_str is not None:
            pnl_frac = float(pnl_pct_str) / 100.0
            _circuit_update_outcome(pnl_frac)
    except Exception:
        pass

    # AI tahmin logunu performans etiketiyle güncelle
    try:
        _tag_ai_prediction_with_trade(record)
    except Exception:
        # Buradaki hata trade_log'u bozmamalı
        pass


def get_daily_realized_pnl(target_date: Optional[date] = None) -> float:
    """
    trade_log.json içinden verilen gün için gerçekleşmiş (realized) PnL toplamını döndürür.
    - Sadece timestamp_close dolu kayıtlar dikkate alınır.
    - pnl_abs sayısal ise toplanır, aksi durumda o kayıt atlanır.
    target_date None ise bugünün UTC tarihi kullanılır.
    """
    rows = _read_trade_log()
    if not rows:
        return 0.0

    if target_date is None:
        target_date = datetime.utcnow().date()

    total = 0.0
    for r in rows:
        ts_close = r.get("timestamp_close")
        if not ts_close:
            continue
        try:
            dt_close = datetime.fromisoformat(ts_close)
        except Exception:
            continue
        if dt_close.date() != target_date:
            continue
        pnl_abs = r.get("pnl_abs")
        try:
            pnl_f = float(pnl_abs)
        except Exception:
            continue
        total += pnl_f
    return float(total)

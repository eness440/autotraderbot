# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ccxt

# Typing support for function annotations
from typing import Dict  # Added to avoid NameError on Dict

# --- GÜVEN SKORUNA GÖRE KADEMELİ RİSK AYARLARI ---
RISK_TIERS = {
    # Güven Skoru Aralığı -> (Kaldıraç, Cüzdan Kullanım Yüzdesi)
    # Daha konservatif kademeler: yüksek güven bölgelerinde kaldıraç ve cüzdan
    # kullanımını önceki değerlere göre azaltarak aşırı kaldıraç ve risk
    # birikimini engeller. Bu sayede uzun süreli ayakta kalma olasılığı artar.
    # Küçük hesaplarda (<= ~$500) *agresif* cüzdan yüzdeleri liq riskini büyütür.
    # Bu nedenle % kullanım daha düşük tutuldu.
    (0.65, 0.70): (5, 0.06),
    (0.70, 0.75): (8, 0.07),
    (0.75, 0.80): (10, 0.08),
    (0.80, 0.85): (15, 0.10),
    (0.85, 0.90): (20, 0.11),
    (0.90, 0.95): (25, 0.12),
    (0.95, 1.01): (25, 0.12)
}
MAX_PORTFOLIO_RISK_PERCENTAGE = 0.05

# YENİ EKLENDİ: Hesap Büyüklüğüne Göre Cüzdan Limiti Ayarları
WALLET_CAP_TIERS = {
    # Bakiye Eşiği -> İzin Verilen Maksimum Cüzdan Yüzdesi
    # Küçük bakiye: daha düşük cap (aksi halde tek pozisyon bile hesabı kilitleyebilir)
    0: 0.08,      # $0+ için max %8
    1500: 0.12,   # $1500+ için max %12
    3500: 0.18,   # $3500+ için max %18
    5000: 0.22    # $5000+ için max %22
}

# Dinamik risk konfigürasyonu yükle.  risk_optimizer.py tarafından üretilen
# ``risk_config.json`` dosyası mevcutsa, içindeki ``tiers`` bilgileri
# kullanılarak RISK_TIERS sözlüğü güncellenir. Böylece risk optimizasyon
# görevlerinin sonuçları canlı konfigürasyona yansır.
import json as _json

try:
    _rconf_path = Path(__file__).resolve().parent / "risk_config.json"
    if _rconf_path.exists():
        try:
            with _rconf_path.open("r", encoding="utf-8") as _rf:
                _rc = _json.load(_rf)
            if isinstance(_rc, dict):
                _tiers = _rc.get("tiers")
                if isinstance(_tiers, list) and _tiers:
                    _new_tiers: dict[tuple[float, float], tuple[int, float]] = {}
                    for it in _tiers:
                        try:
                            lo = float(it.get("min_conf"))
                            hi = float(it.get("max_conf"))
                            lev = int(it.get("leverage"))
                            pct = float(it.get("wallet_pct"))
                            _new_tiers[(lo, hi)] = (lev, pct)
                        except Exception:
                            continue
                    if _new_tiers:
                        RISK_TIERS.clear()
                        RISK_TIERS.update(_new_tiers)
        except Exception:
            pass
except Exception:
    pass

# -------------------------------------------------------------------------
# Risk Profile Ölçekleme
#
# config.json'da "risk_profile" ve "risk_profiles" alanları tanımlandığında
# kaldıraç ve cüzdan kullanım yüzdesi bu ölçekle çarpılır. Bu sayede
# kullanıcı 'Aggressive', 'Balanced' veya 'Conservative' gibi profiller
# arasında seçim yapabilir. Varsayılan ölçek 1.0'dır.
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def _get_risk_scale() -> float:
    """
    config.json'daki risk_profile ayarına göre bir ölçek döndürür.
    Eğer profil veya ayar bulunamazsa 1.0 döndürülür.
    """
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            profile_name = data.get("risk_profile")
            profiles = data.get("risk_profiles") or {}
            if isinstance(profile_name, str) and isinstance(profiles, dict):
                prof = profiles.get(profile_name)
                if isinstance(prof, dict):
                    scale = prof.get("scale")
                    if isinstance(scale, (int, float)):
                        # Güvenlik için ölçek değerini makul aralıkta tut
                        return max(0.5, min(1.5, float(scale)))
    except Exception:
        pass
    return 1.0

def calculate_tiered_leverage_and_allocation(master_confidence_score: float) -> Dict[str, float]:
    """
    Hibrit Güven Skoruna göre, tanımlı kademelerden uygun kaldıraç ve cüzdan
    kullanımını seçer. Ek olarak, mevcut değerleri daha rafine bir şekilde
    ayarlamak için portföy optimizasyon fonksiyonunu çağırır. Bu sayede
    Kelly veya benzeri optimizasyonlarla önerilen kaldıraç ve cüzdan
    oranı harmanlanır.

    Args:
        master_confidence_score: 0–1 aralığında güven skoru.

    Returns:
        Sözlük biçiminde {'leverage': int, 'wallet_allocation_percent': float}
    """
    leverage: int = 0
    wallet_allocation: float = 0.0
    # Öncelikle tier seçim
    for (min_score, max_score), (lev, alloc) in RISK_TIERS.items():
        if min_score <= master_confidence_score < max_score:
            leverage = lev
            wallet_allocation = alloc
            break
    # Risk profil ölçeğini uygula
    scale = _get_risk_scale()
    try:
        leverage = int(round(max(0, leverage * scale)))
    except Exception:
        leverage = int(leverage)
    try:
        wallet_allocation = float(wallet_allocation) * scale
    except Exception:
        pass
    # Portföy optimizasyonu: optimize_leverage_and_allocation
    try:
        from .portfolio_optimizer import optimize_leverage_and_allocation  # type: ignore
        # RISK_TIERS'taki maksimum değerleri çıkar (üst sınırlar)
        max_leverage_tier = max(v[0] for v in RISK_TIERS.values())
        max_alloc_tier = max(v[1] for v in RISK_TIERS.values())
        # Optimize et
        new_lev, new_alloc = optimize_leverage_and_allocation(
            master_conf=float(master_confidence_score),
            current_leverage=int(leverage),
            current_alloc=float(wallet_allocation),
            risk_reward=1.0,
            max_leverage=max_leverage_tier,
            min_leverage=5,
            max_alloc=max_alloc_tier,
            min_alloc=0.02,
        )
        if isinstance(new_lev, (int, float)) and isinstance(new_alloc, (int, float)):
            leverage = int(round(new_lev))
            wallet_allocation = float(new_alloc)
    except Exception:
        # Optimize fonksiyon yoksa hatasız devam et
        pass

    # Nihai güvenlik clamp'leri
    if leverage > 0:
        leverage = int(max(5, min(25, leverage)))

    # --- Trade parametreleri ile override ---
    # config.json içindeki trade_parameters alanı, dinamik hesaplanan
    # kaldıraç ve cüzdan kullanım yüzdesini override edebilir. Böylece
    # kullanıcı sabit bir kaldıraç veya sabit cüzdan yüzdesi tanımlayabilir.
    try:
        if CONFIG_PATH.exists():
            cfg_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            params = cfg_data.get("trade_parameters") or {}
            # Sabit kaldıraç (null değilse)
            fix_lev = params.get("fixed_leverage")
            if fix_lev is not None:
                try:
                    leverage = int(fix_lev)
                except Exception:
                    pass
            # Sabit cüzdan yüzdesi
            fix_alloc = params.get("wallet_allocation_percent")
            if fix_alloc is not None:
                try:
                    wallet_allocation = float(fix_alloc)
                except Exception:
                    pass
    except Exception:
        pass
    return {
        'leverage': leverage,
        'wallet_allocation_percent': wallet_allocation
    }

# YENİ EKLENDİ: Dinamik Cüzdan Limiti Uygulama Fonksiyonu
def apply_dynamic_wallet_cap(wallet_allocation_percent, total_balance_usdt):
    """
    Hesap büyüklüğüne göre, hesaplanan cüzdan kullanım yüzdesine bir üst limit uygular.
    """
    # Bakiye eşiklerini büyükten küçüğe kontrol et
    for balance_threshold, max_allocation in sorted(WALLET_CAP_TIERS.items(), reverse=True):
        if total_balance_usdt > balance_threshold:
            # Eğer hesaplanan yüzde, o seviyenin limitinden büyükse, limiti kullan.
            capped_allocation = min(wallet_allocation_percent, max_allocation)
            if capped_allocation != wallet_allocation_percent:
                # print(f"[SERMAYE KORUMA]: Bakiye (${total_balance_usdt:.2f}) > ${balance_threshold}. Cüzdan kullanımı %{int(capped_allocation*100)} ile sınırlandı.")
                pass
            return capped_allocation

    # Eğer hiçbir eşik aşılmadıysa (bakiye 1500'den küçükse), orijinal yüzdeyi kullan
    return wallet_allocation_percent


def calculate_position_size(total_balance_usdt, wallet_allocation_percent, leverage, entry_price):
    """
    Dinamik cüzdan kullanımı ve kaldıraca göre pozisyon büyüklüğünü (coin adedi) hesaplar.
    """
    if wallet_allocation_percent == 0 or leverage == 0:
        return 0.0

    capital_to_use = total_balance_usdt * wallet_allocation_percent
    position_size_usd = capital_to_use * leverage
    position_size_coin = position_size_usd / entry_price

    return round(position_size_coin, 3)

# --- KADEMELİ GİRİŞ ve ÇIKIŞ SEVİYELERİ ---

def get_entry_levels(total_position_size, confidence_score):
    """
    Güven skoruna göre kademeli giriş miktarlarını belirler (Scaling In).
    """
    if confidence_score < 0.65:
        return []

    first_entry = total_position_size * 0.50
    second_entry = total_position_size * 0.25
    third_entry = total_position_size * 0.25

    return [round(first_entry, 3), round(second_entry, 3), round(third_entry, 3)]

def get_exit_levels(entry_price, tp_distance, total_position_size, side='long'):
    """
    Kademeli kâr alma (Scaling Out) için hedefleri ve miktarları hesaplar.
    """
    tp1_price = 0
    if side == 'long':
        tp1_price = entry_price + (tp_distance * 0.5)
    else:  # 'short'
        tp1_price = entry_price - (tp_distance * 0.5)

    exit_levels = [
        {
            'price': round(tp1_price, 4),
            'size_percentage': 0.50,
            'triggered': False
        }
    ]

    return exit_levels

# --- VOLATİLİTEYE GÖRE DİNAMİK AYARLAR ---

# ATR / fiyat oranına göre volatilite kademeleri
# Kullanıcı geribildirimi üzerine SL çarpanları artırıldı. Böylece stop-loss
# mesafesi, ATR'nin 2–3 katı civarında ayarlanır ve dar mesafelerden dolayı
# emriniz reddedilmez.
VOLATILITY_TIERS = [
    # düşük vol: ATR oranı <0.005 → SL çarpanı 2.0 (önceki 1.2 idi)
    {"min_ratio": 0.0,    "max_ratio": 0.005,  "category": "low",    "sl_mult": 2.0, "tp_mult": 3.0, "risk_factor": 1.0},
    # orta vol: ATR oranı 0.005–0.015 → SL çarpanı 2.5 (önceki 1.0 idi)
    {"min_ratio": 0.005,  "max_ratio": 0.015,  "category": "medium", "sl_mult": 2.5, "tp_mult": 2.5, "risk_factor": 0.9},
    # yüksek vol: ATR oranı >=0.015 → SL çarpanı 3.0 (önceki 1.5 idi). TP biraz daha dar.
    {"min_ratio": 0.015,  "max_ratio": float("inf"), "category": "high",   "sl_mult": 3.0, "tp_mult": 2.0, "risk_factor": 0.7},
]

def classify_volatility(atr, price):
    """
    ATR/fiyat oranına göre volatilite kategorisi ve TP/SL katsayılarını döndürür.

    Dönüş:
      {
        "category": "low"|"medium"|"high",
        "sl_mult": float,
        "tp_mult": float,
        "risk_factor": float,
        "atr_ratio": float
      }
    veya None
    """
    try:
        if atr is None or price is None:
            return None
        atr_f = float(atr)
        price_f = float(price)
        if atr_f <= 0 or price_f <= 0:
            return None
        ratio = atr_f / price_f
    except Exception:
        return None

    for tier in VOLATILITY_TIERS:
        if tier["min_ratio"] <= ratio < tier["max_ratio"]:
            out = dict(tier)
            out["atr_ratio"] = ratio
            return out
    return None

def adjust_risk_for_volatility(leverage, wallet_allocation_percent, atr, price):
    """
    Volatiliteye göre kaldıraç ve cüzdan kullanım yüzdesini ölçekler.

    - Düşük vol: risk_factor ~1.0, değişiklik az.
    - Yüksek vol: risk_factor <1, kaldıraç ve pozisyon boyutu küçülür.
    """
    info = classify_volatility(atr, price)
    if not info:
        return leverage, wallet_allocation_percent, None

    rf = float(info.get("risk_factor", 1.0))

    try:
        base_lev = leverage or 0
        lev = int(round(max(0, base_lev * rf)))
    except Exception:
        lev = leverage

    try:
        if wallet_allocation_percent is not None:
            alloc = wallet_allocation_percent * rf
        else:
            alloc = wallet_allocation_percent
    except Exception:
        alloc = wallet_allocation_percent

    return lev, alloc, info

# --- DİNAMİK TP/SL ve TRAILING STOP ---

def calculate_dynamic_tp_sl(last_analyzed_data, multiplier_sl=2.0, multiplier_tp=4.0):
    """
    ATR bazlı dinamik TP ve SL mesafelerini hesaplar.
    """
    try:
        current_atr = last_analyzed_data['ATR_1h'].iloc[-1]
        sl_distance = current_atr * multiplier_sl
        tp_distance = current_atr * multiplier_tp
        return current_atr, sl_distance, tp_distance
    except (KeyError, IndexError):
        return None, None, None

def manage_trailing_stop(current_price, entry_price, current_sl_price, atr_sl_distance, is_long=True):
    """
    Adaptif Trailing Stop Loss (TSL) algoritmasını uygular.
    """
    if is_long:
        trailing_sl = current_price - atr_sl_distance
        # Yeni SL'i mevcut SL ve trailing SL'in maksimumu olarak seç
        new_sl_price = max(current_sl_price, trailing_sl)
        # Breakeven: fiyat yeterince ilerlediyse SL'i giriş fiyatının üstüne taşımayın
        # long pozisyonlarda SL giriş fiyatından daha yüksek olamaz; böylece pozisyon
        # kârda olsa bile marjin gereksiz yere sıkışmaz.
        if new_sl_price > entry_price:
            new_sl_price = entry_price
    else:
        trailing_sl = current_price + atr_sl_distance
        new_sl_price = min(current_sl_price, trailing_sl)
        # Short pozisyonlarda, SL giriş fiyatından daha düşük olamaz
        if new_sl_price < entry_price:
            new_sl_price = entry_price
    return new_sl_price

# --- YENİ: STANDART STOP-LOSS HESABI (tek kaynak) ---

def _clamp_to_tick(px: float, tick_size: float | None) -> float:
    if tick_size is None or tick_size <= 0:
        return float(px)
    # En yakın tick'e yuvarla
    return round(px / tick_size) * tick_size

def compute_stop_loss(
    side: str,
    entry_price: float,
    atr: float | None = None,
    atr_mult: float = 1.0,
    tick_size: float | None = None,
    # fallback SL genişletildi: 0.5% -> 1.0%
    percent_fallback: float | None = 0.01,
    last_price: float | None = None,
) -> float:
    """
    LONG: SL = entry - atr*mult (yoksa entry*(1 - %fallback))
    SHORT: SL = entry + atr*mult (yoksa entry*(1 + %fallback))

    - last_price verilirse mantık kontrolü yapılır:
        long → SL < last, short → SL > last (gerekirse 1 tick ayar)
    - Tick/precision clamp uygulanır.
    """
    side = str(side or "").lower()
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    if atr is not None and float(atr) > 0:
        if side == "long":
            sl = float(entry_price) - float(atr) * float(atr_mult)
        else:
            sl = float(entry_price) + float(atr) * float(atr_mult)
    else:
        pct = float(percent_fallback or 0.0)
        if side == "long":
            sl = float(entry_price) * (1.0 - pct)
        else:
            sl = float(entry_price) * (1.0 + pct)

    # last price ile tutarlılık
    if last_price is not None and float(last_price) > 0:
        step = tick_size if tick_size and tick_size > 0 else (float(last_price) * 0.001)
        if side == "long" and sl >= last_price:
            sl = float(last_price) - float(step)
        elif side == "short" and sl <= last_price:
            sl = float(last_price) + float(step)

    return _clamp_to_tick(sl, tick_size)

# -------------------------------------------------------------------------
# ÜCRET VE FONLAMA BİLİNÇLİ TP AYARLARI
#
# Bu fonksiyon, TP mesafesini işlem maliyetlerini (maker/taker fee) ve
# olası funding rate'leri hesaba katarak yeniden ayarlar. Özellikle
# hedef mesafesi dar olan işlemlerde bu maliyetler getiriyi silebileceği
# için, TP mesafesinin biraz genişletilmesi tavsiye edilir.
def adjust_tp_for_costs(
    tp_distance: float,
    entry_price: float,
    maker_fee_rate: float = 0.0005,
    funding_rate: float = 0.0001,
) -> float:
    """TP mesafesini ücret ve funding oranları için ayarla.

    Args:
        tp_distance: Hesaplanan orijinal TP mesafesi (mutlak fiyat farkı).
        entry_price: İşleme girilen fiyat.
        maker_fee_rate: Maker komisyon oranı (0.0005 = %0.05).
        funding_rate: 8 saatlik veya günlük tahmini funding oranı.

    Returns:
        Yeni TP mesafesi. Eğer maliyetler çok küçükse, değer büyük
        değişmez; aksi halde mesafe hafif artırılır.
    """
    try:
        base = float(tp_distance)
        # Toplam masraf oranı
        cost_ratio = float(maker_fee_rate) + float(funding_rate)
        if cost_ratio <= 0.0:
            return base
        # Fiyat başına maliyet
        cost_per_unit = float(entry_price) * cost_ratio
        # Ekstra mesafe: maliyetin bir katı kadar ekleyebiliriz
        extra = cost_per_unit
        new_dist = base + extra
        return max(new_dist, 0.0)
    except Exception:
        return float(tp_distance)

# --- PORTFÖY RİSK YÖNETİMİ ---
def check_portfolio_level_risk(open_positions, new_trade_risk_usd, balance):
    """
    Tüm açık pozisyonların ve yeni işlemin toplam riskini hesaplar ve limitle karşılaştırır.

    open_positions beklenen yapı (örnek):
      {
        "BTC/USDT": {
          "average_entry_price": float,
          "current_sl": float,
          "entries": [
              {"size": float},
              ...
          ]
        },
        ...
      }
    """
    if not open_positions:
        open_positions = {}

    current_total_risk_usd = 0.0

    for symbol, pos_data in open_positions.items():
        if not isinstance(pos_data, dict):
            continue

        entries = pos_data.get('entries') or []
        try:
            total_size = sum(float(entry.get('size', 0.0)) for entry in entries)
        except Exception:
            total_size = 0.0

        if total_size <= 0:
            continue

        try:
            avg_price = float(pos_data.get('average_entry_price'))
            current_sl = float(pos_data.get('current_sl'))
        except (TypeError, ValueError):
            # Gerekli alanlar yoksa bu pozisyonun riskini hesaplayamıyoruz, atla.
            continue

        risk_per_unit = abs(avg_price - current_sl)
        position_risk_usd = risk_per_unit * total_size
        current_total_risk_usd += position_risk_usd

    potential_total_risk = current_total_risk_usd + float(new_trade_risk_usd or 0.0)
    risk_limit_usd = float(balance) * MAX_PORTFOLIO_RISK_PERCENTAGE

    if potential_total_risk > risk_limit_usd:
        # print(f"!!! PORTFÖY RİSK UYARISI: Toplam risk (${potential_total_risk:.2f}), limiti (${risk_limit_usd:.2f}) aşıyor. İşlem İPTAL.")
        return False

    # print(f"[PORTFÖY RİSK KONTROLÜ]: Toplam Potansiyel Risk: ${potential_total_risk:.2f} (Limit: ${risk_limit_usd:.2f}) -> ONAYLANDI")
    return True

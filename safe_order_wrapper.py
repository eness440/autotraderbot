# -*- coding: utf-8 -*-
"""
safe_order_wrapper.py (Async Native v5 - ATOMIC SL/TP UPDATED)
- OKX Perpetual (tdMode=isolated, one-way ile uyumlu, posSide kullanmadan)
- DRY-RUN desteği (gerçek emir yok)
- async native: create_order, cancel_order, position fetch hepsi awaitable
- YENİ: await call(op, ...) → tek giriş noktası (rate-limit retry + backoff)
- [FIX] safe_submit_entry_plan artık attachAlgoOrds (atomik SL/TP) destekler.
"""

import asyncio
import time
import uuid
import re
from logger import get_logger

# Dinamik hesap risk yöneticisi
try:
    from .account_risk_manager import adjust_position_size  # type: ignore
except Exception:
    # Fallback fonksiyon: parametreleri değiştirmeden döndürür
    async def adjust_position_size(exchange, symbol, current_qty, leverage, entry_price, max_risk_pct=0.05):  # type: ignore
        return current_qty

# Slipaj tahmini için modül. Sipariş defteri tabanlı slippage hesabı
try:
    from .slippage_model import estimate_slippage  # type: ignore
except Exception:
    async def estimate_slippage(*args, **kwargs) -> float:  # type: ignore
        return 0.0

log = get_logger("safe_order_wrapper")


def _parse_okx_price_band(err_text: str):
    """Parse OKX dynamic price band info from a 51006 error string.

    OKX returns a message like:
      "Order price is not within the price limit (max buy price: 143.34, min sell price: 127.27)"

    Returns:
      (max_buy, min_sell) floats or (None, None) if not found.
    """
    if not err_text:
        return None, None
    # Try both plain text and JSON-ish payloads.
    # We intentionally keep regex permissive because OKX may vary spacing.
    max_buy = None
    min_sell = None
    try:
        m1 = re.search(r"max\s*buy\s*price\s*:\s*([0-9]*\.?[0-9]+)", err_text, flags=re.IGNORECASE)
        if m1:
            max_buy = float(m1.group(1))
    except Exception:
        max_buy = None
    try:
        m2 = re.search(r"min\s*sell\s*price\s*:\s*([0-9]*\.?[0-9]+)", err_text, flags=re.IGNORECASE)
        if m2:
            min_sell = float(m2.group(1))
    except Exception:
        min_sell = None
    return max_buy, min_sell

# -----------------------------------------------------------------------------
# Configuration for adaptive order retries
#
# When placing entry orders, insufficient margin or max position size
# errors (51008 and 51004) trigger an adaptive loop that reduces the
# leverage and/or position size and retries.  Without bounds this loop can
# iterate many times, especially on illiquid or very low‑priced tokens,
# causing unnecessary delays and potentially still failing.  Limit the
# number of adaptive retries via ``MAX_ADAPT_TRIES``.  A sensible default
# of 3 keeps the logic responsive without being overly persistent.
MAX_ADAPT_TRIES: int = 3

# Microcap adapt threshold: if the price for a given entry leg is below this
# threshold, adaptive order retries will be severely limited (one attempt).  This
# prevents extremely low‑priced tokens from entering an endless adapt loop.
MICROCAP_PRICE_THRESHOLD: float = 0.001

# Runtime blacklist for symbols that are not supported on OKX.  When a symbol
# triggers an "unknown market" type error (e.g., "does not have market
# symbol"), it is added here to avoid further order attempts in the same
# session.  Note: this set is in‑memory only; if persistent storage is
# required across restarts it should be saved to disk by the caller.
UNSUPPORTED_SYMBOLS: set[str] = set()

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────
def normalize_okx_symbol(symbol: str) -> str:
    """BTC/USDT → BTC-USDT-SWAP"""
    base, quote = symbol.split("/")
    return f"{base}-{quote}-SWAP"


async def safe_set_leverage(
    exchange,
    symbol: str,
    leverage: int,
    td_mode: str = "isolated",
):
    """
    OKX için kaldıraç ayarını güvenli şekilde yapar.
    - ccxt.set_leverage senkron olduğu için asyncio.to_thread ile sarılır.
    - one-way (net) mod ile uyumlu olması için sadece mgnMode gönderilir.
    """
    okx_symbol = normalize_okx_symbol(symbol)

    def _call():
        return exchange.set_leverage(
            leverage,
            okx_symbol,
            {
                "mgnMode": td_mode,
            },
        )

    try:
        res = await asyncio.to_thread(_call)
        log.info(f"[LEV] {symbol} lev={leverage}x olarak ayarlandı.")
        return res
    except Exception as e:
        log.warning(f"[LEV_FAIL] {symbol}: kaldıraç ayarı yapılamadı: {e}")
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    """OKX / ccxt rate-limit / quota hatasını metinden yakalamaya çalışır."""
    msg = str(exc)
    low = msg.lower()
    if "rate limit" in low or "too many requests" in low:
        return True
    if "429" in msg:
        return True
    if "quota" in low or "exceeded" in low:
        return True
    if "frequency" in low and "limit" in low:
        return True
    return False


async def _call_with_rate_limit_retry(
    op,
    *args,
    label: str = "ORDER",
    max_retries: int = 5,
    base_delay: float = 0.3,
    backoff: float = 1.7,
    propagate: bool = False,
    **kwargs,
):
    """
    Her türlü trade/order çağrısını rate-limit güvenli şekilde çalıştırır.
    - Eğer op async/coroutine ise doğrudan await eder.
    - Eğer op normal (senkron) bir fonksiyonsa asyncio.to_thread ile çalıştırır.
    - Bazı durumlarda op bir coroutine OBJESİ döndürebilir; o zaman da ayrıca await edilir.
    - Rate-limit/429/402 benzeri hatalarda exponential backoff ile tekrar dener.
    - Non-rate-limit hatalarında ilk hatada bırakır (eski davranışı korur).
    """
    delay = base_delay

    # Record the start time to measure latency across attempts.  We record
    # latency for the entire call, including any retries, as a single API
    # invocation from the caller's perspective.  The attempt counter
    # captures how many retries were needed.
    start_ts = time.time()
    # Import metrics and circuit breaker lazily to avoid circular import
    try:
        from .metrics_manager import record_api_call, record_api_error  # type: ignore
    except Exception:
        record_api_call = None  # type: ignore
        record_api_error = None  # type: ignore
    try:
        from .circuit_breaker import record_error as _circuit_record_error  # type: ignore
    except Exception:
        def _circuit_record_error() -> None:  # type: ignore
            return

    for attempt in range(1, max_retries + 1):
        try:
            # 1) Fonksiyon async mi, sync mi ayır
            if asyncio.iscoroutinefunction(op):
                res = await op(*args, **kwargs)
            else:
                # Senkron ccxt fonksiyonlarını event-loop'u bloklamadan çalıştır
                res = await asyncio.to_thread(op, *args, **kwargs)

            # 2) Bazı wrapper'lar coroutine OBJESİ döndürebilir → tekrar await et
            if asyncio.iscoroutine(res):
                res = await res

            # Successful call: record latency and retries
            if record_api_call:
                try:
                    latency = time.time() - start_ts
                    record_api_call(latency, attempt - 1)
                except Exception:
                    pass
            return res

        except Exception as e:
            # Rate limit hatalarını exponential backoff ile yeniden dene
            if _is_rate_limit_error(e) and attempt < max_retries:
                log.warning(
                    f"[RLIMIT] {label}: rate-limit/quota (try {attempt}/{max_retries}) "
                    f"→ sleep {delay:.2f}s"
                )
                await asyncio.sleep(delay)
                delay *= backoff
                continue

            # Non-rate-limit hatası: propagate=True ise exception'ı tekrar fırlat
            if propagate:
                # Record error and rethrow
                if record_api_error:
                    try:
                        record_api_error()
                    except Exception:
                        pass
                try:
                    _circuit_record_error()
                except Exception:
                    pass
                raise

            # Aksi halde hata mesajı logla, circuit breaker'a bildir ve metrikleri güncelle
            log.warning(f"[{label}_ERR] non-rate-limit error (try {attempt}): {e}")
            if record_api_error:
                try:
                    record_api_error()
                except Exception:
                    pass
            try:
                _circuit_record_error()
            except Exception:
                pass
            # small pause before giving up
            await asyncio.sleep(0.2)
            break

    # If we exhausted retries without success, record the call (latency and retries) as a failure
    if record_api_call:
        try:
            latency = time.time() - start_ts
            # attempt equals the final attempt count; subtract 1 because count includes first try
            record_api_call(latency, attempt - 1)
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# GENEL AMAÇLI TEK GİRİŞ NOKTASI
# ──────────────────────────────────────────────────────────────────────────────
async def call(op, *args, label: str = "CALL", **kwargs):
    """
    Borsa/ccxt fonksiyonlarını TEK YERDEN çalıştır:
      await call(exchange.create_order, sym, "limit", "buy", sz, px, params)
    """
    return await _call_with_rate_limit_retry(op, *args, label=label, **kwargs)


async def adaptive_order_sleep(
    exchange,
    min_sleep: float = 0.05,
    default_sleep: float = 0.1,
):
    """
    Emirler arası dinamik sleep:
    - OKX response header'lardaki kalan limit / reset zamanı varsa onu kullanır.
    - Yoksa ccxt.exchange.rateLimit (ms) değerini kullanır.
    - O da yoksa default_sleep uygular.
    """
    try:
        headers = getattr(exchange, "last_response_headers", None) or {}
        rem = None
        reset = None
        if isinstance(headers, dict):
            rem = headers.get("x-ratelimit-remaining") or headers.get("X-RateLimit-Remaining")
            reset = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")

        sleep_sec = None
        try:
            if rem is not None and reset is not None:
                rem_val = float(rem)
                reset_val = float(reset)
                now = time.time()
                # reset epoch timestamp gibi görünüyorsa (şimdiden büyükse) pencereyi hesapla
                if reset_val > now + 1:
                    window = max(0.0, reset_val - now)
                else:
                    # Aksi halde saniye cinsinden süre olarak ele al
                    window = max(0.0, reset_val)
                if rem_val > 0 and window > 0:
                    sleep_sec = max(min_sleep, window / rem_val)
        except Exception:
            sleep_sec = None

        if sleep_sec is None:
            rl_ms = getattr(exchange, "rateLimit", None)
            if isinstance(rl_ms, (int, float)) and rl_ms > 0:
                sleep_sec = max(min_sleep, rl_ms / 1000.0)
            else:
                sleep_sec = default_sleep

        await asyncio.sleep(sleep_sec)
    except Exception:
        await asyncio.sleep(default_sleep)


# ──────────────────────────────────────────────────────────────────────────────
# ACİL KAPAMA YARDIMCI FONKSİYONU
# ──────────────────────────────────────────────────────────────────────────────
async def _emergency_close_position(
    exchange,
    symbol: str,
    okx_symbol: str,
    side_sl: str,
    td_mode: str,
    ccy: str,
):
    """
    Pozisyon var ama SL yoksa, tüm pozisyonu reduceOnly MARKET ile kapatır.
    Bu sayede SL borsaya gitmediyse bile likidasyona kadar açık kalmaz.
    """
    try:
        positions = await call(exchange.fetch_positions, label=f"EMERG-POS-{symbol}")
    except Exception as e:
        log.critical(f"[EMERGENCY] {symbol}: pozisyonlar okunamadı, kapatma denemesi yapılamıyor: {e}")
        return

    # Pozisyon büyüklüğünü ve yönünü tespit et
    pos_size = 0.0
    pos_sign = 0.0
    try:
        for p in positions or []:
            raw_sym = p.get("symbol")
            if not raw_sym:
                continue
            # "BTC/USDT:USDT" → "BTC/USDT"
            sym_clean = str(raw_sym).split(":")[0]
            if sym_clean != symbol:
                continue
            sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
            try:
                sz = float(sz)
            except Exception:
                sz = 0.0
            if abs(sz) > 0:
                pos_size = abs(sz)
                pos_sign = 1.0 if sz > 0 else -1.0
                break
    except Exception as e:
        log.critical(f"[EMERGENCY] {symbol}: pozisyon boyutu okunamadı: {e}")
        return

    if pos_size <= 0:
        log.warning(f"[EMERGENCY] {symbol}: pozisyon bulunamadı, kapatılacak kontrat yok.")
        return

    # Pozisyon yönüne göre kapanış tarafını belirle: long → sell, short → buy
    close_side = "sell" if pos_sign > 0 else "buy"

    try:
        log.critical(
            f"[EMERGENCY] {symbol}: SL yok, pozisyon acil MARKET ile kapatılıyor. "
            f"side={close_side}, size={pos_size}"
        )
        res = await _call_with_rate_limit_retry(
            exchange.create_order,
            okx_symbol,
            "market",
            close_side,
            pos_size,
            None,
            {
                "tdMode": td_mode,
                "ccy": ccy,
                "reduceOnly": True,
            },
            label=f"EMERGENCY-CLOSE-{symbol}",
            propagate=True,
        )
        if res is None:
            log.critical(f"[EMERGENCY_FAIL] {symbol}: emergency market close create_order başarısız (retry sonrası).")
        else:
            log.critical(f"[EMERGENCY_OK] {symbol}: emergency market close gönderildi id={res.get('id')}")
    except Exception as e:
        log.critical(f"[EMERGENCY_CRIT] {symbol}: emergency market close hatası: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# ASENKRON EMİR PLANLARI
# ──────────────────────────────────────────────────────────────────────────────
async def safe_submit_entry_plan(
    exchange,
    symbol: str,
    base_side: str,
    total_size: float,
    entry_sizes: list,
    leverage: int,
    dry_run: bool = True,
    tick_size: float = None,
    limit_price: float = None,
    sl_price: float = None,  # [FIX] Atomic SL parametresi
    tp_price: float = None   # [FIX] Atomic TP parametresi
):
    """
    Giriş emirlerini güvenli şekilde oluşturur (async-native).
    - posSide göndermez, bu sayede one-way (net) modda da 51000 hatası vermez.
    - Fiyatı borsanın price-limit aralığına clamp etmeye çalışır.
    - [FIX] attachAlgoOrds ile giriş anında SL/TP ekler (Atomic Protection).
    """
    try:
        okx_symbol = normalize_okx_symbol(symbol)
        td_mode = "isolated"
        ccy = "USDT"

        # Enstrüman var mı kontrol et.  Eğer sembol daha önce desteklenmediği
        # için kara listeye alındıysa, hiç deneme yapma.
        if symbol in UNSUPPORTED_SYMBOLS:
            log.warning(f"[ENTRY_SKIP] {symbol}: symbol disabled due to missing market.")
            return None
        try:
            market = exchange.market(okx_symbol)
        except Exception:
            log.warning(f"[ENTRY_SKIP] {symbol}: okx does not have market symbol {okx_symbol}")
            # Bu sembolü kara listeye ekle ki aynı oturumda tekrar denenmesin
            try:
                UNSUPPORTED_SYMBOLS.add(symbol)
            except Exception:
                pass
            return None

        price_limits = market.get("limits", {}).get("price") or {}
        max_price = price_limits.get("max")
        min_price = price_limits.get("min")

        log.info(f"[ENTRY] {symbol} plan başlatılıyor (dry={dry_run})...")

        # Bu plan boyunca en az bir emrin başarıyla oluşturulup oluşturulmadığını takip et.
        # Eğer hiçbir giriş emri başarıya ulaşmazsa, plan başarısız sayılır ve çıkış emirleri gönderilmez.
        entry_success = False

        for idx, qty in enumerate(entry_sizes):
            side = "buy" if base_side == "long" else "sell"

            # İlk giriş market, sonraki girişler limit emri
            order_type = "market" if idx == 0 else "limit"

            price_for_order = None
            if order_type == "limit":
                # limit_price yoksa markete düş
                if limit_price is None:
                    order_type = "market"
                else:
                    price_for_order = limit_price

            # Price-limit clamp
            if price_for_order is not None:
                # Slipaj tahmini: limit girişlerde beklenen slip oranını hesapla
                try:
                    slip_ratio = await estimate_slippage(exchange, symbol, float(qty), side=side)
                except Exception:
                    slip_ratio = 0.0
                try:
                    # side buy ise fiyatı yukarı, sell ise aşağı ayarla
                    if slip_ratio and isinstance(slip_ratio, (int, float)):
                        if side == "buy":
                            price_for_order = price_for_order * (1.0 + float(slip_ratio))
                        else:
                            price_for_order = price_for_order * (1.0 - float(slip_ratio))
                except Exception:
                    pass
                # Clamp after slip adjustment
                if max_price is not None and price_for_order > max_price:
                    price_for_order = max_price
                if min_price is not None and price_for_order < min_price:
                    price_for_order = min_price

            # --- Hesap bazlı risk kontrolü ---
            # Giriş fiyatı olarak limit fiyatını veya mevcut piyasa fiyatını kullan.
            try:
                entry_price_for_risk = price_for_order
                if entry_price_for_risk is None:
                    try:
                        ticker = await call(exchange.fetch_ticker, symbol, label=f"RISK-TICK-{symbol}")
                        if ticker:
                            entry_price_for_risk = float(ticker.get("last"))
                    except Exception:
                        entry_price_for_risk = None
                # Sadece market/limit emirleri için risk ayarı yap
                if entry_price_for_risk:
                    adj_qty = await adjust_position_size(
                        exchange=exchange,
                        symbol=symbol,
                        current_qty=float(qty),
                        leverage=leverage,
                        entry_price=entry_price_for_risk,
                    )
                    # Eğer ayarlanmış miktar 0 veya çok küçükse bu entry'yi atla
                    if not adj_qty or adj_qty <= 0:
                        log.info(f"[ACC-RISK] {symbol}: hesap risk limiti nedeniyle giriş atlandı (idx={idx}).")
                        continue
                    if adj_qty < qty:
                        log.info(f"[ACC-RISK] {symbol}: miktar {qty:.6f}→{adj_qty:.6f} küçültüldü (idx={idx}).")
                    # qty değişkenini güncelle
                    qty = float(adj_qty)
            except Exception:
                pass

            # DRY-RUN
            if dry_run:
                log.info(
                    f"[DRY] {symbol}:{ccy} {side} {qty} {order_type}"
                    f"{' @ ' + str(price_for_order) if price_for_order is not None else ''} lev={leverage}"
                )
                await asyncio.sleep(0.05)
                continue

            # Parametreler
            params_base = {
                "tdMode": td_mode,
                "ccy": ccy,
            }

            # ------------------------
            # Idempotent client order ID
            # ------------------------
            # Benzersiz ve OKX'in kabul ettiği formatta bir client id üret.
            # OKX dokümantasyonuna göre ``clOrdId`` en fazla 32 karakter
            # uzunluğunda olmalı ve sadece alfanümerik karakterler içermelidir.
            # Önceki sürümlerde tire ("-") kullanımı bazı durumlarda
            # ``51000 Parameter clOrdId error`` hatasına yol açıyordu.  Bu
            # nedenle id oluştururken tire ve diğer özel karakterleri
            # kaldırıyoruz ve uzunluğu 32 karakter ile sınırlandırıyoruz.
            try:
                # Sembol adından ayraç karakterlerini kaldır
                base_sym = symbol.replace("/", "").replace("-", "")
                # Milisaniye zaman damgası
                ts = int(time.time() * 1000)
                # 8 karakterlik rastgele hex
                rnd = uuid.uuid4().hex[:8]
                raw_id = f"{base_sym}{ts}{rnd}"
                # OKX limiti: sadece ilk 32 karakteri kullan
                client_oid = raw_id[:32]
                params_base["clientOrderId"] = client_oid
                params_base["clOrdId"] = client_oid
            except Exception:
                # id oluşamazsa parametreyi eklemeyip devam et. Bu durumda
                # idempotent davranış garanti edilmez ancak emir yine de
                # gönderilir.
                pass
            # Eğer SL veya TP varsa 'attachAlgoOrds' ekle
            # Bu özellik sayesinde giriş emri gerçekleşir gerçekleşmez SL/TP kurulur.
            attach_list = []
            if sl_price:
                attach_list.append({
                    "slTriggerPx": str(sl_price),
                    "slOrdPx": "-1"
                })
            if tp_price:
                attach_list.append({
                    "tpTriggerPx": str(tp_price),
                    "tpOrdPx": "-1"
                })
            if attach_list:
                params_base["attachAlgoOrds"] = attach_list

            # Adaptif deneme: leverage/size azaltarak tekrar dene.  Burada
            # microcap semboller için adaptasyon sayısını kısaltıyoruz.  Eğer
            # giriş fiyatı MICROCAP_PRICE_THRESHOLD'in altındaysa, bu token
            # muhtemelen aşırı küçük değerli olduğundan adaptatif döngüyü
            # minimumda tutuyoruz (1 deneme).  Aksi takdirde global
            # ``MAX_ADAPT_TRIES`` değeri kullanılır.
            current_qty = float(qty)
            current_lever = int(leverage)
            attempt = 0
            max_attempts_local = MAX_ADAPT_TRIES
            try:
                # entry_price_for_risk değişkenini burada kullanabiliriz.
                price_val = entry_price_for_risk if 'entry_price_for_risk' in locals() else None
                if price_val and isinstance(price_val, (int, float)):
                    if price_val < MICROCAP_PRICE_THRESHOLD:
                        max_attempts_local = 1
            except Exception:
                pass
            # Retry until we reach the maximum number of adaptive attempts
            # Initialize ``res`` so it is always defined for the partial fill check below.
            res = None
            while attempt < max_attempts_local:
                params = dict(params_base)
                params["lever"] = current_lever
                try:
                    res = await _call_with_rate_limit_retry(
                        exchange.create_order,
                        okx_symbol,
                        order_type,
                        side,
                        current_qty,
                        price_for_order,
                        params,
                        label=f"ENTRY-{symbol}",
                        propagate=True,
                    )
                    if res is None:
                        log.warning(f"[ENTRY_FAIL] {symbol}: create_order başarısız (retry sonrası).")
                    else:
                        log.info(f"[LIVE] {symbol}: {order_type} gönderildi id={res.get('id')}")
                    break  # başarı veya None: sonraki emir
                except Exception as e:
                    msg = str(e)
                    # Certain error codes require immediate abort of the
                    # adaptive loop.  For example, code 59668 indicates
                    # that leverage cannot be changed while other orders are
                    # active, so there is no point in retrying.
                    if "59668" in msg:
                        log.warning(
                            f"[LEV_FAIL] {symbol}: {msg.strip()} – leverage change not allowed, aborting."
                        )
                        break
                    # 51004: max position size limit; 51008: insufficient margin
                    if ("51004" in msg or "51008" in msg) and attempt < MAX_ADAPT_TRIES - 1:
                        attempt += 1
                        code_51004 = "51004" in msg
                        code_51008 = "51008" in msg
                        # Hem kaldıraç hem miktarı azaltarak yeniden dene
                        # 1) Leverage azalt (en az 1x)
                        if current_lever > 1:
                            new_lever = max(1, int(current_lever * 0.8))
                            if new_lever < current_lever:
                                log.warning(
                                    f"[ADAPT] {symbol}: Hata {msg.strip()} → leverage {current_lever}x→{new_lever}x düşürülüyor."
                                )
                                current_lever = new_lever
                                await safe_set_leverage(exchange, symbol, current_lever, td_mode)
                        # 2) Miktarı azalt
                        shrink_factor = 0.8 if code_51004 else 0.7
                        new_qty = current_qty * shrink_factor
                        if new_qty < 1e-8:
                            log.warning(
                                f"[ADAPT] {symbol}: miktar çok küçük ({current_qty:.6f}), emir iptal ediliyor."
                            )
                            break
                        if new_qty < current_qty:
                            log.warning(
                                f"[ADAPT] {symbol}: Hata {msg.strip()} → miktar {current_qty:.6f}→{new_qty:.6f} küçültülüyor."
                            )
                            current_qty = new_qty
                        continue
                    # Diğer hatalar: logla ve çık.  Eğer hata mesajı
                    # sembolün mevcut olmadığını belirtiyorsa, bu
                    # sembolü kara listeye ekleyerek sonraki işlemleri
                    # engelle.
                    if "does not have market symbol" in msg.lower():
                        try:
                            UNSUPPORTED_SYMBOLS.add(symbol)
                        except Exception:
                            pass
                        log.warning(f"[ENTRY_ABORT] {symbol}: unsupported symbol, added to blacklist.")
                    log.warning(f"[ENTRY_FAIL] {symbol}: {e}")
                    break

            # ------------------------------------------------------
            # Bu giriş bacağı sonrası, global entry_success flag'ini güncelle.
            # Eğer create_order sonucu bir id döndürdüyse, en az bir emir başarıyla gönderilmiş demektir.
            try:
                if res is not None and res.get("id") is not None:
                    entry_success = True
            except Exception:
                pass

            # ------------------------------------------------------
            # Giriş emrinin durumu ve doldurma oranı kontrolü
            # ------------------------------------------------------
            # create_order çağrısı döndükten sonra, emrin durumu ve doldurulan
            # miktarı incelenir.  Eğer emir reddedilmiş veya iptal
            # edilmişse, bu durum loglanır.  Ayrıca, eğer emir kısmi
            # doldurulmuşsa (filled < amount), bu da uyarı olarak
            # bildirilir.  Bu kurallar, partial fill ve rejected
            # senaryolarını netleştirmek için eklenmiştir.
            if not dry_run:
                try:
                    if res is not None:
                        status = str(res.get("status", "")).lower() if res.get("status") is not None else ""
                        filled_val = res.get("filled") if res.get("filled") is not None else 0
                        amount_val = res.get("amount") if res.get("amount") is not None else 0
                        # Cast to float when possible
                        try:
                            filled_num = float(filled_val)
                        except Exception:
                            filled_num = 0.0
                        try:
                            amount_num = float(amount_val)
                        except Exception:
                            amount_num = 0.0
                        # Rejected or cancelled orders
                        if status in ("canceled", "cancelled", "rejected"):
                            log.warning(
                                f"[ENTRY_WARN] {symbol}: order {res.get('id')} (client {params_base.get('clientOrderId')}) status={status}, emir reddedildi veya iptal edildi."
                            )
                        # Partial fill detection
                        elif amount_num > 0 and filled_num < amount_num:
                            # If filled ratio is below 100%, log the partial fill
                            log.warning(
                                f"[ENTRY_PARTIAL] {symbol}: order {res.get('id')} (client {params_base.get('clientOrderId')}) partially filled {filled_num:.6f}/{amount_num:.6f}."
                            )
                except Exception:
                    # Silent catch to avoid breaking trading loop
                    pass

        # Return success status; include flag to signal whether entry was placed
        return {"status": ("ok" if entry_success else "fail"), "symbol": symbol}

    except Exception as e:
        log.exception(f"safe_submit_entry_plan hata: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ASENKRON ÇIKIŞ (TP/SL)
# ──────────────────────────────────────────────────────────────────────────────
async def safe_submit_exit_plan(
    exchange,
    symbol: str,
    base_side: str,
    tp_levels: list,
    sl_price: float = None,
    dry_run: bool = True,
    tick_size: float = None
):
    """
    Take-Profit ve Stop-Loss emirlerini güvenli oluşturur (async-native).
    - posSide göndermez (one-way ile uyumlu).
    - TP fiyatlarını borsa price-limit aralığına clamp eder.
    - SL fiyatını last price ile tutarlı olacak şekilde gerekirse düzeltir
      (long için SL < last, short için SL > last).
    - SL emri borsada bulunamazsa otomatik emergency-close çalışır.
    """
    try:
        okx_symbol = normalize_okx_symbol(symbol)
        td_mode = "isolated"
        ccy = "USDT"

        # Enstrüman var mı kontrol et
        try:
            market = exchange.market(okx_symbol)
        except Exception:
            log.warning(f"[EXIT_SKIP] {symbol}: okx does not have market symbol {okx_symbol}")
            return None

        price_limits = market.get("limits", {}).get("price") or {}
        max_price = price_limits.get("max")
        min_price = price_limits.get("min")

        log.info(f"[EXIT] {symbol} TP/SL planı başlatılıyor (dry={dry_run})...")

        # ------------------------------------------------------------------
        # Küçük TP bacaklarını filtrele
        # OKX her sembol için minimum emir miktarı sınırı uyguluyor. Eğer
        # hesaplanan TP bacağının miktarı bu sınırın altında kalırsa, emir
        # reddedilecektir. Bu nedenle tp_levels listesini gönderilmeden önce
        # eliyoruz. Eğer tüm bacaklar çok küçükse, tüm miktar bir araya
        # getirilerek tek bir TP emri olarak gönderilir. Böylece min
        # miktar kuralı korunur ve gereksiz hata kodlarından kaçınılır.
        try:
            min_qty = 0.0
            # limits.amount.min varsa onu al; bazı borsalarda None olabilir
            amt_limits = market.get("limits", {}).get("amount", {}) if market else {}
            if amt_limits:
                m = amt_limits.get("min")
                if m is not None:
                    try:
                        min_qty = float(m)
                    except Exception:
                        min_qty = 0.0
            # Filtreyi yalnızca pozitif min_qty varsa uygula
            if min_qty and tp_levels:
                # Toplam miktar
                total_sz = 0.0
                for lvl in tp_levels:
                    try:
                        total_sz += float(lvl.get("size") or 0.0)
                    except Exception:
                        pass
                valid_levels: list[dict] = []
                for lvl in tp_levels:
                    try:
                        sz = float(lvl.get("size") or 0.0)
                    except Exception:
                        sz = 0.0
                    # Eğer parça boyutu min sınırın altındaysa atla
                    if sz < min_qty:
                        log.warning(
                            f"[TP-SKIP] {symbol}: TP bacağı min amount altı (%.6f < %.6f), bu bacak atlanıyor.",
                            sz, min_qty
                        )
                        continue
                    valid_levels.append(lvl)
                # Hiç geçerli bacak kalmazsa ve toplam miktar min_qty'den büyükse tek bir TP oluştur
                if not valid_levels:
                    if total_sz >= min_qty:
                        # Tek TP: ilk bacağın fiyatını kullan
                        try:
                            first_price = float(tp_levels[0].get("price"))
                        except Exception:
                            first_price = None
                        if first_price is not None:
                            valid_levels = [{"price": first_price, "size": total_sz}]
                    # Eğer toplam da min altındaysa, TP emirleri tamamen atlanır
                tp_levels = valid_levels
        except Exception:
            # Filtre hatası durumunda tp_levels olduğu gibi bırak
            pass

        side_tp = "sell" if base_side == "long" else "buy"
        side_sl = "sell" if base_side == "long" else "buy"

        # TP emirleri için toplam miktar (SL için de kullanılacak)
        total_tp_qty = 0.0
        for lvl in tp_levels:
            try:
                total_tp_qty += float(lvl.get("size") or 0.0)
            except Exception:
                continue

        # TP emirleri (Limit)
        for lvl in tp_levels:
            qty = lvl.get("size")
            price = lvl.get("price")

            # Price-limit clamp
            if price is not None:
                if max_price is not None and price > max_price:
                    price = max_price
                if min_price is not None and price < min_price:
                    price = min_price

            if dry_run:
                log.info(
                    f"[DRY] TP {symbol}:{ccy} {side_tp} {qty} limit @ {price}"
                )
                await asyncio.sleep(0.05)
                continue

            params = {
                "tdMode": td_mode,
                "ccy": ccy,
                "reduceOnly": True,
            }
            try:
                # propagate=True so non-rate-limit errors (e.g. OKX 51006 price-band)
                # can be handled by the outer exception handler (band clamp + retry).
                res = await _call_with_rate_limit_retry(
                    exchange.create_order,
                    okx_symbol,
                    "limit",
                    side_tp,
                    qty,
                    price,
                    params,
                    label=f"TP-{symbol}",
                    propagate=True,
                )
                if res is None:
                    log.warning(f"[EXIT_FAIL] {symbol}: TP create_order başarısız (retry sonrası).")
                else:
                    log.info(f"[LIVE-TP] {symbol}: limit TP gönderildi id={res.get('id')}")
            except Exception as e:
                err_text = str(e)
                # Handle OKX dynamic price band error (51006) by clamping within band and retrying once
                max_buy, min_sell = _parse_okx_price_band(err_text)
                if (max_buy is not None) or (min_sell is not None):
                    adj_price = price
                    try:
                        if side_tp == "buy" and max_buy is not None and adj_price is not None:
                            adj_price = min(float(adj_price), float(max_buy))
                        if side_tp == "sell" and min_sell is not None and adj_price is not None:
                            adj_price = max(float(adj_price), float(min_sell))
                    except Exception:
                        adj_price = price
                    if adj_price is not None and adj_price != price:
                        try:
                            log.warning(f"[TP-{symbol}_BAND] TP price {price} out of band; clamped to {adj_price} and retrying once.")
                            res2 = await call_safe(
                                exchange,
                                "create_order",
                                okx_symbol,
                                "limit",
                                side_tp,
                                qty,
                                adj_price,
                                params,
                                label=f"TP-{symbol}-band",
                            )
                            if res2 is None:
                                log.warning(f"[EXIT_FAIL] {symbol}: TP create_order başarısız (band retry sonrası).")
                            else:
                                log.info(f"[LIVE-TP] {symbol}: limit TP gönderildi id={res2.get('id')} (band clamped)")
                            continue
                        except Exception as e2:
                            log.warning(f"[EXIT_FAIL] {symbol}: band retry failed: {e2}")
                log.warning(f"[EXIT_FAIL] {symbol}: {e}")

        # Stop-Loss (Yedek, MARKET + stopLossPrice)
        if sl_price and total_tp_qty > 0:
            if dry_run:
                log.info(f"[DRY] SL {symbol}:{ccy} {side_sl} market stop @ {sl_price}, qty={total_tp_qty}")
            else:
                try:
                    # Last price al, SL'yi kurala göre düzelt
                    last_price = None
                    try:
                        ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"TICKER-SL-{symbol}")
                        last_price = ticker.get("last")
                    except Exception:
                        last_price = None

                    adj_sl = sl_price
                    if last_price is not None:
                        step = tick_size or (last_price * 0.001)
                        if base_side == "long" and adj_sl >= last_price:
                            adj_sl = last_price - step
                        elif base_side == "short" and adj_sl <= last_price:
                            adj_sl = last_price + step

                    params = {
                        "tdMode": td_mode,
                        "ccy": ccy,
                        "stopLossPrice": float(adj_sl),
                        "reduceOnly": True,
                    }
                    res = await _call_with_rate_limit_retry(
                        exchange.create_order,
                        okx_symbol,
                        "market",          # OKX: type="market" + stopLossPrice → trigger SL
                        side_sl,
                        float(total_tp_qty),
                        None,
                        params,
                        label=f"SL-{symbol}",
                    )
                    if res is None:
                        log.warning(f"[STOP_FAIL] {symbol}: stop-loss create_order başarısız (retry sonrası).")
                    else:
                        log.info(f"[LIVE-SL] {symbol}: stop-loss gönderildi id={res.get('id')} (sl={adj_sl})")
                except Exception as e:
                    log.warning(f"[STOP_FAIL] {symbol}: {e}")

        # ──────────────────────────────────────────────────────────────────────
        # TP/SL GERÇEKTEN BORSADA VAR MI? OTOMATİK KONTROL + EMERGENCY CLOSE
        # ──────────────────────────────────────────────────────────────────────
        try:
            if not dry_run:
                # 1) Açık pozisyon var mı kontrol et
                pos_exists = False
                try:
                    positions = await call(exchange.fetch_positions, label=f"POS-CHECK-{symbol}")
                    for p in positions:
                        raw_sym = p.get("symbol")
                        if not raw_sym:
                            continue
                        # "BTC/USDT:USDT" → "BTC/USDT"
                        sym_clean = str(raw_sym).split(":")[0]
                        if sym_clean != symbol:
                            continue
                        sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
                        try:
                            sz = float(sz)
                        except Exception:
                            sz = 0.0
                        if abs(sz) > 0:
                            pos_exists = True
                            break
                except Exception as e:
                    log.warning(f"[SL_CHECK] {symbol}: pozisyonlar okunamadı: {e}")

                # 2) Açık emirlerden TP/SL say
                tp_exists = False
                sl_exists = False
                try:
                    orders = await call(exchange.fetch_open_orders, okx_symbol, label=f"OPEN-CHECK-{symbol}")
                except Exception as e:
                    log.warning(f"[SL_CHECK] {symbol}: open_orders okunamadı: {e}")
                    orders = []

                for o in orders:
                    try:
                        o_side = o.get("side")
                        o_type = o.get("type")
                        info = o.get("info") or {}

                        # TP: limit emir, pozisyonun kapanış yönünde
                        if o_side == side_tp and o_type == "limit":
                            tp_exists = True

                        # TP ayrıca attachAlgoOrds ile 'tpTriggerPx' veya 'tpOrdPx' alanlarıyla gelebilir
                        if o_side == side_tp and (
                            info.get("tpTriggerPx") is not None or info.get("tpOrdPx") is not None
                        ):
                            tp_exists = True

                        # SL: stop / stop-market veya stopLoss / slTriggerPx / slOrdPx alanı olan emir
                        # OKX attachAlgoOrds ile girilenler bazen open_orders'da değil algo_orders'da görünebilir,
                        # ancak bu kontrol manuel SL için hala faydalıdır.
                        if o_side == side_sl and (
                            o_type in ("stop", "stop-market")
                            or info.get("slTriggerPx") is not None
                            or info.get("slOrdPx") is not None
                            or info.get("stopLossPrice") is not None
                            or info.get("tpTriggerPx") is not None  # attachAlgoOrds
                            or info.get("tpOrdPx") is not None
                            or "algoId" in info  # Algoritmik emir (attach ile gelen)
                        ):
                            sl_exists = True
                    except Exception:
                        continue

                if pos_exists:
                    # SL emri anlık sorguda görünmüyorsa, direkt emergency close yapmak
                    # yerine sadece uyarı ver. Asıl kontrol stop_order_watchdog tarafından
                    # yapılacaktır. Bu sayede API gecikmesi veya filtre farkları yüzünden
                    # gereksiz acil kapamalar önlenir.
                    if not sl_exists:
                        log.warning(
                            f"[SAFE_EXIT] {symbol}: SL emri anlık sorguda görünmüyor. "
                            f"Gerçek kontrol stop_order_watchdog ile yapılacak."
                        )
                    if not tp_exists:
                        log.warning(
                            f"[WARN] {symbol}: açık pozisyon için TP emri görünmüyor."
                        )
        except Exception as e:
            log.warning(f"[SL_CHECK_FAIL] {symbol}: {e}")

        return {"status": "ok", "symbol": symbol}

    except Exception as e:
        log.exception(f"safe_submit_exit_plan hata: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# TRAILING STOP / STOP-LOSS GÜNCELLEME
#
# Aşağıdaki yardımcı fonksiyon, mevcut reduceOnly stop-loss emirlerini iptal eder
# ve yeni bir stop-loss emri gönderir. Bu fonksiyon trailing stop stratejilerinde
# kullanılmak üzere tasarlanmıştır. Yalnızca açık pozisyonlar için çalışır.

async def safe_update_stop_loss(
    exchange,
    symbol: str,
    new_sl: float,
    tick_size: float | None = None,
    base_side: str | None = None,
    dry_run: bool | None = None,
) -> None:
    """
    Mevcut reduceOnly stop-loss emirlerini iptal eder ve yeni bir stop-loss market emri gönderir.

    Args:
        exchange: ccxt exchange instance (async)
        symbol: Örnek "BTC/USDT"
        new_sl: Yeni stop-loss fiyatı
        tick_size: Fiyat hassasiyeti; sağlanmışsa SL bu adıma yuvarlanır
        base_side: "long" veya "short"; pozisyon yönü. Yoksa pozisyonlardan otomatik belirlenir.
        dry_run: None ise global WRAPPER_DRY_RUN kullanılır; True ise emir gönderilmez.
    """
    try:
        okx_symbol = normalize_okx_symbol(symbol)
        td_mode = "isolated"
        ccy = "USDT"
        # dry_run parametresi belirtilmemişse main_bot_async içindeki
        # WRAPPER_DRY_RUN değişkenini kullanmayı dene; bulunamazsa
        # varsayılan olarak gerçek emir gönderilir (dry_run=False). Bu
        # değişiklik, global bağımlılığı azaltmak ve config üzerinden
        # kontrol edilebilirlik sağlamak için yapıldı.
        # ------------------------------------------------------------------
        # Determine dry_run flag
        #
        # Historically this function imported WRAPPER_DRY_RUN from
        # main_bot_async when dry_run was not explicitly provided.  That
        # created a hidden dependency on the trading loop and made it
        # impossible to override the behaviour from configuration.  To fix
        # this bug we now derive the default dry_run from the configuration
        # file.  If the configuration file defines a boolean in
        # ``trade_parameters.dry_run`` it will be used; otherwise the
        # default is ``False``.
        if dry_run is None:
            try:
                from pathlib import Path
                import json
                cfg_path = Path(__file__).resolve().parent / "config.json"
                if cfg_path.exists():
                    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
                    params = cfg_data.get("trade_parameters") or {}
                    if params.get("dry_run") is not None:
                        dry_run = bool(params.get("dry_run"))
                    else:
                        dry_run = False
                else:
                    dry_run = False
            except Exception:
                dry_run = False
        # Pozisyon yönü bilinmiyorsa tespit etmeye çalış
        if base_side not in ("long", "short"):
            try:
                positions = await call(exchange.fetch_positions, label=f"SLUPD-POS-{symbol}")
                for p in positions or []:
                    raw_sym = p.get("symbol")
                    if not raw_sym:
                        continue
                    sym_clean = str(raw_sym).split(":")[0]
                    if sym_clean != symbol:
                        continue
                    # Birçok borsada pozisyon yönü 'side' ya da 'positionSide' alanında bulunur
                    pos_side = p.get("side") or p.get("positionSide") or None
                    if pos_side:
                        base_side = str(pos_side).lower()
                        break
            except Exception:
                base_side = None
        side_sl = "sell" if base_side == "long" else "buy"
        # Mevcut stop-loss emirlerini iptal et
        if not dry_run:
            try:
                orders = await call(exchange.fetch_open_orders, okx_symbol, label=f"SLUPD-OPEN-{symbol}")
                for o in orders or []:
                    try:
                        info = o.get("info") or {}
                        # reduceOnly + stop-loss niteliği taşıyan emirleri belirle
                        is_reduce = (
                            o.get("reduceOnly")
                            or info.get("reduceOnly")
                            or info.get("sideEffectType") == "ReduceOnly"
                        )
                        has_sl = (
                            info.get("stopLossPrice") is not None
                            or info.get("slTriggerPx") is not None
                            or (info.get("tpTriggerPx") is None and info.get("algoId") is not None)
                        )
                        if is_reduce and has_sl:
                            await call(exchange.cancel_order, o.get("id"), okx_symbol, label=f"SLUPD-CANCEL-{symbol}")
                    except Exception:
                        continue
            except Exception:
                pass
        # Pozisyon büyüklüğünü al
        pos_size = None
        try:
            positions = await call(exchange.fetch_positions, label=f"SLUPD-POS2-{symbol}")
            for p in positions or []:
                raw_sym = p.get("symbol")
                if not raw_sym:
                    continue
                if str(raw_sym).split(":")[0] != symbol:
                    continue
                sz = p.get("contracts") or p.get("size") or p.get("positionAmt")
                try:
                    sz_f = float(sz)
                except Exception:
                    sz_f = 0.0
                if abs(sz_f) > 0:
                    pos_size = abs(sz_f)
                    break
        except Exception:
            pos_size = None
        if not pos_size:
            return None
        # Son fiyatı alarak SL'i güncelle
        adj_sl = float(new_sl)
        if not dry_run:
            last_price = None
            try:
                ticker = await call(exchange.fetch_ticker, okx_symbol, label=f"SLUPD-TICKER2-{symbol}")
                last_price = ticker.get("last")
            except Exception:
                last_price = None
            if last_price is not None and float(last_price) > 0:
                # Adım, verilen tick_size yoksa last_price*0.1% olarak kullanılır
                step = tick_size or (float(last_price) * 0.001)
                if base_side == "long":
                    if adj_sl >= float(last_price):
                        adj_sl = float(last_price) - step
                else:
                    if adj_sl <= float(last_price):
                        adj_sl = float(last_price) + step
        # Emir oluştur
        if dry_run:
            log.info(f"[DRY] SL_UPDATE {symbol}:{ccy} {side_sl} stop @ {adj_sl} qty={pos_size}")
        else:
            params = {
                "tdMode": td_mode,
                "ccy": ccy,
                "stopLossPrice": float(adj_sl),
                "reduceOnly": True,
            }
            try:
                res = await _call_with_rate_limit_retry(
                    exchange.create_order,
                    okx_symbol,
                    "market",
                    side_sl,
                    float(pos_size),
                    None,
                    params,
                    label=f"SLUPD-{symbol}",
                )
                if res is not None:
                    log.info(f"[SL_UPDATE] {symbol}: yeni SL emri gönderildi @ {adj_sl:.6f}")
                else:
                    log.warning(f"[SL_UPDATE_FAIL] {symbol}: stop update create_order başarısız (retry sonrası)")
            except Exception as e:
                log.warning(f"[SL_UPDATE_FAIL] {symbol}: {e}")
        return None
    except Exception as e:
        log.warning(f"safe_update_stop_loss hata: {e}")
        return None

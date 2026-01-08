import ccxt
import pandas as pd
import talib as ta
import numpy as np
import time
import asyncio
import inspect
import logging
import os

# Logger kurulumu
log = logging.getLogger(__name__)

def _env_flag(key: str, default: str = "0") -> bool:
    return str(os.getenv(key, default)).strip().lower() in ("1", "true", "yes", "y", "on")

DEBUG_MTF = _env_flag("DEBUG_MTF", "0") or _env_flag("BOT_DEBUG", "0")


def _resolve_symbol(exchange, symbol: str):
    """
    OKX (ve diğer borsalar) için sembolü gerçek market anahtarına çevirir.
    Örnek:
      - "BTC/USDT" spot veya swap sembolü varsa onu kullanır
      - "BTC/USDT:USDT" / "BTC/USDT:USDC" varyantlarını dener
      - OKX swap id'si "BTC-USDT-SWAP" olan marketi bulup onun symbol alanını döndürür
    """
    mkts = exchange.markets or {}
    
    # [DEBUG] İlk 3 çağrı için print
    if not hasattr(_resolve_symbol, '_debug_count'):
        _resolve_symbol._debug_count = 0
    if _resolve_symbol._debug_count < 3:
        print(f"[RESOLVE_DEBUG] exchange.markets sayısı: {len(mkts)}")
        if len(mkts) == 0:
            print(f"[RESOLVE_CRITICAL] MARKETS BOŞ! load_markets() çağrılmamış olabilir!")
        _resolve_symbol._debug_count += 1

    # 1) Doğrudan sembol
    if symbol in mkts:
        return symbol

    # 2) OKX tarzı "BTC/USDT:USDT" ve "BTC/USDT:USDC"
    for alt in (f"{symbol}:USDT", f"{symbol}:USDC"):
        if alt in mkts:
            return alt

    # 3) OKX SWAP id → symbol eşleşmesi (BTC/USDT → BTC-USDT-SWAP)
    try:
        target_id = f"{symbol.replace('/', '-')}-SWAP"
        for m in mkts.values():
            if m.get("id") == target_id:
                sym = m.get("symbol")
                if sym:
                    # symbol anahtar olarak varsa onu kullan
                    if sym in mkts:
                        return sym
                    return sym
    except Exception:
        pass

    # Hiçbiri bulunamadı
    return None

def _ensure_symbol_exists(exchange, symbol: str):
    """
    Eski fonksiyon korunuyor ama artık _resolve_symbol kullanıyor.
    Sadece varlık kontrolü yapar.
    """
    return _resolve_symbol(exchange, symbol) is not None

def fetch_and_analyze_data(exchange, symbol, timeframe='15m', limit=1500):
    """
    Çoklu borsa uyumlu teknik analiz (20+ indikatör).
    OKX market kontrolü, NaN-tolerans, temiz kolonlar.
    [GÜNCELLEME]: 300 mum limitini aşmak için Pagination Loop eklendi.
    """
    try:
        resolved_symbol = _resolve_symbol(exchange, symbol)
        if not resolved_symbol:
            return pd.DataFrame()

        all_ohlcv = []
        tf_seconds = exchange.parse_timeframe(timeframe)
        since = exchange.milliseconds() - (limit * tf_seconds * 1000)
        since -= (10 * tf_seconds * 1000)

        retry_count = 0
        
        while len(all_ohlcv) < limit:
            try:
                chunk_size = 100 
                ohlcv = exchange.fetch_ohlcv(resolved_symbol, timeframe, since=int(since), limit=chunk_size)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                last_time = ohlcv[-1][0]
                since = last_time + 1
                
                if last_time >= exchange.milliseconds() - (tf_seconds * 1000):
                    break

                time.sleep(0.1) 
                
                if len(ohlcv) < chunk_size and len(all_ohlcv) < limit:
                    break

            except Exception as e:
                retry_count += 1
                if retry_count > 3:
                    break
                time.sleep(1)
        
        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # Momentum / Osilatör
        df['RSI'] = ta.RSI(close, timeperiod=14)
        df['STOCH_K'], df['STOCH_D'] = ta.STOCH(high, low, close, 14, 3, 3)
        df['WILLR'] = ta.WILLR(high, low, close, timeperiod=14)
        df['MOM'] = ta.MOM(close, timeperiod=10)

        # Trend
        macd, macd_signal, macd_hist = ta.MACD(close, 12, 26, 9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        df['ADX'] = ta.ADX(high, low, close, timeperiod=14)
        df['SAR'] = ta.SAR(high, low, acceleration=0.02, maximum=0.2)

        # Volatilite
        upper, middle, lower = ta.BBANDS(close, timeperiod=20)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = upper, middle, lower
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        df['NATR'] = ta.NATR(high, low, close, timeperiod=14)

        # Ichimoku
        try:
            period_tenkan = 9
            period_kijun = 26
            period_span_b = 52
            highest_high_tenkan = high.rolling(window=period_tenkan).max()
            lowest_low_tenkan = low.rolling(window=period_tenkan).min()
            tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2.0
            df['ICHIMOKU_TENKAN'] = tenkan_sen
            highest_high_kijun = high.rolling(window=period_kijun).max()
            lowest_low_kijun = low.rolling(window=period_kijun).min()
            kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2.0
            df['ICHIMOKU_KIJUN'] = kijun_sen
            span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(period_kijun)
            df['ICHIMOKU_SPAN_A'] = span_a
            highest_high_span_b = high.rolling(window=period_span_b).max()
            lowest_low_span_b = low.rolling(window=period_span_b).min()
            span_b = ((highest_high_span_b + lowest_low_span_b) / 2.0).shift(period_kijun)
            df['ICHIMOKU_SPAN_B'] = span_b
        except Exception:
            df['ICHIMOKU_TENKAN'] = np.nan
            df['ICHIMOKU_KIJUN'] = np.nan
            df['ICHIMOKU_SPAN_A'] = np.nan
            df['ICHIMOKU_SPAN_B'] = np.nan

        # Fibonacci
        try:
            fib_window = 50
            max_high = high.rolling(window=fib_window).max()
            min_low = low.rolling(window=fib_window).min()
            diff = max_high - min_low
            df['FIB_0_382'] = max_high - diff * 0.382
            df['FIB_0_618'] = max_high - diff * 0.618
        except Exception:
            df['FIB_0_382'] = np.nan
            df['FIB_0_618'] = np.nan

        # Hacim
        df['OBV'] = ta.OBV(close, volume)
        df['AD'] = ta.AD(high, low, close, volume)

        # Hareketli Ortalamalar
        df['SMA_10'] = ta.SMA(close, timeperiod=10)
        df['EMA_20'] = ta.EMA(close, timeperiod=20)
        df['EMA_50'] = ta.EMA(close, timeperiod=50)
        df['WMA_50'] = ta.WMA(close, timeperiod=50)
        df['WMA_100'] = ta.WMA(close, timeperiod=100)
        df['HT_TRENDLINE'] = ta.HT_TRENDLINE(close)

        # NaN temizliği
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.dropna(inplace=True)

        try:
            q1, q3 = np.nanpercentile(df['close'], [5, 95])
            if q1 > 0 and q3 > 0 and q3 > q1:
                cap_low = q1 * 0.1
                cap_high = q3 * 10.0
                df = df[(df['close'] >= cap_low) & (df['close'] <= cap_high)]
        except Exception:
            pass
        
        df = analyze_data_with_zscore(df)
        return df

    except Exception as e:
        return pd.DataFrame()


def analyze_data_with_zscore(df: pd.DataFrame, window=50, temperature=2.0) -> pd.DataFrame:
    """Z-Score + Temperature Scaling."""
    if df.empty:
        return df

    targets = ['RSI', 'MOM', 'ADX', 'ATR', 'MACD_Hist', 'WILLR']
    
    for col in targets:
        if col not in df.columns:
            continue
        
        roll_mean = df[col].rolling(window=window).mean()
        roll_std = df[col].rolling(window=window).std()
        
        z_col_name = f"{col}_z"
        df[z_col_name] = (df[col] - roll_mean) / (roll_std + 1e-9)
        
        norm_col_name = f"{col}_norm"
        df[norm_col_name] = 1.0 / (1.0 + np.exp(-df[z_col_name] / temperature))
        
        df[z_col_name] = df[z_col_name].fillna(0.0)
        df[norm_col_name] = df[norm_col_name].fillna(0.5)

    return df


def get_multi_timeframe_analysis(exchange, symbol):
    """Çoklu timeframe analiz: 5m, 15m, 1h, 4h"""
    timeframes = ['5m', '15m', '1h', '4h']
    multi_data = {}
    for tf in timeframes:
        df = fetch_and_analyze_data(exchange, symbol, timeframe=tf, limit=1500)
        if df.empty:
            continue
        df.columns = [f"{col}_{tf}" for col in df.columns]
        multi_data[tf] = df
    return multi_data


async def _maybe_await(x):
    """Await helper for ccxt.async_support compatibility."""
    if inspect.isawaitable(x):
        return await x
    return x


def calc_vwap(high, low, close, volume):
    """VWAP hesaplama yardımcı fonksiyonu."""
    try:
        typical_price = (high + low + close) / 3.0
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)
        cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
        return cumulative_tp_vol / cumulative_vol
    except Exception:
        return np.full_like(close, np.nan)



def _compute_indicators_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same indicator columns as fetch_and_analyze_data_async to an OHLCV dataframe.

    Expects columns: timestamp, open, high, low, close, volume
    """
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values

        # RSI
        df['RSI'] = ta.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        # EMA
        df['EMA_20'] = ta.EMA(close, timeperiod=20)
        df['EMA_50'] = ta.EMA(close, timeperiod=50)

        # ATR
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)

        # ADX
        df['ADX'] = ta.ADX(high, low, close, timeperiod=14)

        # Bollinger
        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # VWAP
        df['VWAP'] = calc_vwap(high, low, close, volume)

        # pivot
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='all')
        return df
    except Exception:
        return df


async def _fetch_ohlcv_paginated(exchange, resolved_symbol: str, timeframe: str, limit: int = 1500):
    """Fetch OHLCV with pagination (since) to reliably fill long lookbacks."""
    import inspect, asyncio, ccxt
    out = []
    since = None
    remaining = int(limit)
    # OKX maximum per call is typically 300/1000 depending; we keep chunk modest
    chunk = min(300, max(100, limit))
    tries = 0
    while remaining > 0:
        per = min(chunk, remaining)
        tries += 1
        if tries > 20:
            break
        try:
            if inspect.iscoroutinefunction(getattr(exchange, 'fetch_ohlcv', None)):
                data = await exchange.fetch_ohlcv(resolved_symbol, timeframe=timeframe, since=since, limit=per)
            else:
                data = await asyncio.to_thread(exchange.fetch_ohlcv, resolved_symbol, timeframe, since, per)
        except Exception as e:
            # retry a couple times on network/limit errors
            msg = str(e).lower()
            if tries < 4 and ('timeout' in msg or 'rate' in msg or '429' in msg or 'too many' in msg):
                await asyncio.sleep(0.6 * tries)
                continue
            break
        if not data:
            break
        out.extend(data)
        remaining = limit - len(out)
        # advance since by last candle + 1ms to avoid duplicates
        try:
            since = int(data[-1][0]) + 1
        except Exception:
            since = None
        # if returned less than requested, stop
        if len(data) < per:
            break
        # small pacing
        await asyncio.sleep(0)
    # De-duplicate by timestamp
    if not out:
        return []
    seen = set()
    dedup = []
    for row in out:
        try:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
        except Exception:
            pass
        dedup.append(row)
    dedup.sort(key=lambda r: r[0])
    # keep last 'limit'
    if len(dedup) > limit:
        dedup = dedup[-limit:]
    return dedup

# ==========================================================================
# [FIX] ASYNC VERSİYONU - _resolve_symbol EKLENDİ
# ==========================================================================
async def fetch_and_analyze_data_async(exchange, symbol, timeframe='15m', limit=1500):
    """Async-safe version of fetch_and_analyze_data.

    Works with both sync ccxt exchanges and ccxt.async_support exchanges.
    
    [FIX] _resolve_symbol eklendi - OKX SWAP sembollerini doğru çözümler.
    """
    try:
        # Optional debug counters (guarded)
        if not hasattr(fetch_and_analyze_data_async, '_call_count'):
            fetch_and_analyze_data_async._call_count = 0
        fetch_and_analyze_data_async._call_count += 1

        if fetch_and_analyze_data_async._call_count <= 5 and DEBUG_MTF:
            print(f'[ASYNC_DEBUG] fetch_and_analyze_data_async çağrıldı: {symbol}, tf={timeframe}')

        # [FIX] Sembolü OKX formatına çevir
        resolved_symbol = _resolve_symbol(exchange, symbol)
        if not resolved_symbol:
            log.warning(f'[ASYNC_FETCH] {symbol} için OKX market bulunamadı, atlanıyor.')
            return pd.DataFrame()

        # Debug log (ilk birkaç sembol için)
        if DEBUG_MTF:
            if not hasattr(fetch_and_analyze_data_async, '_debug_count'):
                fetch_and_analyze_data_async._debug_count = 0
            if fetch_and_analyze_data_async._debug_count < 5:
                log.info(f'[ASYNC_FETCH] {symbol} -> resolved: {resolved_symbol} | tf={timeframe}')
                fetch_and_analyze_data_async._debug_count += 1

        async def _fetch_ohlcv_with_retry():
            max_tries = 4
            delay = 0.6
            for attempt in range(1, max_tries + 1):
                try:
                    # Support both sync ccxt and ccxt.async_support
                    if inspect.iscoroutinefunction(getattr(exchange, 'fetch_ohlcv', None)):
                        return await exchange.fetch_ohlcv(resolved_symbol, timeframe=timeframe, limit=limit)
                    return await asyncio.to_thread(exchange.fetch_ohlcv, resolved_symbol, timeframe, None, limit)
                except Exception as e:
                    msg = str(e).lower()
                    retryable = (
                        isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable))
                        or 'timeout' in msg
                        or 'timed out' in msg
                        or 'connection reset' in msg
                        or 'temporarily unavailable' in msg
                        or '429' in msg
                        or 'rate limit' in msg
                        or 'too many requests' in msg
                        or 'frequency limit' in msg
                    )
                    if retryable and attempt < max_tries:
                        await asyncio.sleep(delay)
                        delay = min(delay * 1.7, 6.0)
                        continue
                    # Final failure: log once
                    log.warning(f"[ASYNC_FETCH] {symbol} ({timeframe}) OHLCV fetch failed: {e}")
                    return None
            return None

        ohlcv = await _fetch_ohlcv_with_retry()

        if not ohlcv:
            log.warning(f"[ASYNC_FETCH] {symbol} ({timeframe}) boş OHLCV döndü")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        volume = df['volume'].astype(float).values

        # RSI
        df['RSI'] = ta.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        # EMA
        df['EMA_20'] = ta.EMA(close, timeperiod=20)
        df['EMA_50'] = ta.EMA(close, timeperiod=50)

        # ATR
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        
        # ADX - trend gücü için önemli
        df['ADX'] = ta.ADX(high, low, close, timeperiod=14)

        # Bollinger
        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower

        # VWAP
        df['VWAP'] = calc_vwap(high, low, close, volume)

        # pivot
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3

        # nan cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='all')
        
        # Başarı logu (ilk 3 sembol için)
        if not hasattr(fetch_and_analyze_data_async, '_success_count'):
            fetch_and_analyze_data_async._success_count = 0
        if fetch_and_analyze_data_async._success_count < 3:
            log.info(f"[ASYNC_SUCCESS] {symbol} | {timeframe}: {len(df)} satır, columns={list(df.columns)[:5]}...")
            fetch_and_analyze_data_async._success_count += 1
        
        return df
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if DEBUG_MTF:
            print(f"[ASYNC_ERROR] {symbol} ({timeframe}): {type(e).__name__}: {e}")
            print(f"[ASYNC_TRACEBACK] {tb[:500]}")
        log.warning(f"[ASYNC_FETCH_ERROR] {symbol} ({timeframe}) analiz hatası: {e}")
        # Full traceback only in debug mode
        if DEBUG_MTF:
            log.debug(tb)
        return pd.DataFrame()


async def get_multi_timeframe_analysis_async(exchange, symbol, timeframes=None, limit=1500):
    """Async-safe multi-timeframe analysis with resample-based MTF.

    Strategy:
      - Fetch a single base timeframe (15m) using pagination to avoid empty multi_data.
      - Resample 15m OHLCV into 1h and 4h (and optionally other higher TFs).
      - Compute indicators on each timeframe.

    Returns: dict[tf] -> DataFrame with columns suffixed by _{tf}
    """
    if not hasattr(get_multi_timeframe_analysis_async, '_call_count'):
        get_multi_timeframe_analysis_async._call_count = 0
    get_multi_timeframe_analysis_async._call_count += 1

    tfs = timeframes or ['15m', '1h', '4h']
    # Normalize list
    tfs = [str(t) for t in tfs if t]

    multi_data = {}

    # Resolve OKX swap symbol once
    resolved_symbol = _resolve_symbol(exchange, symbol)
    if not resolved_symbol:
        log.warning(f"[MTF_EMPTY] {symbol}: market resolve failed")
        return {}

    # Base timeframe is 15m
    base_tf = '15m'
    # Need enough 15m candles to build 4h lookback. 4h = 16 * 15m.
    base_limit = int(max(300, min(1600, int(limit) * 16)))

    try:
        ohlcv = await _fetch_ohlcv_paginated(exchange, resolved_symbol, base_tf, base_limit)
    except Exception:
        ohlcv = []

    if not ohlcv:
        # Fallback: try direct function (may still work in some environments)
        df15 = await fetch_and_analyze_data_async(exchange, symbol, timeframe=base_tf, limit=min(int(limit) * 2, 300))
        if df15 is None or df15.empty:
            log.warning(f"[MTF_EMPTY] {symbol}: base OHLCV empty")
            return {}
    else:
        df15 = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df15['timestamp'] = pd.to_datetime(df15['timestamp'], unit='ms')
        df15 = _compute_indicators_from_df(df15)

    # Helper: suffix all columns
    def _suffix(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [f"{c}_{tf}" for c in df.columns]
        return df

    # Always provide 15m if requested
    if '15m' in tfs or base_tf in tfs:
        multi_data['15m'] = _suffix(df15, '15m')

    # Build 1h/4h from resample
    try:
        base_ohlcv = df15[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        base_ohlcv = base_ohlcv.dropna(subset=['timestamp'])
        base_ohlcv = base_ohlcv.set_index('timestamp')

        def _resample(rule: str):
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }
            df = base_ohlcv.resample(rule).agg(agg).dropna(how='any')
            df = df.reset_index()
            return df

        if '1h' in tfs:
            d1h = _resample('1h')
            d1h = _compute_indicators_from_df(d1h)
            multi_data['1h'] = _suffix(d1h, '1h')

        if '4h' in tfs:
            d4h = _resample('4h')
            d4h = _compute_indicators_from_df(d4h)
            multi_data['4h'] = _suffix(d4h, '4h')

    except Exception as e:
        log.warning(f"[MTF_RESAMPLE_ERR] {symbol}: {e}")

    # Optional lower TF (5m) - fetch directly if requested
    if '5m' in tfs:
        try:
            df5 = await fetch_and_analyze_data_async(exchange, symbol, timeframe='5m', limit=min(int(limit), 300))
            if df5 is not None and not df5.empty:
                multi_data['5m'] = _suffix(df5, '5m')
        except Exception:
            pass

    if not multi_data:
        log.warning(f"[MTF_EMPTY] {symbol}: no timeframe data")
    return multi_data


def check_bbands_squeeze(multi_data, tf='1h', window=50, threshold_percent=0.80):
    if tf not in multi_data or multi_data[tf].empty:
        return False, "HATA: Analiz verisi yok."
    df = multi_data[tf].copy()
    upper_col = f'BB_Upper_{tf}'
    lower_col = f'BB_Lower_{tf}'
    
    if upper_col not in df.columns or lower_col not in df.columns:
        return False, f"BB kolonları ({upper_col}, {lower_col}) bulunamadı."

    df['BB_Width'] = df[upper_col] - df[lower_col]
    rolling_mean_width = df['BB_Width'].rolling(window=window).mean()
    if rolling_mean_width.empty or pd.isna(rolling_mean_width.iloc[-1]):
        return False, "Ortalama genişlik hesaplanamadı."
    current_width = df['BB_Width'].iloc[-1]
    average_width = rolling_mean_width.iloc[-1]
    is_squeezed = current_width < (average_width * threshold_percent)
    if is_squeezed:
        return True, "Bollinger SIKIŞMASI tespit edildi."
    return False, "Sıkışma yok."


def check_smart_take_profit(df, side='long'):
    try:
        macd_hist_col = next((c for c in df.columns if 'MACD_Hist' in c), None)
        
        if not macd_hist_col:
            return False, "MACD_Hist kolonu bulunamadı."

        if len(df) < 2:
            return False, "Yetersiz veri."
            
        last, prev = df.iloc[-1], df.iloc[-2]
        if side == 'long' and last[macd_hist_col] < 0 < prev[macd_hist_col]:
            return True, "MACD Histogram pozitiften negatife döndü → momentum zayıfladı."
        elif side == 'short' and last[macd_hist_col] > 0 > prev[macd_hist_col]:
            return True, "MACD Histogram negatife döndü → momentum zayıfladı."
    except Exception as e:
        return False, f"Analiz hatası: {e}"
    return False, "Aktif çıkış sinyali yok."


# ==========================================================================
# STANDALONE TEST
# ==========================================================================
if __name__ == "__main__":
    import ccxt
    import os
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    load_dotenv()
    
    print("=" * 60)
    print("ANALYZER.PY STANDALONE TEST")
    print("=" * 60)
    
    exchange = ccxt.okx({
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_API_SECRET'),
        'password': os.getenv('OKX_API_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    
    use_testnet = os.getenv('OKX_USE_TESTNET', 'true').lower() == 'true'
    exchange.set_sandbox_mode(use_testnet)
    
    print(f"[TEST] Sandbox mode: {use_testnet}")
    
    exchange.load_markets()
    print(f"[TEST] {len(exchange.markets)} market yüklendi")
    
    test_symbol = "BTC/USDT"
    resolved = _resolve_symbol(exchange, test_symbol)
    print(f"[TEST] {test_symbol} -> resolved: {resolved}")
    
    if resolved:
        print(f"\n[TEST] Senkron fetch_and_analyze_data testi...")
        df_sync = fetch_and_analyze_data(exchange, test_symbol, timeframe='1h', limit=50)
        print(f"[TEST] Senkron sonuç: {len(df_sync)} satır")
        if not df_sync.empty:
            print(f"[TEST] Kolonlar: {list(df_sync.columns)[:10]}...")
        
        print(f"\n[TEST] Async get_multi_timeframe_analysis_async testi...")
        
        async def test_async():
            result = await get_multi_timeframe_analysis_async(exchange, test_symbol, timeframes=['5m', '15m', '1h', '4h'], limit=100)
            return result
        
        multi_data = asyncio.run(test_async())
        print(f"[TEST] Multi-timeframe sonuç: {len(multi_data)} timeframe")
        for tf, df in multi_data.items():
            print(f"  - {tf}: {len(df)} satır")
        
        if multi_data:
            print("\n✅ ASYNC FONKSİYONLAR ÇALIŞIYOR!")
        else:
            print("\n❌ ASYNC FONKSİYONLAR BOŞ DÖNDÜ!")
    else:
        print(f"\n❌ Sembol çözümlenemedi: {test_symbol}")
    
    print("\n" + "=" * 60)

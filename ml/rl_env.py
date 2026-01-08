# ml/rl_env.py
# Multi-coin RL ortamı: build_dataset.py'deki TARGET_COINS ile uyumlu
# 
# CHANGELOG:
# - v1.1: sys.exit(1) replaced with proper exceptions for better error handling
# - v1.2: Added RLEnvironmentError custom exception class
# - v1.3: Improved logging and error messages
#
# Prefer gymnasium if available; fall back to classic gym.  Some
# environments may be installed without gymnasium, and stable-baselines3
# still supports the legacy Gym API.  ImportError is caught to fall back.

from __future__ import annotations

import logging

try:
    import gymnasium as gym  # type: ignore
except Exception:
    try:
        import gym  # type: ignore
    except Exception:
        # Fallback stub definitions when neither gymnasium nor gym are available.
        # This stub provides minimal classes to allow the module to be imported
        # without raising ImportError.  It does not implement any actual
        # functionality but satisfies type and attribute references.
        class _StubDiscrete:
            def __init__(self, n: int, **kwargs):
                self.n = n

        class _StubBox:
            def __init__(self, low, high, shape, dtype, **kwargs):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _StubSpaces:
            Discrete = _StubDiscrete
            Box = _StubBox

        class _StubGym:
            Env = object
            spaces = _StubSpaces()

        gym = _StubGym()  # type: ignore

import numpy as np
import pandas as pd
import pathlib
import json
import re
from typing import Optional, Tuple, Dict, Any

# Setup logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exception Classes
# ---------------------------------------------------------------------------
class RLEnvironmentError(Exception):
    """Custom exception for RL environment errors."""
    pass


class RLDataError(RLEnvironmentError):
    """Exception raised when data is missing or invalid."""
    pass


class RLConfigError(RLEnvironmentError):
    """Exception raised when configuration is invalid."""
    pass


# Aynı coin evrenini kullanmak için build_dataset'ten import
try:
    from build_dataset import TARGET_COINS
except ImportError:
    try:
        from ml.build_dataset import TARGET_COINS
    except ImportError:
        TARGET_COINS = []  # Fallback; yine de çalışsın ama filtre zayıflar.

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics"
OHLC_FILE = METRICS_DIR / "ohlc_history.json"


def _log(msg: str, level: str = "info"):
    """Log message with appropriate level."""
    print(msg)
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)


def _normalize_symbol(s: str) -> str:
    """
    'BTC/USDT:USDT' -> 'BTCUSDT'
    'eth-usdt'      -> 'ETHUSDT'
    """
    if s is None:
        return ""
    s = str(s).upper()
    # Sadece harf ve rakam bırak
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _allowed_symbol_set():
    if not TARGET_COINS:
        # Eğer import edilemediyse, filtreyi çok sıkı yapma.
        return None
    return {_normalize_symbol(sym) for sym in TARGET_COINS}


def _prepare_df_for_env(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    ohlc_history.json içindeki DataFrame'i:
    - ts, symbol, close sütunlarına indirger
    - TARGET_COINS ile filtreler
    - ts'e göre sıralar
    
    Raises:
        RLDataError: When data is empty or required columns are missing
    """
    if raw_df.empty:
        raise RLDataError("ohlc_history.json veri çerçevesi boş.")

    df = raw_df.copy()

    # Sütun adlarını case-insensitive yakala
    cols = {c.lower(): c for c in df.columns}

    def pick_col(candidates):
        for key in candidates:
            if key in cols:
                return cols[key]
        # İçeren sütun ara (örneğin 'closePrice' vb.)
        for c in df.columns:
            cl = c.lower()
            if any(key in cl for key in candidates):
                return c
        return None

    ts_col = pick_col(["ts", "timestamp", "time"])
    sym_col = pick_col(["symbol", "instid", "pair"])
    close_col = pick_col(["close", "price", "last"])

    if ts_col is None or close_col is None:
        raise RLDataError(
            f"'ts' veya 'close' tipi sütun bulunamadı. Mevcut sütunlar: {list(df.columns)}"
        )

    if sym_col is None:
        # Sembol yoksa tek sembol gibi davran
        df["symbol"] = "GENERIC"
    else:
        df.rename(columns={sym_col: "symbol"}, inplace=True)

    # Ts ve close'i sabitle
    df.rename(columns={ts_col: "ts", close_col: "close"}, inplace=True)

    # Gereksiz sütunları at, ama debug için ts/symbol/close kalsın
    keep_cols = [c for c in ["ts", "symbol", "close"] if c in df.columns]
    df = df[keep_cols].copy()

    # Sembol normalize
    df["symbol_norm"] = df["symbol"].apply(_normalize_symbol)

    allow = _allowed_symbol_set()
    if allow is not None and len(allow) > 0:
        before = df["symbol_norm"].nunique()
        df = df[df["symbol_norm"].isin(allow)]
        after = df["symbol_norm"].nunique()
        if df.empty:
            raise RLDataError(
                f"TARGET_COINS filtresinden sonra veri kalmadı. "
                f"Önce {before} sembol vardı, filtre sonrası 0."
            )
        _log(f"RL: Coin filtresi -> önce {before}, sonra {after} sembol.")

    # Artık normalize isim üzerinden ilerleyelim
    df["symbol"] = df["symbol_norm"]
    df.drop(columns=["symbol_norm"], inplace=True)

    # Zaman sıralaması
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "close"])
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # float dönüştür
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    if df.empty:
        raise RLDataError("Temizleme sonrası veri kalmadı.")

    return df


class MultiCoinTradeEnv(gym.Env):
    """
    Çoklu‑coin takas ortamı.  Her episode'da rastgele bir coin seçer ve
    gözlem olarak son ``window`` kapanış serisini normalize eder.  Bu sınıf
    aynı zamanda ödülleri ölçekler ve klipsler.  Normalde PPO'nun öğrenmesi
    için ödüllerin küçük aralıkta olması ve aşırı uçların budanması önemlidir.
    Kullanıcı geribildirimine göre log‑return tabanlı ödül ve değerlerin
    ``max_reward`` ile kısıtlanması daha stabil sonuçlar verir.
    
    Raises:
        RLConfigError: When no coins have enough data for the given window
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        full_df: pd.DataFrame,
        window: int = 60,
        fee: float = 0.0006,
        max_reward: float = 0.05,
    ):
        super().__init__()
        # Komisyon/fee (örn. 0.0006 = %0.06).  Bu değer, pozisyon kapatılırken
        # log‑pnl'den düşülecek.
        self.fee = float(fee)
        # Maksimum mutlak ödül.  Pozisyon kapatıldığında hesaplanan log‑pnl
        # ``[-max_reward, max_reward]`` aralığına sıkıştırılır.  Aşırı uçların
        # model güncellemesini bozmasını engeller.
        self.max_reward = float(max_reward)

        # Coin bazlı gruplama
        grouped = {}
        for sym, g in full_df.groupby("symbol", group_keys=False):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) > window + 2:
                grouped[sym] = g

        if not grouped:
            raise RLConfigError(
                f"window={window}'a göre yeterli uzunlukta hiçbir coin yok. "
                f"Veri setinde {full_df['symbol'].nunique()} sembol var ama "
                f"hiçbiri {window + 2} bar'dan fazla veriye sahip değil."
            )

        self.symbols = sorted(grouped.keys())
        self._groups = grouped
        self.n_symbols = len(self.symbols)

        # Bu, log için tam DF (tüm coinler)
        self.full_df = full_df.copy()

        # Ortam state
        self.window = int(window)
        self.pos = 0          # 0: flat, +1: long, -1: short
        self.entry = None
        self.i = None
        self.current_symbol = None
        self.df = None        # aktif coin DF'si

        # Gym spaces
        self.action_space = gym.spaces.Discrete(3)  # 0:flat, 1:long, 2:short
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window, 1),
            dtype=np.float32,
        )

        _log(f"✅ RL Env hazır: {self.n_symbols} sembol | window={self.window}")
        self.reset()

    # ---------------------- INTERNAL HELPERS ---------------------- #
    def _switch_symbol(self):
        """Episode başlangıcında rastgele bir coin seç."""
        self.current_symbol = np.random.choice(self.symbols)
        self.df = self._groups[self.current_symbol]
        # index pointer
        self.i = self.window

    def _obs(self):
        """Son 'window' close serisini normalize ederek döner."""
        closes = self.df["close"].values
        if self.i <= self.window:
            start = self.window
        else:
            start = self.i
        window_slice = closes[start - self.window : start].astype(np.float32)

        if window_slice.size < self.window:
            # Aşırı edge-case, zero-pad
            padded = np.zeros((self.window,), dtype=np.float32)
            padded[-window_slice.size :] = window_slice
            window_slice = padded

        base = window_slice[0] if window_slice[0] != 0 else (np.mean(window_slice) or 1.0)
        x = window_slice / base - 1.0
        return x.reshape(-1, 1).astype(np.float32)

    # ---------------------- GYM API ---------------------- #
    def reset(self, *, seed=None, options=None):
        """
        Ortamı resetler ve başlangıç gözlemini döner.

        Gymnasium + yeni stable-baselines3 kombinasyonunda reset()'ten
        (obs, info) şeklinde 2'li tuple beklenir.
        Burada:
          - obs  : (window, 1) float32 numpy array
          - info : meta bilgiler için dict (şimdilik sadece 'symbol')
        """
        # Bazı gym/gymnasium sürümlerinde super().reset zorunlu değil, o yüzden try/except
        try:
            super().reset(seed=seed)
        except Exception:
            pass

        self.pos = 0
        self.entry = None
        self._switch_symbol()

        obs = self._obs()
        info = {"symbol": self.current_symbol}
        return obs, info

    def step(self, action):
        """
        Execute one time step within the environment.

        :param action: 0 = flat, 1 = long, 2 = short
        :returns:
            obs        : yeni gözlem (np.array, shape=(window, 1))
            reward     : float
            terminated : episode doğal olarak bitti mi?
            truncated  : zaman sınırı / dışsal kesilme var mı?
            info       : ek bilgiler (ör: current symbol)
        """
        # Eğer index zaten sonu geçtiyse: güvenli early-exit
        if self.i is None or self.i >= len(self.df):
            obs = self._obs()
            info = {"symbol": self.current_symbol}
            # Hem terminated hem truncated True diyebiliriz, zaten episode bitik
            return obs, 0.0, True, True, info

        price = float(self.df.loc[self.i, "close"])
        reward: float = 0.0

        # Pozisyon kapatma.  Long için log(price/entry), short için log(entry/price).
        # Bu log‑return yaklaşımı, ham oran (price/entry - 1) yerine kullanılır ve
        # simetrik/ölçeklenebilir bir ödül sağlar.  Hesaplanan getiri komisyon
        # düşüldükten sonra ``[-max_reward, max_reward]`` aralığına kliplenir.
        if self.pos != 0 and action != (1 if self.pos == 1 else 2):
            if self.entry is not None and self.entry > 0:
                # Log‑return hesapla
                if self.pos == 1:
                    pnl = np.log(max(price, 1e-8) / max(self.entry, 1e-8))
                else:
                    pnl = np.log(max(self.entry, 1e-8) / max(price, 1e-8))
                # Komisyonu düş
                pnl -= self.fee
                # Kliple ve ödüle ekle
                pnl_clipped = np.clip(pnl, -self.max_reward, self.max_reward)
                reward += float(pnl_clipped)
            # Pozisyonu kapat
            self.pos = 0
            self.entry = None

        # Pozisyon açma
        if self.pos == 0 and action in (1, 2):
            self.pos = 1 if action == 1 else -1
            # Yeni entry fiyatını kaydet
            self.entry = price

        # Zamanı ilerlet
        self.i += 1

        # Gymnasium API: terminated / truncated ayrı
        terminated = self.i >= len(self.df) - 1  # son bara gelince doğal bitiş
        truncated = False  # şimdilik time-limit vs. kullanmıyoruz

        obs = self._obs()
        info = {"symbol": self.current_symbol}

        return obs, float(reward), terminated, truncated, info


# === ORTAM YÜKLEYİCİ (TÜM COINLER) ===
def load_env_from_metrics_multi(
    window: int = 60, 
    *, 
    max_reward: float = 0.05,
    raise_on_error: bool = True
) -> Optional[MultiCoinTradeEnv]:
    """
    metrics/ohlc_history.json dosyasından:
    - tüm satırları okur
    - TARGET_COINS ile uyumlu sembolleri filtreler
    - MultiCoinTradeEnv döner
    
    Args:
        window: Lookback window for observations
        max_reward: Maximum absolute reward (for clipping)
        raise_on_error: If True, raise exceptions; if False, return None on error
        
    Returns:
        MultiCoinTradeEnv instance or None if raise_on_error=False and error occurs
        
    Raises:
        RLDataError: When data file is missing or empty (if raise_on_error=True)
        RLConfigError: When configuration is invalid (if raise_on_error=True)
    """
    try:
        if not OHLC_FILE.exists():
            raise RLDataError(f"{OHLC_FILE} bulunamadı.")

        try:
            raw = json.loads(OHLC_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            raise RLDataError(f"JSON okuma hatası: {e}")

        if isinstance(raw, dict) and "rows" in raw:
            rows = raw["rows"]
        else:
            rows = raw

        df = pd.DataFrame(rows)
        if df.empty:
            raise RLDataError("ohlc_history.json boş.")

        df_prepared = _prepare_df_for_env(df)

        coin_count = df_prepared["symbol"].nunique()
        total_rows = len(df_prepared)
        _log(f"✅ RL: {coin_count} sembol için {total_rows} satır hazır (window={window}).")

        return MultiCoinTradeEnv(df_prepared, window=window, fee=0.0006, max_reward=max_reward)
        
    except (RLDataError, RLConfigError) as e:
        _log(f"❌ RL: {e}", level="error")
        if raise_on_error:
            raise
        return None
    except Exception as e:
        _log(f"❌ RL: Beklenmeyen hata: {e}", level="error")
        if raise_on_error:
            raise RLEnvironmentError(f"Beklenmeyen hata: {e}")
        return None

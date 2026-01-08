# -*- coding: utf-8 -*-
"""
AutoTraderBot – Streamlit Dashboard (v7.0 ULTIMATE CONTROL CENTER)

🎛️ TÜM BOT KONTROLÜ TEK PANELDEN:
- Risk Yönetimi (Kill Switch, Circuit Breaker, Max Leverage)
- AI Yapılandırması (Model seçimi, Hybrid weights, Max confidence)
- Trade Parametreleri (Threshold, Position size, Timeframe)
- Model Eğitimi (BiLSTM, RL trigger)
- Canlı Monitoring (Positions, PnL, Logs)
- Calibration & Optimization
- Symbol Management
- API Key Management (masked)
- Alert & Notification Settings

CHANGELOG:
- v7.0: Complete rewrite with full bot control capabilities
- Removed os.system, using subprocess.run for security
- Added real-time config editing
- Added model training triggers
- Added comprehensive risk management panel
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import io
import re
import sys
import time
import json
import math
import glob
import pathlib
import subprocess
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
APP_DIR = pathlib.Path(".").resolve()

try:
    import settings as _settings
    RUNTIME_DIR = pathlib.Path(_settings.RUNTIME_DIR)
    LOG_DIR = pathlib.Path(_settings.LOG_DIR)
    FLAG_KILL = pathlib.Path(_settings.KILL_FLAG)
    FLAG_PAUSE = pathlib.Path(_settings.PAUSE_FLAG)
except Exception:
    RUNTIME_DIR = APP_DIR / "runtime"
    LOG_DIR = APP_DIR / "logs"
    FLAG_KILL = RUNTIME_DIR / "KILL"
    FLAG_PAUSE = RUNTIME_DIR / "PAUSE"

METRICS_DIR = APP_DIR / "metrics"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
ML_DIR = APP_DIR / "ml"

for p in (RUNTIME_DIR, LOG_DIR, METRICS_DIR, MODELS_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Config files
CONFIG_FILE = APP_DIR / "config.json"
SETTINGS_FILE = APP_DIR / "settings.py"
PROMPTS_FILE = APP_DIR / "prompts.json"
CALIBRATION_FILE = APP_DIR / "calibration.json"
RISK_SCHEDULE_FILE = APP_DIR / "risk_schedule.json"
RISK_CONFIG_FILE = APP_DIR / "risk_config.json"
SYMBOLS_FILE = APP_DIR / "symbols_okx.json"
COOLDOWNS_FILE = APP_DIR / "cooldowns.json"

# Metrics files
OHLC_HISTORY_FILE = METRICS_DIR / "ohlc_history.json"
AI_PRED_FILE = METRICS_DIR / "ai_predictions.json"
SIGNALS_FILE = METRICS_DIR / "signals_last.json"
OPEN_POS_FILE = METRICS_DIR / "open_positions.json"
BILSTM_METRICS_FILE = METRICS_DIR / "bilstm_metrics.json"
RL_METRICS_FILE = METRICS_DIR / "rl_metrics.json"
TRADE_LOG_FILE = DATA_DIR / "trade_log.json"

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def safe_load_json(path: pathlib.Path, default=None):
    """Safely load JSON file with error handling."""
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"JSON yükleme hatası ({path.name}): {e}")
    return default


def safe_save_json(path: pathlib.Path, data, backup: bool = True):
    """Safely save JSON file with optional backup."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        if backup and path.exists():
            backup_path = path.with_suffix(f".{int(time.time())}.bak")
            try:
                import shutil
                shutil.copy2(path, backup_path)
            except Exception:
                pass
        
        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        # Replace original
        if os.name == 'nt':  # Windows
            if path.exists():
                path.unlink()
        os.replace(str(tmp_path), str(path))
        return True
    except Exception as e:
        st.error(f"JSON kaydetme hatası: {e}")
        return False


def tail_file(path: pathlib.Path, max_bytes: int = 200_000) -> str:
    """Read last portion of a file."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            content = f.read()
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def list_log_files() -> List[str]:
    """List all log files sorted by modification time."""
    try:
        return sorted(
            [p.name for p in LOG_DIR.glob("*.log")],
            key=lambda n: (LOG_DIR / n).stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return []


def human_time_ago(ts_str: str) -> str:
    """Convert timestamp to human-readable 'time ago' format."""
    try:
        if not ts_str:
            return "N/A"
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        
        if delta.days > 0:
            return f"{delta.days}d önce"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h önce"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m önce"
        else:
            return f"{delta.seconds}s önce"
    except Exception:
        return "N/A"


def run_subprocess_safe(cmd: List[str], timeout: int = 60) -> Tuple[bool, str]:
    """Run subprocess safely without os.system."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(APP_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Timeout expired"
    except Exception as e:
        return False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "auto_refresh_sec": 30,
    "theme": "dark",
    
    # AI Configuration
    "ai_mode": "hybrid",
    "openai_model": "gpt-4o-mini",
    "deepseek_model": "deepseek-chat",
    "max_ai_confidence": 0.85,
    
    # Hybrid Weights
    "hybrid_weights": {
        "chatgpt": 0.40,
        "deepseek": 0.40,
        "bilstm": 0.10,
        "ppo_rl": 0.10
    },
    
    # Decision Weights
    "decision_weights": {
        "ai": 0.50,
        "tech": 0.40,
        "sent": 0.10
    },
    
    # Risk Configuration
    "risk": {
        "max_leverage": 25,
        "max_position_pct": 0.10,
        "max_portfolio_risk_pct": 0.05,
        "max_drawdown_alert": 0.15,
        "margin_ratio_warn": 0.50,
        "daily_loss_limit_pct": 0.05,
        "api_429_rate_warn": 0.05
    },
    
    # Circuit Breaker
    "circuit_breaker": {
        "error_threshold": 5,
        "anomaly_threshold": 5,
        "cooldown_seconds": 900,
        "large_loss_threshold": -0.05
    },
    
    # Kill Switch Thresholds
    "kill_switch": {
        "reduce_threshold": -0.03,
        "halt_threshold": -0.05,
        "stop_threshold": -0.07
    },
    
    # Trading Parameters
    "trading": {
        "min_confidence_threshold": 0.70,
        "default_timeframe": "15m",
        "max_open_positions": 5,
        "cooldown_minutes": 30,
        "enable_shorts": True,
        "enable_dca": False
    },
    
    # Calibration
    "calibration": {
        "a": 3.0,
        "b": 0.0
    },
    
    # Token Costs (per 1k tokens)
    "token_cost_per_1k": {
        "gpt-4o-mini": 0.00038,
        "gpt-4o": 0.010,
        "deepseek-chat": 0.00027,
        "deepseek-reasoner": 0.0010
    },
    
    # Notifications
    "notifications": {
        "telegram_enabled": False,
        "discord_enabled": False,
        "email_enabled": False,
        "telegram_chat_id": "",
        "discord_webhook": ""
    }
}


def ensure_config():
    """Ensure config.json exists with all required fields."""
    cfg = safe_load_json(CONFIG_FILE, {})
    changed = False
    
    def deep_update(base: dict, updates: dict) -> bool:
        modified = False
        for k, v in updates.items():
            if k not in base:
                base[k] = v
                modified = True
            elif isinstance(v, dict) and isinstance(base.get(k), dict):
                if deep_update(base[k], v):
                    modified = True
        return modified
    
    if deep_update(cfg, DEFAULT_CONFIG):
        safe_save_json(CONFIG_FILE, cfg, backup=False)
    
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🤖 AutoTraderBot Control Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
APP_CFG = ensure_config()

# Theme
if APP_CFG.get("theme", "dark") == "dark":
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .metric-card { 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 10px; padding: 15px; margin: 5px 0;
            border: 1px solid #30363d;
        }
        .status-ok { color: #00ff88; }
        .status-warn { color: #ffaa00; }
        .status-error { color: #ff4444; }
        .control-section {
            background: #161b22; border-radius: 10px;
            padding: 20px; margin: 10px 0;
            border: 1px solid #30363d;
        }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - BOT STATUS & QUICK CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
    st.title("🤖 Bot Kontrol")
    
    # Bot Status
    st.markdown("### 📊 Durum")
    
    is_killed = FLAG_KILL.exists()
    is_paused = FLAG_PAUSE.exists()
    
    if is_killed:
        st.error("🔴 BOT DURDURULDU (KILL)")
    elif is_paused:
        st.warning("⏸️ BOT DURAKLATILDI")
    else:
        st.success("🟢 BOT AKTİF")
    
    # Quick Controls
    st.markdown("### ⚡ Hızlı Kontrol")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ BAŞLAT", use_container_width=True, type="primary"):
            FLAG_KILL.unlink(missing_ok=True)
            FLAG_PAUSE.unlink(missing_ok=True)
            st.success("Bot başlatıldı!")
            st.rerun()
    
    with col2:
        if st.button("⏸️ DURAKLAT", use_container_width=True):
            FLAG_PAUSE.touch()
            st.warning("Bot duraklatıldı")
            st.rerun()
    
    if st.button("🛑 ACİL DURDUR (KILL)", use_container_width=True, type="secondary"):
        FLAG_KILL.touch()
        st.error("Bot durduruldu!")
        st.rerun()
    
    st.markdown("---")
    
    # Auto Refresh
    auto_refresh = st.checkbox(
        "🔄 Otomatik Yenile",
        value=True,
        help="Dashboard'u otomatik yeniler"
    )
    
    refresh_interval = st.slider(
        "Yenileme Aralığı (sn)",
        min_value=10,
        max_value=120,
        value=APP_CFG.get("auto_refresh_sec", 30)
    )
    
    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_interval}'>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Özet")
    
    # Load metrics
    open_positions = safe_load_json(OPEN_POS_FILE, [])
    ai_predictions = safe_load_json(AI_PRED_FILE, [])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Açık Pozisyon", len(open_positions))
    with col2:
        st.metric("AI Tahmin", len(ai_predictions))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Genel Bakış",
    "⚠️ Risk Yönetimi",
    "🤖 AI Yapılandırma",
    "📊 Trade Parametreleri",
    "🎯 Calibration",
    "📈 Pozisyonlar",
    "📜 Loglar",
    "🔧 Model Eğitimi",
    "🎛️ Gelişmiş Ayarlar",
    "💰 AI Maliyet"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0: GENEL BAKIŞ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🏠 Genel Bakış")
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Load various metrics
    trade_log = safe_load_json(TRADE_LOG_FILE, [])
    bilstm_metrics = safe_load_json(BILSTM_METRICS_FILE, {})
    rl_metrics = safe_load_json(RL_METRICS_FILE, {})
    
    with col1:
        st.metric(
            "Toplam Trade",
            len(trade_log),
            help="Tüm zamanlar"
        )
    
    with col2:
        wins = sum(1 for t in trade_log if t.get("pnl_pct", 0) > 0)
        total = len(trade_log) if trade_log else 1
        win_rate = (wins / total) * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{wins} kazanç"
        )
    
    with col3:
        total_pnl = sum(t.get("pnl_pct", 0) for t in trade_log)
        st.metric(
            "Toplam PnL",
            f"{total_pnl:.2f}%",
            delta="Kümülatif"
        )
    
    with col4:
        bilstm_acc = bilstm_metrics.get("accuracy", "N/A")
        if isinstance(bilstm_acc, (int, float)):
            st.metric("BiLSTM Acc", f"{bilstm_acc:.1%}")
        else:
            st.metric("BiLSTM Acc", "N/A")
    
    with col5:
        rl_update = rl_metrics.get("last_update", "N/A")
        st.metric("RL Update", human_time_ago(rl_update))
    
    st.markdown("---")
    
    # Two columns: Recent Signals & Open Positions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Son AI Sinyalleri")
        signals = safe_load_json(SIGNALS_FILE, [])
        if signals:
            df_signals = pd.DataFrame(signals[-10:])
            if not df_signals.empty:
                display_cols = ["symbol", "decision", "confidence", "model"]
                display_cols = [c for c in display_cols if c in df_signals.columns]
                st.dataframe(df_signals[display_cols], use_container_width=True)
        else:
            st.info("Henüz sinyal yok")
    
    with col2:
        st.subheader("📊 Açık Pozisyonlar")
        open_pos = safe_load_json(OPEN_POS_FILE, [])
        if open_pos:
            df_pos = pd.DataFrame(open_pos)
            if not df_pos.empty:
                display_cols = ["symbol", "side", "size", "entry_price", "unrealized_pnl"]
                display_cols = [c for c in display_cols if c in df_pos.columns]
                st.dataframe(df_pos[display_cols] if display_cols else df_pos, use_container_width=True)
        else:
            st.info("Açık pozisyon yok")
    
    st.markdown("---")
    
    # Master Confidence Chart
    st.subheader("📊 Master Confidence Dağılımı")
    if signals:
        df_conf = pd.DataFrame(signals[-50:])
        if "confidence" in df_conf.columns and "symbol" in df_conf.columns:
            fig = px.bar(
                df_conf,
                x="symbol",
                y="confidence",
                color="confidence",
                color_continuous_scale="RdYlGn",
                title="Son 50 Sinyal - Master Confidence"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RİSK YÖNETİMİ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("⚠️ Risk Yönetimi Kontrol Paneli")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    risk_cfg = cfg.get("risk", DEFAULT_CONFIG["risk"])
    cb_cfg = cfg.get("circuit_breaker", DEFAULT_CONFIG["circuit_breaker"])
    ks_cfg = cfg.get("kill_switch", DEFAULT_CONFIG["kill_switch"])
    
    # Circuit Breaker Status
    st.subheader("🔌 Circuit Breaker Durumu")
    
    try:
        from circuit_breaker import get_state, reset as cb_reset
        cb_state = get_state()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if cb_state["is_triggered"]:
                st.error(f"🔴 AKTİF - {cb_state['remaining_cooldown_sec']}s kaldı")
            else:
                st.success("🟢 Normal")
        with col2:
            st.metric("Error Count", f"{cb_state['error_count']}/{cb_state['thresholds']['error']}")
        with col3:
            st.metric("Anomaly Count", f"{cb_state['anomaly_count']}/{cb_state['thresholds']['anomaly']}")
        with col4:
            st.metric("Toplam Trigger", cb_state['total_triggers'])
        
        if st.button("🔄 Circuit Breaker Sıfırla"):
            cb_reset()
            st.success("Circuit breaker sıfırlandı!")
            st.rerun()
    except Exception as e:
        st.warning(f"Circuit breaker modülü yüklenemedi: {e}")
    
    st.markdown("---")
    
    # Risk Parameters
    st.subheader("📊 Risk Parametreleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎚️ Temel Risk Limitleri")
        
        new_max_leverage = st.slider(
            "Maksimum Kaldıraç",
            min_value=1,
            max_value=50,
            value=int(risk_cfg.get("max_leverage", 25)),
            help="İzin verilen maksimum kaldıraç (Önerilen: 25x)"
        )
        
        new_max_position = st.slider(
            "Maksimum Pozisyon (%)",
            min_value=1,
            max_value=50,
            value=int(risk_cfg.get("max_position_pct", 10) * 100),
            help="Bakiyenin yüzde kaçı tek pozisyona ayrılabilir"
        ) / 100
        
        new_portfolio_risk = st.slider(
            "Portföy Risk Limiti (%)",
            min_value=1,
            max_value=20,
            value=int(risk_cfg.get("max_portfolio_risk_pct", 5) * 100),
            help="Toplam açık risk limiti"
        ) / 100
        
        new_daily_loss = st.slider(
            "Günlük Kayıp Limiti (%)",
            min_value=1,
            max_value=20,
            value=int(risk_cfg.get("daily_loss_limit_pct", 5) * 100),
            help="Günlük maksimum kayıp"
        ) / 100
    
    with col2:
        st.markdown("#### 🛑 Kill Switch Eşikleri")
        
        new_reduce = st.slider(
            "Reduce Eşiği (%)",
            min_value=-20,
            max_value=0,
            value=int(ks_cfg.get("reduce_threshold", -3) * 100),
            help="Bu kayıpta pozisyon küçültülür"
        ) / 100
        
        new_halt = st.slider(
            "Halt Eşiği (%)",
            min_value=-30,
            max_value=0,
            value=int(ks_cfg.get("halt_threshold", -5) * 100),
            help="Bu kayıpta yeni pozisyon açılmaz"
        ) / 100
        
        new_stop = st.slider(
            "Stop Eşiği (%)",
            min_value=-50,
            max_value=0,
            value=int(ks_cfg.get("stop_threshold", -7) * 100),
            help="Bu kayıpta tüm pozisyonlar kapatılır"
        ) / 100
        
        st.markdown("#### 🔌 Circuit Breaker")
        
        new_error_threshold = st.number_input(
            "Error Threshold",
            min_value=1,
            max_value=20,
            value=int(cb_cfg.get("error_threshold", 5)),
            help="Kaç ardışık hatada breaker tetiklenir"
        )
        
        new_cooldown = st.number_input(
            "Cooldown (saniye)",
            min_value=60,
            max_value=3600,
            value=int(cb_cfg.get("cooldown_seconds", 900)),
            help="Breaker tetiklendikten sonra bekleme süresi"
        )
    
    if st.button("💾 Risk Ayarlarını Kaydet", type="primary"):
        cfg["risk"]["max_leverage"] = new_max_leverage
        cfg["risk"]["max_position_pct"] = new_max_position
        cfg["risk"]["max_portfolio_risk_pct"] = new_portfolio_risk
        cfg["risk"]["daily_loss_limit_pct"] = new_daily_loss
        cfg["kill_switch"]["reduce_threshold"] = new_reduce
        cfg["kill_switch"]["halt_threshold"] = new_halt
        cfg["kill_switch"]["stop_threshold"] = new_stop
        cfg["circuit_breaker"]["error_threshold"] = new_error_threshold
        cfg["circuit_breaker"]["cooldown_seconds"] = new_cooldown
        
        if safe_save_json(CONFIG_FILE, cfg):
            st.success("✅ Risk ayarları kaydedildi!")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: AI YAPILANDIRMA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🤖 AI Yapılandırma")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    hybrid_w = cfg.get("hybrid_weights", DEFAULT_CONFIG["hybrid_weights"])
    decision_w = cfg.get("decision_weights", DEFAULT_CONFIG["decision_weights"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Seçimi")
        
        ai_mode = st.selectbox(
            "AI Modu",
            options=["hybrid", "chatgpt_only", "deepseek_only", "bilstm_only", "disabled"],
            index=["hybrid", "chatgpt_only", "deepseek_only", "bilstm_only", "disabled"].index(
                cfg.get("ai_mode", "hybrid")
            ),
            help="Hangi AI modellerinin kullanılacağı"
        )
        
        openai_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"].index(
                cfg.get("openai_model", "gpt-4o-mini")
            ) if cfg.get("openai_model", "gpt-4o-mini") in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"] else 0
        )
        
        deepseek_model = st.selectbox(
            "DeepSeek Model",
            options=["deepseek-chat", "deepseek-reasoner"],
            index=["deepseek-chat", "deepseek-reasoner"].index(
                cfg.get("deepseek_model", "deepseek-chat")
            ) if cfg.get("deepseek_model", "deepseek-chat") in ["deepseek-chat", "deepseek-reasoner"] else 0
        )
        
        max_ai_conf = st.slider(
            "Maksimum AI Confidence",
            min_value=0.50,
            max_value=1.0,
            value=float(cfg.get("max_ai_confidence", 0.85)),
            step=0.05,
            help="AI'ın verebileceği maksimum güven skoru"
        )
    
    with col2:
        st.subheader("⚖️ Hybrid Ağırlıklar")
        
        st.info("Toplamı 1.0 olmalı")
        
        w_chatgpt = st.slider(
            "ChatGPT Ağırlığı",
            min_value=0.0,
            max_value=1.0,
            value=float(hybrid_w.get("chatgpt", 0.4)),
            step=0.05
        )
        
        w_deepseek = st.slider(
            "DeepSeek Ağırlığı",
            min_value=0.0,
            max_value=1.0,
            value=float(hybrid_w.get("deepseek", 0.4)),
            step=0.05
        )
        
        w_bilstm = st.slider(
            "BiLSTM Ağırlığı",
            min_value=0.0,
            max_value=1.0,
            value=float(hybrid_w.get("bilstm", 0.1)),
            step=0.05
        )
        
        w_rl = st.slider(
            "PPO-RL Ağırlığı",
            min_value=0.0,
            max_value=1.0,
            value=float(hybrid_w.get("ppo_rl", 0.1)),
            step=0.05
        )
        
        total_w = w_chatgpt + w_deepseek + w_bilstm + w_rl
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ Toplam ağırlık: {total_w:.2f} (1.0 olmalı)")
        else:
            st.success(f"✅ Toplam: {total_w:.2f}")
    
    st.markdown("---")
    
    st.subheader("🎯 Karar Ağırlıkları (AI/Tech/Sent)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        d_ai = st.slider("AI Ağırlığı", 0.0, 1.0, float(decision_w.get("ai", 0.5)), 0.05)
    with col2:
        d_tech = st.slider("Teknik Ağırlığı", 0.0, 1.0, float(decision_w.get("tech", 0.4)), 0.05)
    with col3:
        d_sent = st.slider("Sentiment Ağırlığı", 0.0, 1.0, float(decision_w.get("sent", 0.1)), 0.05)
    
    total_d = d_ai + d_tech + d_sent
    if abs(total_d - 1.0) > 0.01:
        st.warning(f"⚠️ Toplam: {total_d:.2f}")
    
    if st.button("💾 AI Ayarlarını Kaydet", type="primary"):
        cfg["ai_mode"] = ai_mode
        cfg["openai_model"] = openai_model
        cfg["deepseek_model"] = deepseek_model
        cfg["max_ai_confidence"] = max_ai_conf
        cfg["hybrid_weights"] = {
            "chatgpt": w_chatgpt,
            "deepseek": w_deepseek,
            "bilstm": w_bilstm,
            "ppo_rl": w_rl
        }
        cfg["decision_weights"] = {
            "ai": d_ai,
            "tech": d_tech,
            "sent": d_sent
        }
        
        if safe_save_json(CONFIG_FILE, cfg):
            st.success("✅ AI ayarları kaydedildi!")
            st.rerun()
    
    st.markdown("---")
    
    # LLM Prompt Editor
    st.subheader("📝 LLM Sistem Prompt Editörü")
    
    prompts = safe_load_json(PROMPTS_FILE, {})
    current_prompt = prompts.get("trading_system_prompt", "")
    
    if not current_prompt:
        current_prompt = """You are an institutional-grade quantitative crypto futures strategist.

Goal:
- Evaluate long/short opportunities on OKX USDT perpetual swaps.
- Focus on *risk-adjusted* returns and capital preservation.
- Be conservative by default.

Rules:
- Never exceed {max_confidence} on master_confidence.
- If the setup is mediocre, prefer "skip" or "hold".
- Return strict JSON format."""
    
    new_prompt = st.text_area(
        "Sistem Prompt",
        value=current_prompt,
        height=300,
        help="ChatGPT ve DeepSeek için kullanılan sistem promptu"
    )
    
    if st.button("💾 Prompt Kaydet"):
        prompts["trading_system_prompt"] = new_prompt
        if safe_save_json(PROMPTS_FILE, prompts, backup=True):
            st.success("✅ Prompt kaydedildi!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TRADE PARAMETRELERİ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📊 Trade Parametreleri")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    trading_cfg = cfg.get("trading", DEFAULT_CONFIG["trading"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Giriş Kriterleri")
        
        min_conf = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.50,
            max_value=0.95,
            value=float(trading_cfg.get("min_confidence_threshold", 0.70)),
            step=0.01,
            help="Bu seviyenin altında trade açılmaz"
        )
        
        default_tf = st.selectbox(
            "Varsayılan Timeframe",
            options=["5m", "15m", "1h", "4h"],
            index=["5m", "15m", "1h", "4h"].index(trading_cfg.get("default_timeframe", "15m"))
        )
        
        max_positions = st.number_input(
            "Maksimum Açık Pozisyon",
            min_value=1,
            max_value=20,
            value=int(trading_cfg.get("max_open_positions", 5))
        )
        
        cooldown_min = st.number_input(
            "Cooldown (dakika)",
            min_value=5,
            max_value=120,
            value=int(trading_cfg.get("cooldown_minutes", 30)),
            help="Aynı sembolde yeniden işlem için bekleme"
        )
    
    with col2:
        st.subheader("⚙️ Trade Özellikleri")
        
        enable_shorts = st.checkbox(
            "Short Pozisyonları Aktif",
            value=trading_cfg.get("enable_shorts", True)
        )
        
        enable_dca = st.checkbox(
            "DCA (Dollar Cost Averaging) Aktif",
            value=trading_cfg.get("enable_dca", False)
        )
        
        st.markdown("---")
        
        st.subheader("📋 Sembol Listesi")
        
        symbols = safe_load_json(SYMBOLS_FILE, [])
        
        st.info(f"Aktif sembol sayısı: {len(symbols)}")
        
        symbols_text = st.text_area(
            "Semboller (her satıra bir tane)",
            value="\n".join(symbols),
            height=200
        )
        
        if st.button("📋 Sembolleri Güncelle"):
            new_symbols = [s.strip() for s in symbols_text.split("\n") if s.strip()]
            if safe_save_json(SYMBOLS_FILE, new_symbols):
                st.success(f"✅ {len(new_symbols)} sembol kaydedildi!")
    
    st.markdown("---")
    
    if st.button("💾 Trade Parametrelerini Kaydet", type="primary"):
        cfg["trading"]["min_confidence_threshold"] = min_conf
        cfg["trading"]["default_timeframe"] = default_tf
        cfg["trading"]["max_open_positions"] = max_positions
        cfg["trading"]["cooldown_minutes"] = cooldown_min
        cfg["trading"]["enable_shorts"] = enable_shorts
        cfg["trading"]["enable_dca"] = enable_dca
        
        if safe_save_json(CONFIG_FILE, cfg):
            st.success("✅ Trade parametreleri kaydedildi!")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🎯 Calibration & Optimizasyon")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    cal_cfg = cfg.get("calibration", DEFAULT_CONFIG["calibration"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Sigmoid Calibration")
        
        st.latex(r"calibrated = \frac{1}{1 + e^{-a(raw - 0.5) + b}}")
        
        cal_a = st.slider(
            "a (Eğim)",
            min_value=1.0,
            max_value=10.0,
            value=float(cal_cfg.get("a", 3.0)),
            step=0.5,
            help="Yüksek değer = daha keskin geçiş"
        )
        
        cal_b = st.slider(
            "b (Kayma)",
            min_value=-2.0,
            max_value=2.0,
            value=float(cal_cfg.get("b", 0.0)),
            step=0.1,
            help="Orta noktayı kaydırır"
        )
        
        # Preview chart
        x = np.linspace(0, 1, 100)
        y = 1 / (1 + np.exp(-cal_a * (x - 0.5) + cal_b))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Calibrated'))
        fig.add_trace(go.Scatter(x=x, y=x, mode='lines', name='Linear', line=dict(dash='dash')))
        fig.update_layout(
            title="Calibration Curve Preview",
            xaxis_title="Raw Confidence",
            yaxis_title="Calibrated Confidence",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Risk Schedule")
        
        risk_schedule = safe_load_json(RISK_SCHEDULE_FILE, {
            "tiers": [
                {"min_conf": 0.65, "max_conf": 0.70, "leverage": 8},
                {"min_conf": 0.70, "max_conf": 0.75, "leverage": 12},
                {"min_conf": 0.75, "max_conf": 0.80, "leverage": 18},
                {"min_conf": 0.80, "max_conf": 0.90, "leverage": 25}
            ]
        })
        
        st.markdown("#### Kaldıraç Seviyeleri")
        
        tiers = risk_schedule.get("tiers", [])
        for i, tier in enumerate(tiers):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.text(f"Tier {i+1}")
            with col_b:
                st.text(f"{tier['min_conf']:.0%} - {tier['max_conf']:.0%}")
            with col_c:
                st.text(f"{tier['leverage']}x")
        
        st.markdown("---")
        
        st.subheader("🔧 Cooldown Yönetimi")
        
        cooldowns = safe_load_json(COOLDOWNS_FILE, {})
        
        if cooldowns:
            st.dataframe(pd.DataFrame([
                {"symbol": k, "until": v}
                for k, v in cooldowns.items()
            ]))
            
            if st.button("🗑️ Tüm Cooldown'ları Temizle"):
                if safe_save_json(COOLDOWNS_FILE, {}):
                    st.success("Cooldown'lar temizlendi!")
                    st.rerun()
        else:
            st.info("Aktif cooldown yok")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Calibration Kaydet", type="primary"):
            cfg["calibration"]["a"] = cal_a
            cfg["calibration"]["b"] = cal_b
            
            # Also update calibration.json
            cal_file_data = {
                "type": "logistic",
                "a": cal_a,
                "b": cal_b
            }
            safe_save_json(CALIBRATION_FILE, cal_file_data)
            
            if safe_save_json(CONFIG_FILE, cfg):
                st.success("✅ Calibration ayarları kaydedildi!")
    
    with col2:
        if st.button("🔄 Calibration Scriptini Çalıştır"):
            with st.spinner("Calibration çalışıyor..."):
                success, output = run_subprocess_safe(
                    [sys.executable, "calibrate_confidence.py"],
                    timeout=120
                )
                if success:
                    st.success("✅ Calibration tamamlandı!")
                else:
                    st.error(f"❌ Hata: {output}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: POZİSYONLAR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("📈 Pozisyon Yönetimi")
    
    # Open Positions
    st.subheader("📊 Açık Pozisyonlar")
    open_pos = safe_load_json(OPEN_POS_FILE, [])
    
    if open_pos:
        df_pos = pd.DataFrame(open_pos)
        st.dataframe(df_pos, use_container_width=True)
        
        # PnL Summary
        if "unrealized_pnl" in df_pos.columns:
            total_unrealized = df_pos["unrealized_pnl"].sum()
            st.metric("Toplam Unrealized PnL", f"${total_unrealized:.2f}")
    else:
        st.info("Açık pozisyon bulunmuyor")
    
    st.markdown("---")
    
    # Trade History
    st.subheader("📜 Trade Geçmişi")
    
    trade_log = safe_load_json(TRADE_LOG_FILE, [])
    
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if "symbol" in df_trades.columns:
                symbols_filter = st.multiselect(
                    "Sembol Filtresi",
                    options=df_trades["symbol"].unique().tolist()
                )
        with col2:
            show_last = st.number_input("Son N trade", min_value=10, max_value=500, value=50)
        
        # Apply filters
        df_display = df_trades.tail(show_last)
        if symbols_filter:
            df_display = df_display[df_display["symbol"].isin(symbols_filter)]
        
        st.dataframe(df_display, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Trade", len(df_trades))
        with col2:
            if "pnl_pct" in df_trades.columns:
                wins = len(df_trades[df_trades["pnl_pct"] > 0])
                st.metric("Kazanan", wins)
        with col3:
            if "pnl_pct" in df_trades.columns:
                losses = len(df_trades[df_trades["pnl_pct"] <= 0])
                st.metric("Kaybeden", losses)
        with col4:
            if "pnl_pct" in df_trades.columns:
                avg_pnl = df_trades["pnl_pct"].mean()
                st.metric("Ortalama PnL", f"{avg_pnl:.2f}%")
    else:
        st.info("Trade geçmişi bulunamadı")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: LOGLAR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📜 Log Görüntüleyici")
    
    log_files = list_log_files()
    
    if log_files:
        selected_log = st.selectbox("Log Dosyası Seç", log_files)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_lines = st.slider("Maksimum Satır", 100, 2000, 500)
        with col2:
            if st.button("🔄 Yenile"):
                st.rerun()
        
        log_path = LOG_DIR / selected_log
        log_content = tail_file(log_path, max_bytes=max_lines * 200)
        
        # Filter options
        filter_text = st.text_input("Filtre (regex)", placeholder="ERROR|WARN")
        
        if filter_text:
            try:
                pattern = re.compile(filter_text, re.IGNORECASE)
                lines = log_content.split("\n")
                filtered_lines = [l for l in lines if pattern.search(l)]
                log_content = "\n".join(filtered_lines[-max_lines:])
            except Exception:
                st.warning("Geçersiz regex pattern")
        
        st.code(log_content, language="log")
    else:
        st.info("Log dosyası bulunamadı")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("🔧 Model Eğitimi & Güncelleme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 BiLSTM Model")
        
        bilstm_metrics = safe_load_json(BILSTM_METRICS_FILE, {})
        
        st.metric("Son Güncelleme", human_time_ago(bilstm_metrics.get("last_update", "")))
        st.metric("Accuracy", f"{bilstm_metrics.get('accuracy', 'N/A')}")
        st.metric("Samples", bilstm_metrics.get("samples", "N/A"))
        
        if st.button("🚀 BiLSTM Eğitimini Başlat", type="primary"):
            with st.spinner("BiLSTM eğitimi başlatılıyor..."):
                script_path = ML_DIR / "bilstm_train.py"
                if not script_path.exists():
                    script_path = APP_DIR / "bilstm_train.py"
                
                if script_path.exists():
                    success, output = run_subprocess_safe(
                        [sys.executable, str(script_path)],
                        timeout=600
                    )
                    if success:
                        st.success("✅ BiLSTM eğitimi tamamlandı!")
                    else:
                        st.error(f"❌ Hata: {output[:500]}")
                else:
                    st.error("bilstm_train.py bulunamadı!")
    
    with col2:
        st.subheader("🎮 RL (PPO) Model")
        
        rl_metrics = safe_load_json(RL_METRICS_FILE, {})
        
        st.metric("Son Güncelleme", human_time_ago(rl_metrics.get("last_update", "")))
        st.metric("Timesteps", rl_metrics.get("timesteps", "N/A"))
        st.metric("Window", rl_metrics.get("window", "N/A"))
        
        if st.button("🚀 RL Eğitimini Başlat", type="primary"):
            with st.spinner("RL eğitimi başlatılıyor..."):
                script_path = ML_DIR / "rl_train.py"
                if not script_path.exists():
                    script_path = APP_DIR / "rl_train.py"
                
                if script_path.exists():
                    success, output = run_subprocess_safe(
                        [sys.executable, str(script_path), "--timesteps", "50000"],
                        timeout=1800
                    )
                    if success:
                        st.success("✅ RL eğitimi tamamlandı!")
                    else:
                        st.error(f"❌ Hata: {output[:500]}")
                else:
                    st.error("rl_train.py bulunamadı!")
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Oluşturma")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ds_window = st.number_input("Window", min_value=30, max_value=300, value=180)
    with col2:
        ds_horizon = st.number_input("Horizon", min_value=1, max_value=50, value=12)
    with col3:
        ds_threshold = st.number_input("Threshold", min_value=0.001, max_value=0.01, value=0.002, format="%.3f")
    
    if st.button("📊 Dataset Oluştur"):
        with st.spinner("Dataset oluşturuluyor..."):
            script_path = ML_DIR / "build_dataset.py"
            if not script_path.exists():
                script_path = APP_DIR / "build_dataset.py"
            
            if script_path.exists():
                success, output = run_subprocess_safe(
                    [sys.executable, str(script_path),
                     "--window", str(ds_window),
                     "--horizon", str(ds_horizon),
                     "--threshold", str(ds_threshold)],
                    timeout=300
                )
                if success:
                    st.success("✅ Dataset oluşturuldu!")
                else:
                    st.error(f"❌ Hata: {output[:500]}")
            else:
                st.error("build_dataset.py bulunamadı!")
    
    st.markdown("---")
    
    st.subheader("🔄 Auto Updater")
    
    if st.button("▶️ Auto Update Cycle Çalıştır"):
        with st.spinner("Auto updater çalışıyor..."):
            script_path = APP_DIR / "auto_updater.py"
            if script_path.exists():
                success, output = run_subprocess_safe(
                    [sys.executable, str(script_path)],
                    timeout=3600
                )
                if success:
                    st.success("✅ Auto update tamamlandı!")
                else:
                    st.error(f"❌ Hata: {output[:500]}")
            else:
                st.error("auto_updater.py bulunamadı!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: GELİŞMİŞ AYARLAR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("🎛️ Gelişmiş Ayarlar")
    
    # Raw Config Editor
    st.subheader("📝 Config.json Editörü")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    
    config_text = st.text_area(
        "config.json",
        value=json.dumps(cfg, indent=2, ensure_ascii=False),
        height=400
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Config Kaydet", type="primary"):
            try:
                new_cfg = json.loads(config_text)
                if safe_save_json(CONFIG_FILE, new_cfg):
                    st.success("✅ Config kaydedildi!")
                    st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON Hatası: {e}")
    
    with col2:
        if st.button("🔄 Varsayılana Sıfırla"):
            if safe_save_json(CONFIG_FILE, DEFAULT_CONFIG):
                st.success("✅ Varsayılan ayarlar yüklendi!")
                st.rerun()
    
    st.markdown("---")
    
    # Notification Settings
    st.subheader("🔔 Bildirim Ayarları")
    
    notif_cfg = cfg.get("notifications", DEFAULT_CONFIG["notifications"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        telegram_enabled = st.checkbox("Telegram Bildirimleri", value=notif_cfg.get("telegram_enabled", False))
        telegram_chat_id = st.text_input("Telegram Chat ID", value=notif_cfg.get("telegram_chat_id", ""), type="password")
    
    with col2:
        discord_enabled = st.checkbox("Discord Bildirimleri", value=notif_cfg.get("discord_enabled", False))
        discord_webhook = st.text_input("Discord Webhook URL", value=notif_cfg.get("discord_webhook", ""), type="password")
    
    if st.button("💾 Bildirim Ayarlarını Kaydet"):
        cfg["notifications"]["telegram_enabled"] = telegram_enabled
        cfg["notifications"]["telegram_chat_id"] = telegram_chat_id
        cfg["notifications"]["discord_enabled"] = discord_enabled
        cfg["notifications"]["discord_webhook"] = discord_webhook
        
        if safe_save_json(CONFIG_FILE, cfg):
            st.success("✅ Bildirim ayarları kaydedildi!")
    
    st.markdown("---")
    
    # System Info
    st.subheader("ℹ️ Sistem Bilgisi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Python Version", sys.version.split()[0])
    with col2:
        st.metric("Platform", sys.platform)
    with col3:
        import streamlit
        st.metric("Streamlit Version", streamlit.__version__)
    
    # File sizes
    st.markdown("#### 📁 Dosya Boyutları")
    
    files_info = [
        ("config.json", CONFIG_FILE),
        ("trade_log.json", TRADE_LOG_FILE),
        ("ai_predictions.json", AI_PRED_FILE),
        ("ohlc_history.json", OHLC_HISTORY_FILE),
    ]
    
    for name, path in files_info:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            st.text(f"{name}: {size_mb:.2f} MB")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9: AI MALİYET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.header("💰 AI Maliyet Analizi")
    
    cfg = safe_load_json(CONFIG_FILE, DEFAULT_CONFIG)
    token_costs = cfg.get("token_cost_per_1k", DEFAULT_CONFIG["token_cost_per_1k"])
    
    # AI Predictions analysis
    ai_preds = safe_load_json(AI_PRED_FILE, [])
    
    if ai_preds:
        df_ai = pd.DataFrame(ai_preds)
        
        # Model distribution
        if "model" in df_ai.columns:
            st.subheader("📊 Model Kullanım Dağılımı")
            
            model_counts = df_ai["model"].value_counts()
            fig = px.pie(values=model_counts.values, names=model_counts.index, title="Model Dağılımı")
            st.plotly_chart(fig, use_container_width=True)
        
        # Estimated costs (rough)
        st.subheader("💵 Tahmini Maliyet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Assume ~500 tokens per prediction
            tokens_per_pred = 500
            total_preds = len(ai_preds)
            
            for model, cost_per_1k in token_costs.items():
                model_preds = len(df_ai[df_ai.get("model", "") == model]) if "model" in df_ai.columns else 0
                if model_preds > 0:
                    estimated_cost = (model_preds * tokens_per_pred / 1000) * cost_per_1k
                    st.metric(f"{model} Tahmini Maliyet", f"${estimated_cost:.4f}")
        
        with col2:
            st.markdown("#### 💰 Token Maliyetleri (1K token)")
            for model, cost in token_costs.items():
                st.text(f"{model}: ${cost}")
        
        # Performance by model
        if "model" in df_ai.columns and "confidence" in df_ai.columns:
            st.subheader("📈 Model Performansı")
            
            model_perf = df_ai.groupby("model").agg({
                "confidence": ["mean", "std", "count"]
            }).round(3)
            
            st.dataframe(model_perf, use_container_width=True)
    else:
        st.info("AI tahmin verisi bulunamadı")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 AutoTraderBot Dashboard v7.0 | 
        📅 {date} |
        ⚡ Powered by Streamlit
    </div>
    """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)

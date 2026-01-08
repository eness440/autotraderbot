# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
# Determine the base directory of the package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use find_dotenv to locate a .env file in the current directory or any
# parent directory.  If none is found, fall back to a .env in BASE_DIR.
_found_env = find_dotenv()
ENV_PATH = _found_env if _found_env else os.path.join(BASE_DIR, ".env")
# Call load_dotenv with the resolved path.  If the file does not exist,
# load_dotenv silently ignores it, leaving the environment unchanged.
load_dotenv(ENV_PATH)

def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y")

def env_float(key: str, default: float = 0.0) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

# ---------------------------------------------------------------------------
# Environment validation
#
# The bot historically started even when critical API keys were missing,
# failing later during the first live API call (often with confusing errors).
# We validate required variables early and provide actionable messages.

def missing_env(keys: list[str]) -> list[str]:
    """Return a list of missing/empty environment variables."""
    out: list[str] = []
    for k in keys:
        v = os.getenv(k)
        if v is None or str(v).strip() == "":
            out.append(k)
    return out


def require_env(keys: list[str], *, context: str = "", strict: bool = True) -> None:
    """Validate that required env vars exist.

    If strict is True and any variable is missing, raise RuntimeError with a
    readable message. If strict is False, only log via stdout (settings.py has
    no logger dependency).
    """
    missing = missing_env(keys)
    if not missing:
        return

    ctx = f" ({context})" if context else ""
    msg = (
        "Missing required environment variables" + ctx + ": "
        + ", ".join(missing)
        + "\n\nFix:\n"
        + "1) Create a .env file next to settings.py or project root\n"
        + "2) Add the variables above\n"
        + "3) Re-run the bot\n"
    )
    if strict:
        raise RuntimeError(msg)
    print("[ENV][WARN] " + msg)


def write_env_example(path: str) -> None:
    """Write a minimal .env.example file."""
    template = """# Copy to .env and fill values\n\n# OKX\nOKX_API_KEY=\nOKX_API_SECRET=\nOKX_API_PASSPHRASE=\nOKX_USE_TESTNET=true\n\n# OpenAI (optional; required only if ChatGPT is enabled)\nOPENAI_API_KEY=\nOPENAI_FILTER_MODEL=gpt-4o-mini\nOPENAI_DECISION_MODEL=gpt-4o-mini\n\n# DeepSeek (optional; required only if DeepSeek is enabled)\nDEEPSEEK_API_KEY=\nDEEPSEEK_DECISION_MODEL=deepseek-reasoner\n\n# Optional providers\nMACRO_API_KEY=\nCOINGLASS_API_KEY=\nETHERSCAN_API_KEY=\n"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)

# OKX
OKX_API_KEY        = env_str("OKX_API_KEY")
OKX_API_SECRET     = env_str("OKX_API_SECRET")
OKX_API_PASSPHRASE = env_str("OKX_API_PASSPHRASE")
OKX_USE_TESTNET    = env_bool("OKX_USE_TESTNET", True)

# OpenAI (şu an boş olabilir)
OPENAI_API_KEY        = env_str("OPENAI_API_KEY")
OPENAI_FILTER_MODEL   = env_str("OPENAI_FILTER_MODEL", "gpt-4o-mini")
OPENAI_DECISION_MODEL = "gpt-4o-mini"

# DeepSeek configuration
# Use environment variables if set; fallback to defaults for development.
DEEPSEEK_API_KEY        = env_str("DEEPSEEK_API_KEY", "")
DEEPSEEK_DECISION_MODEL = env_str("DEEPSEEK_DECISION_MODEL", "deepseek-reasoner")

# Genel
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Durum dosyaları
#
# Botun durdurulması veya duraklatılması ile ilgili bayrak dosyaları
# hem arayüz tarafında (dashboard_streamlit.py) hem de bot tarafında
# aynı konumda tutulmalıdır. Önceki sürümlerde bot `state/PAUSE.flag` ve
# `state/KILL.flag` dosyalarını izlerken, arayüz `runtime/PAUSE` ve
# `runtime/KILL` dosyalarını kullanıyordu. Bu tutarsızlık nedeniyle
# kill/pause düğmeleri beklenen etkiyi göstermiyordu. Aşağıdaki ayarlar
# iki tarafın da aynı dosyaları kullanmasını sağlar.

# runtime dizini, proje kökünde bulunan ve kullanıcı arayüzünün de
# kullandığı klasördür. Dosya isimlerinde uzantı kullanılmaz.
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
os.makedirs(RUNTIME_DIR, exist_ok=True)

# Duraklatma bayrağı: var ise bot yeni işlem açmaz, mevcut açık pozisyonları
# kontrol etmeye devam eder. Arayüzde PAUSE düğmesine basıldığında
# bu dosya oluşturulur ve kullanıcı kaldırana kadar bot bekleme modunda kalır.
PAUSE_FLAG = os.path.join(RUNTIME_DIR, "PAUSE")

# Kill bayrağı: var ise bot güvenli bir şekilde kapanır. Arayüzde KILL
# düğmesine basıldığında bu dosya oluşturulur ve bot bir sonraki döngüde
# kendisini durdurur.
KILL_FLAG = os.path.join(RUNTIME_DIR, "KILL")

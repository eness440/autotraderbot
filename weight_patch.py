# ml/weight_patch.py
import json, statistics
from pathlib import Path

CONFIG_FILE = Path("config.json")
AI_PRED_FILE = Path("metrics/ai_predictions.json")


def _safe_float(x, default=0.0):
    """None veya bozuk değerleri güvenle float'a çevirir."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def auto_rebalance():
    """
    ChatGPT, DeepSeek, BiLSTM, RL ağırlıklarını performansa göre günceller.

    Önceki basit metrik: conf * (1 + pnl)
    Yeni yaklaşım: model bazlı win-rate + ortalama confidence
      - win: outcome='win' veya direction_ok=True veya pnl>0
      - perf_score = win_rate * avg_conf
      - Sonra bu skorlar normalize edilip config.json içindeki hybrid_weights'e yazılır.
    """
    try:
        if not AI_PRED_FILE.exists():
            return
        data = json.loads(AI_PRED_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return

        # Son 500 kayıt üzerinden model bazlı istatistik
        stats = {}
        for r in data[-500:]:
            m = str(r.get("model", "")).lower()
            if not m:
                continue

            conf = _safe_float(r.get("confidence"), 0.0)
            pnl = _safe_float(r.get("pnl"), 0.0)

            # Win tanımı: direction_ok / outcome / pnl > 0 sinyallerinden herhangi biri
            outcome = str(r.get("outcome", "")).lower()
            direction_ok = bool(r.get("direction_ok")) if r.get("direction_ok") is not None else None

            is_win = False
            if direction_ok is True:
                is_win = True
            elif outcome == "win":
                is_win = True
            elif pnl > 0:
                is_win = True

            s = stats.setdefault(m, {"n": 0, "wins": 0, "sum_conf": 0.0, "sum_pnl": 0.0})
            s["n"] += 1
            s["sum_conf"] += conf
            s["sum_pnl"] += pnl
            if is_win:
                s["wins"] += 1

        if not stats:
            return

        # Model bazlı performans skoru: win_rate * avg_conf
        perf_scores = {}
        for m, s in stats.items():
            n = s["n"]
            if n <= 0:
                continue
            win_rate = s["wins"] / float(n)
            avg_conf = s["sum_conf"] / float(n) if n > 0 else 0.0

            # PnL'i hafifçe dikkate almak istersen buraya ek ağırlık eklenebilir.
            perf = win_rate * avg_conf
            if perf > 0:
                perf_scores[m] = perf

        if not perf_scores:
            return

        total = sum(perf_scores.values())
        if total <= 0:
            return

        # Normalize et
        weights = {m: round(v / total, 3) for m, v in perf_scores.items()}

        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
        cfg["hybrid_weights"] = {**cfg.get("hybrid_weights", {}), **weights}
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print("[WeightPatch] Güncellendi (win-rate bazlı):", weights)
    except Exception as e:
        print("[WeightPatch] Hata:", e)

# -*- coding: utf-8 -*-
"""
auto_updater.py
---------------

This module orchestrates periodic refreshes of data, models and risk
parameters for the trading bot.  Despite its name, **it does not update
its own code nor pull any external updates**; it simply coordinates the
execution of internal scripts.  The update routine comprises:

1. **Data update**: invoke ``ml/build_dataset.py`` to generate a fresh
   supervised dataset for the BiLSTM model.
2. **BiLSTM training**: re-train the BiLSTM classifier on the latest data.
3. **RL training**: re-train the PPO agent via ``ml/rl_train.py``.
4. **Weight patch**: apply updated logistic weights to the decision layer
   via ``ml/weight_patch.py``.
5. **Risk calibration**: update calibration curves and leverage
   schedules through ``calibrate_confidence.py``.

Each task writes progress to ``logs/update.log`` and the overall
scheduling is controlled by the ``UPDATE_PLAN`` dictionary.  You can run
this module manually with::

    python auto_updater.py

to perform a one-time update.  Alternatively schedule it via cron or
another external scheduler.
"""

import os, sys, json, time, traceback, glob, threading, subprocess
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Any  # Added for provider tasks
import numpy as np

HYBRID_MODE = True
ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
METRICS_DIR = os.path.join(ROOT, "metrics")
MODELS_DIR = os.path.join(ROOT, "models")
STATE_PATH = os.path.join(ROOT, "update_state.json")
LOG_PATH = os.path.join(LOG_DIR, "update.log")

for d in [LOG_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# When spawning child processes for model training we want to avoid
# indefinite hangs.  You can adjust the timeout via the
# ``TRAIN_TIMEOUT`` environment variable (defaults to 600 seconds).
# Each subprocess call below will respect this timeout to give
# long-running training routines enough time to finish, but will
# eventually terminate so the auto updater does not stall forever.
TRAIN_TIMEOUT = int(os.getenv('TRAIN_TIMEOUT', '600'))

# Ensure child python processes use UTF-8 stdout/stderr (fixes Windows cp1254 emoji crash)
SUBPROC_ENV = os.environ.copy()
SUBPROC_ENV.setdefault("PYTHONIOENCODING", "utf-8")
SUBPROC_ENV.setdefault("PYTHONUTF8", "1")

# Ensure subprocess can import project packages (fixes "No module named 'ml'" and relative import issues)
_existing_pp = SUBPROC_ENV.get("PYTHONPATH", "")
_sep = ";" if os.name == "nt" else ":"
SUBPROC_ENV["PYTHONPATH"] = (ROOT if not _existing_pp else (_existing_pp + _sep + ROOT))

# Separate (longer) timeouts for training tasks (override via env if needed)
BILSTM_TIMEOUT = int(os.getenv("BILSTM_TIMEOUT", str(max(TRAIN_TIMEOUT, 10800))))
RL_TIMEOUT = int(os.getenv("RL_TIMEOUT", str(max(TRAIN_TIMEOUT, 7200))))

def _utc():
    # timezone-aware UTC timestamp (avoids deprecated utcnow())
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _log(msg):
    """
    Write a log line to both stdout and the update log file.  If the
    current log exceeds a size threshold, rotate it by renaming with
    a timestamp suffix and prune older rotated logs.  This prevents
    unbounded growth of the ``update.log`` file and provides simple
    retention.  Rotations occur at most once per call and use a
    best‑effort approach; failures are silently ignored.

    Args:
        msg: A message string to log.
    """
    line = f"[{_utc()}] {msg}"
    print(line)
    try:
        # Rotate if file is too large (>10MB)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 10_000_000:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rotated = os.path.join(LOG_DIR, f"update_{ts}.log")
            try:
                os.rename(LOG_PATH, rotated)
            except Exception:
                pass
            # Prune old rotated logs beyond 5 most recent
            try:
                logs = sorted(
                    [f for f in os.listdir(LOG_DIR) if f.startswith("update_") and f.endswith(".log")],
                    reverse=True,
                )
                for old in logs[5:]:
                    try:
                        os.remove(os.path.join(LOG_DIR, old))
                    except Exception:
                        pass
            except Exception:
                pass
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: pass
    return default

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    except: pass

def _parse_iso_utc(iso_str: str) -> datetime:
    """
    Parse ISO timestamp from state file into a timezone-aware UTC datetime.
    Accepts both "...Z" and offsets like "+00:00". If tzinfo is missing, assumes UTC.
    """
    s = (iso_str or "").strip()
    # handle trailing Z
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _due(last_iso, days):
    if not last_iso: return True
    try:
        last = _parse_iso_utc(last_iso)
        now = datetime.now(timezone.utc)
        return (now - last) >= timedelta(days=days)
    except: return True

def _run_subprocess_stream(cmd, *, log_file_path: str, timeout_s: int, cwd: str, env: dict,
                           tag: str, heartbeat_s: int = 30) -> tuple[int, list[str], bool]:
    """
    Run subprocess while streaming combined stdout/stderr into:
      - a dedicated subprocess log file (log_file_path)
      - the main logs/update.log (tee)
    Also emits periodic heartbeat logs so the main console doesn't look stuck.

    Returns: (returncode, tail_lines, timed_out)
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    tail = deque(maxlen=80)

    start_t = time.monotonic()
    last_hb = start_t

    try:
        with open(log_file_path, "a", encoding="utf-8") as lf, open(LOG_PATH, "a", encoding="utf-8") as uf:
            lf.write(f"\n===== [{_utc()}] START {tag} =====\n")
            lf.write("CMD: " + " ".join(map(str, cmd)) + "\n\n")
            lf.flush()

            uf.write(f"\n===== [{_utc()}] [{tag}] START =====\n")
            uf.write("CMD: " + " ".join(map(str, cmd)) + "\n\n")
            uf.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            def reader():
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        # write raw line to subprocess log
                        lf.write(line)
                        lf.flush()

                        # tee the same line into update.log (prefixed)
                        # (line already includes \n)
                        uf.write(f"[{_utc()}] [{tag}] {line}")
                        uf.flush()

                        s = line.rstrip("\n")
                        if s:
                            tail.append(s)
                except Exception:
                    pass

            rt = threading.Thread(target=reader, daemon=True)
            rt.start()

            timed_out = False
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    elapsed = now - start_t

                    if elapsed >= timeout_s:
                        timed_out = True
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        time.sleep(1)
                        try:
                            if proc.poll() is None:
                                proc.kill()
                        except Exception:
                            pass
                        break

                    if (now - last_hb) >= heartbeat_s:
                        last_hb = now
                        last_line = tail[-1] if len(tail) else ""
                        if last_line:
                            _log(f"[{tag}] hâlâ çalışıyor... (geçen: {int(elapsed)}s) son çıktı: {last_line[:180]}")
                        else:
                            _log(f"[{tag}] hâlâ çalışıyor... (geçen: {int(elapsed)}s)")

            try:
                rt.join(timeout=2)
            except Exception:
                pass

            rc = proc.returncode if proc.returncode is not None else -1

            lf.write(f"\n===== [{_utc()}] END {tag} rc={rc} timeout={timed_out} =====\n")
            lf.flush()

            uf.write(f"\n===== [{_utc()}] [{tag}] END rc={rc} timeout={timed_out} =====\n")
            uf.flush()

            return rc, list(tail), timed_out

    except Exception as e:
        _log(f"[{tag}] subprocess stream error: {e}")
        return -1, list(tail), False

# --- Update plan frequencies (days) for each task.  A value of 0.5 means
# roughly twice per day, 1.0 means daily, 7 means weekly, etc.  Additional
# tasks such as macro and external data updates are defined below.
UPDATE_PLAN = {
    "data_update": 0.5,
    "weight_patch": 0.5,
    # Note: training tasks are removed from the auto‑updater.  BiLSTM and RL
    # models are trained externally and the updater should not spawn heavy
    # training jobs during an update cycle.  This avoids blocking the
    # asynchronous trading loop.  If you wish to re‑enable training here,
    # simply add the corresponding tasks back to UPDATE_PLAN.
    "risk": 7,
    # Update social sentiment hourly (approx. 8 times per day)
    "update_sentiment": 0.34,
    # Update on-chain metrics daily
    "update_onchain": 1.0,
    # Monitor model performance daily and adjust weights
    "model_monitor": 1.0,
    # Optimise parameters weekly
    "param_optimize": 7.0,
    # NEW: refresh macro event calendar daily
    "update_macro": 1.0,
    # NEW: refresh external metrics (options/whale/liq) daily
    "update_external": 1.0,
    # NEW: retrain the transformer model weekly
    "train_transformer": 7.0
    ,
    # Update trending social coins daily.  This fetches CoinMarketCap and
    # LunarCrush trending lists and writes them into data/social_trends.json.
    "update_social_trends": 1.0,
    # Re-enable training tasks: retrain BiLSTM weekly
    "train_bilstm": 7.0,
    # Retrain RL agent weekly.  Adjust the frequency as needed.
    "train_rl": 7.0
    ,
    # Run performance analysis daily.  This computes PnL stats and updates cooldowns.
    "performance_analysis": 1.0
    ,
    # Enable daily liquidation heatmap updates via liquidation_data_provider
    "update_liquidation": 1.0
    ,
    # Enable daily options market metrics updates
    "update_ops": 1.0
    ,
    # Enable daily whale alert updates
    "update_whale": 1.0
    ,
    # Hyperparameter search for logistic regression calibration.  This task
    # calls hyperparameter_search.py to tune model parameters.  Run
    # monthly (every 30 days) by default.
    "hyperparameter_search": 30.0
    ,
    # Risk tier optimisation based on recent volatility.  This task
    # updates risk_config.json via risk_optimizer.py.  Run weekly.
    "risk_optimize": 7.0
}

# ──────────────────────────────── TASKS ────────────────────────────────

def task_data_update():
    try:
        build_path = os.path.join(ROOT, "build_dataset.py") # ml klasöründe değilse root'ta ara
        if not os.path.exists(build_path):
            build_path = os.path.join(ROOT, "ml", "build_dataset.py")

        if not os.path.exists(build_path):
            return {"ok": False, "error": "build_dataset.py missing"}

        _log("[DATA] Veri seti yenileniyor...")
        cmd = [sys.executable, build_path, "--window", "180", "--horizon", "12"]
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=TRAIN_TIMEOUT,
            cwd=ROOT,
            env=SUBPROC_ENV,
        )

        # Only apply weight patch here.  BiLSTM and RL training are now
        # managed externally to avoid blocking the updater loop.  If you
        # wish to train models automatically, invoke the training scripts
        # manually or re‑enable the tasks in UPDATE_PLAN.
        task_weight_patch()
        return {"ok": True}
    except Exception as e:
        _log(f"[ERR] Data update fail: {e}")
        return {"ok": False, "error": str(e)}

def task_weight_patch():
    try:
        import weight_patch
        weight_patch.auto_rebalance()
        return {"ok": True}
    except: return {"ok": False}

def task_train_bilstm():
    """Train the BiLSTM model using the standalone PyTorch script."""
    try:
        data_file = os.path.join(ROOT, "data", "supervised_w180_h12_g0.parquet")
        if not os.path.exists(data_file):
            _log("[TRAIN-BiLSTM] Veri dosyası yok, eğitim atlandı.")
            return {"ok": False}

        script_path = os.path.join(ROOT, "ml", "bilstm_train.py")
        if not os.path.isfile(script_path):
            _log("[TRAIN-BiLSTM] bilstm_train.py bulunamadı, eğitim atlandı.")
            return {"ok": False}

        cmd = [sys.executable, script_path, "--data", os.path.basename(data_file), "--epochs", "20"]
        _log("[TRAIN-BiLSTM] PyTorch BiLSTM modeli eğitiliyor...")

        bilstm_sublog = os.path.join(LOG_DIR, "bilstm_train_subprocess.log")
        # Backup current best model and metrics for rollback
        prev_metrics_path = os.path.join(METRICS_DIR, "bilstm_metrics.json")
        prev_metrics_data = None
        prev_acc = None
        # Model path (trained weights saved by bilstm_train)
        model_dir = os.path.join(ROOT, "models")
        prev_model_path = os.path.join(model_dir, "bilstm_best.pt")
        backup_model_path = None
        backup_metrics_path = None
        try:
            if os.path.exists(prev_metrics_path):
                with open(prev_metrics_path, "r", encoding="utf-8") as f:
                    prev_metrics_data = json.load(f)
                # Try both keys for backward compatibility
                prev_acc = prev_metrics_data.get("last_accuracy") or prev_metrics_data.get("best_val_acc")
                if prev_acc is not None:
                    prev_acc = float(prev_acc)
            # Copy current best model to a timestamped backup
            if os.path.exists(prev_model_path):
                ts = int(time.time())
                backup_model_path = os.path.join(model_dir, f"bilstm_best_{ts}.pt.bak")
                try:
                    import shutil
                    shutil.copy2(prev_model_path, backup_model_path)
                except Exception:
                    backup_model_path = None
            # Also back up metrics file
            if os.path.exists(prev_metrics_path):
                ts = int(time.time())
                backup_metrics_path = os.path.join(METRICS_DIR, f"bilstm_metrics_{ts}.json.bak")
                try:
                    import shutil
                    shutil.copy2(prev_metrics_path, backup_metrics_path)
                except Exception:
                    backup_metrics_path = None
        except Exception:
            pass

        rc, tail, timed_out = _run_subprocess_stream(
            cmd,
            log_file_path=bilstm_sublog,
            timeout_s=BILSTM_TIMEOUT,
            cwd=ROOT,
            env=SUBPROC_ENV,
            tag="BiLSTM",
            heartbeat_s=30,
        )

        if timed_out:
            _log(f"[TRAIN-BiLSTM] Eğitim scripti zaman aşımına uğradı. Detaylar: {bilstm_sublog}")
            return {"ok": False, "error": "bilstm_train timeout"}

        if rc != 0:
            _log(f"[TRAIN-BiLSTM] Eğitim scripti başarısız. Detaylar: {bilstm_sublog}")
            if tail:
                _log("[TRAIN-BiLSTM] Son satırlar: " + " | ".join(tail[-5:])[:900])
            return {"ok": False, "error": "bilstm_train failed"}

        # After training, evaluate the new metrics and decide whether to deploy
        new_metrics_path = os.path.join(METRICS_DIR, "bilstm_metrics.json")
        new_acc = None
        try:
            if os.path.exists(new_metrics_path):
                with open(new_metrics_path, "r", encoding="utf-8") as f:
                    new_data = json.load(f)
                new_acc = new_data.get("last_accuracy") or new_data.get("best_val_acc")
                if new_acc is not None:
                    new_acc = float(new_acc)
        except Exception:
            new_acc = None
        # Determine whether the new model is better; fallback to previous if worse
        rollback = False
        if prev_acc is not None and new_acc is not None:
            if new_acc < prev_acc:
                rollback = True
        elif prev_acc is not None and new_acc is None:
            rollback = True
        # If rollback is needed, restore previous model and metrics
        if rollback:
            _log("[TRAIN-BiLSTM] Yeni model mevcut modelden daha kötü. Geri alınıyor...")
            # Restore model file
            if backup_model_path and os.path.exists(backup_model_path):
                try:
                    import shutil
                    shutil.copy2(backup_model_path, prev_model_path)
                except Exception as e:
                    _log(f"[TRAIN-BiLSTM] Model geri yükleme başarısız: {e}")
            # Restore metrics file
            if backup_metrics_path and os.path.exists(backup_metrics_path):
                try:
                    import shutil
                    shutil.copy2(backup_metrics_path, prev_metrics_path)
                except Exception as e:
                    _log(f"[TRAIN-BiLSTM] Metrics geri yükleme başarısız: {e}")
            return {"ok": False, "error": "bilstm_eval_did_not_improve"}
        # Otherwise, training succeeded and new model is kept
        metrics_payload = {"last_update": _utc()}
        _save_json(os.path.join(METRICS_DIR, "bilstm_metrics.json"), metrics_payload)
        _log("[TRAIN-BiLSTM] PyTorch modeli eğitimi tamamlandı ve güncellendi.")
        return {"ok": True}

    except Exception as e:
        _log(f"[ERR-BiLSTM] {e}")
        return {"ok": False, "error": str(e)}

def task_train_rl() -> dict:
    """Train or refresh the PPO reinforcement learning agent."""
    try:
        algo = os.getenv("RL_ALGO", "ppo").lower()
        advanced_script = os.path.join(ROOT, "ml", "rl_train_advanced.py")
        if os.path.isfile(advanced_script):
            # Run as module so "ml" package imports work
            cmd = [sys.executable, "-m", "ml.rl_train_advanced", "--algo", algo, "--steps", "10000", "--model_dir", os.path.join(ROOT, "models")]
            _log(f"[TRAIN-RL] {algo.upper()} ajanı eğitiliyor (advanced)...")
        else:
            train_script = os.path.join(ROOT, "ml", "rl_train.py")
            if not os.path.isfile(train_script):
                _log("[TRAIN-RL] rl_train.py bulunamadı, eğitim atlandı.")
                return {"ok": False, "error": "rl_train.py missing"}
            # Run as module so relative imports work
            cmd = [sys.executable, "-m", "ml.rl_train", "--timesteps", "10000", "--device", "cpu"]
            _log("[TRAIN-RL] PPO ajanı eğitiliyor (legacy)...")

        rl_sublog = os.path.join(LOG_DIR, "rl_train_subprocess.log")
        rc, tail, timed_out = _run_subprocess_stream(
            cmd,
            log_file_path=rl_sublog,
            timeout_s=RL_TIMEOUT,
            cwd=ROOT,
            env=SUBPROC_ENV,
            tag="RL",
            heartbeat_s=30,
        )

        if timed_out:
            _log(f"[TRAIN-RL] RL eğitim scripti zaman aşımına uğradı. Detaylar: {rl_sublog}")
            return {"ok": False, "error": "rl_train timeout"}

        if rc != 0:
            _log(f"[TRAIN-RL] RL eğitim scripti başarısız. Detaylar: {rl_sublog}")
            if tail:
                _log("[TRAIN-RL] Son satırlar: " + " | ".join(tail[-5:])[:900])

            try:
                train_script = os.path.join(ROOT, "ml", "rl_train.py")
                if os.path.isfile(train_script):
                    # Fallback also as module
                    fallback_cmd = [sys.executable, "-m", "ml.rl_train", "--timesteps", "10000", "--device", "cpu"]
                    _log("[TRAIN-RL] Falling back to legacy PPO training...")

                    rc2, tail2, timed_out2 = _run_subprocess_stream(
                        fallback_cmd,
                        log_file_path=rl_sublog,
                        timeout_s=RL_TIMEOUT,
                        cwd=ROOT,
                        env=SUBPROC_ENV,
                        tag="RL-LEGACY",
                        heartbeat_s=30,
                    )

                    if timed_out2:
                        _log(f"[TRAIN-RL] Legacy RL training timeout. Detaylar: {rl_sublog}")
                        return {"ok": False, "error": "rl_train timeout"}

                    if rc2 != 0:
                        _log(f"[TRAIN-RL] Legacy RL training also failed. Detaylar: {rl_sublog}")
                        if tail2:
                            _log("[TRAIN-RL] Son satırlar: " + " | ".join(tail2[-5:])[:900])
                        return {"ok": False, "error": "rl_train failed"}

                    rl_metrics = {"last_update": _utc(), "algorithm": "ppo"}
                    _save_json(os.path.join(METRICS_DIR, "rl_metrics.json"), rl_metrics)
                    _log("[TRAIN-RL] Legacy PPO training completed.")
                    return {"ok": True}
                else:
                    return {"ok": False, "error": "rl_train.py missing"}
            except Exception as fallback_exc:
                _log(f"[ERR-RL] Fallback RL training failed: {fallback_exc}")
                return {"ok": False, "error": "rl_train failed"}

        rl_metrics = {"last_update": _utc(), "algorithm": algo}
        _save_json(os.path.join(METRICS_DIR, "rl_metrics.json"), rl_metrics)
        _log(f"[TRAIN-RL] RL eğitim ({algo.upper()}) tamamlandı.")
        return {"ok": True}

    except Exception as e:
        _log(f"[ERR-RL] {e}")
        return {"ok": False, "error": str(e)}

def task_risk():
    try:
        for script in ["risk_dataset_builder.py", "risk_calibrator.py", "risk_schedule_builder.py"]:
            if os.path.exists(os.path.join(ROOT, script)):
                subprocess.run([sys.executable, os.path.join(ROOT, script)], check=False)

        try:
            import importlib, controller_async
            importlib.reload(controller_async)
        except: pass

        return {"ok": True}
    except Exception as e:
        _log(f"[ERR-RISK] {e}")
        return {"ok": False}

def task_update_sentiment() -> dict:
    """Update social sentiment metrics.

    This task first refreshes the trending coin list using
    ``update_social_trends`` from ``social_scanner``.  It then computes
    per‑symbol sentiment scores via
    ``update_social_sentiment_for_symbols`` using NewsAPI, LunarCrush and
    the Fear & Greed Index.  If no trending coins are available, it
    falls back to a global sentiment update via ``update_social_sentiment``.

    Returns a dict indicating success and the updated sentiment data.
    """
    try:
        from social_scanner import update_social_trends  # type: ignore
    except Exception as e:
        _log(f"[ERR-SENTIMENT] Could not import social_scanner: {e}")
        return {"ok": False, "error": str(e)}
    try:
        from social_sentiment_updater import (
            update_social_sentiment,
            update_social_sentiment_for_symbols,
        )  # type: ignore
    except Exception as e:
        _log(f"[ERR-SENTIMENT] Could not import social_sentiment_updater: {e}")
        return {"ok": False, "error": str(e)}
    try:
        # Refresh trending coins and compute sentiment for them
        trending_syms = update_social_trends(limit=10)
        if trending_syms:
            res = update_social_sentiment_for_symbols(trending_syms)
        else:
            # Fallback to global sentiment update
            res = update_social_sentiment()
        _save_json(os.path.join(METRICS_DIR, "sentiment_update.json"), {"last_update": _utc()})
        _log("[SENTIMENT] Sosyal sentiment güncellendi.")
        return {"ok": True, "data": res}
    except Exception as e:
        _log(f"[ERR-SENTIMENT] {e}")
        return {"ok": False, "error": str(e)}

def task_update_onchain() -> dict:
    try:
        from onchain_data_updater import update_onchain_metrics  # type: ignore
        res = update_onchain_metrics()
        _save_json(os.path.join(METRICS_DIR, "onchain_update.json"), {"last_update": _utc()})
        _log("[ONCHAIN] On-chain metrikler güncellendi.")
        return {"ok": True, "data": res}
    except Exception as e:
        _log(f"[ERR-ONCHAIN] {e}")
        return {"ok": False, "error": str(e)}

def task_model_monitor() -> dict:
    try:
        from model_performance_monitor import (
            evaluate_models,
            adjust_weights,
            deactivate_underperforming_models,
        )  # type: ignore
        report = evaluate_models()
        adjust_weights(report)
        deactivate_underperforming_models(report)
        _save_json(os.path.join(METRICS_DIR, "model_monitor.json"), {"last_update": _utc(), "report": report})
        _log("[MODEL-MONITOR] Model performansı analiz edildi ve ağırlıklar güncellendi.")
        return {"ok": True, "report": report}
    except Exception as e:
        _log(f"[ERR-MODEL-MON] {e}")
        return {"ok": False, "error": str(e)}

def task_param_optimize() -> dict:
    try:
        from parameter_optimizer import run_optimization  # type: ignore
        res = run_optimization(n_trials=20)
        if res:
            _save_json(
                os.path.join(METRICS_DIR, "param_opt.json"),
                {"last_update": _utc(), "result": res},
            )
            try:
                best_params = res.get("best_params", {})
                cfg_path = os.path.join(ROOT, "config.json")
                cfg = {}
                if os.path.isfile(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                updated = False
                cd = best_params.get("cooldown_min")
                if cd is not None:
                    try:
                        cfg["trade_cooldown_min"] = int(cd)
                        updated = True
                    except Exception:
                        pass
                slm = best_params.get("stop_loss_mult")
                if slm is not None:
                    try:
                        cfg["stop_loss_mult"] = float(slm)
                        updated = True
                    except Exception:
                        pass
                c_val = best_params.get("C")
                if c_val is not None:
                    try:
                        cfg["logistic_C"] = float(c_val)
                        updated = True
                    except Exception:
                        pass
                ma_short = best_params.get("ma_short")
                ma_long = best_params.get("ma_long")
                if ma_short is not None:
                    cfg["ma_short"] = int(ma_short)
                    updated = True
                if ma_long is not None:
                    cfg["ma_long"] = int(ma_long)
                    updated = True
                if updated:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
                    _log(f"[PARAM-OPT] Config updated with optimisation results: {best_params}")
            except Exception as e:
                _log(f"[PARAM-OPT] Config update failed: {e}")
            _log("[PARAM-OPT] Parametre optimizasyonu tamamlandı.")
            return {"ok": True, "result": res}
        else:
            _log("[PARAM-OPT] Parametre optimizasyonu çalıştırılamadı veya veri yok.")
            return {"ok": False}
    except Exception as e:
        _log(f"[ERR-PARAM-OPT] {e}")
        return {"ok": False, "error": str(e)}

def task_update_macro() -> dict:
    try:
        from macro_data_updater import update_macro_events  # type: ignore
    except Exception as e:
        _log(f"[ERR-MACRO-IMP] Could not import macro_data_updater: {e}")
        return {"ok": False, "error": str(e)}
    try:
        events = update_macro_events()
        _save_json(os.path.join(METRICS_DIR, "macro_update.json"), {"last_update": _utc(), "num_events": len(events)})
        _log(f"[MACRO] Güncel makro etkinlikler kaydedildi (n={len(events)})")
        return {"ok": True, "num_events": len(events)}
    except Exception as e:
        _log(f"[ERR-MACRO] {e}")
        return {"ok": False, "error": str(e)}

def task_update_external() -> dict:
    try:
        from external_data_updater import update_external_metrics  # type: ignore
    except Exception as e:
        _log(f"[ERR-EXTERNAL-IMP] Could not import external_data_updater: {e}")
        return {"ok": False, "error": str(e)}
    try:
        res = update_external_metrics()
        n = len([k for k in res.keys() if k != "updated_at"])
        _save_json(os.path.join(METRICS_DIR, "external_update.json"), {"last_update": _utc(), "num_symbols": n})
        _log(f"[EXTERNAL] Dış metrikler güncellendi (n={n})")
        return {"ok": True, "num_symbols": n}
    except Exception as e:
        _log(f"[ERR-EXTERNAL] {e}")
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Additional provider update tasks
#
# These helper functions fetch data from the individual optional providers:
# options data (implied vol, skew, open interest), liquidation heatmaps and
# whale alerts.  They are not scheduled by default but can be added to
# UPDATE_PLAN if desired.  Each task persists its results into a separate
# JSON file within the metrics directory.

def _resolve_symbols() -> list[str]:
    """Resolve trading symbols from data/symbols_okx.json or return defaults."""
    try:
        syms = []
        syms_path = os.path.join(ROOT, "data", "symbols_okx.json")
        if os.path.isfile(syms_path):
            with open(syms_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    syms = [str(s).upper() for s in loaded if isinstance(s, str)]
        if not syms:
            syms = ["BTC/USDT", "ETH/USDT"]
        return syms
    except Exception:
        return ["BTC/USDT", "ETH/USDT"]


def task_update_ops() -> dict:
    """Fetch options market metrics for each symbol and save results."""
    try:
        from ops_data_provider import fetch_options_metrics  # type: ignore
    except Exception as e:
        _log(f"[ERR-OPS-IMP] Could not import ops_data_provider: {e}")
        return {"ok": False, "error": str(e)}
    symbols = _resolve_symbols()
    # Deribit options are effectively limited to BTC/ETH in this project.
    # Avoid calling the provider for unsupported symbols to reduce network
    # churn and connection resets.
    supported_bases = {
        s.strip().upper() for s in os.getenv("OPTIONS_SUPPORTED_BASES", "BTC,ETH").split(",") if s.strip()
    }
    if not supported_bases:
        supported_bases = {"BTC", "ETH"}
    data: dict[str, Any] = {}
    for sym in symbols:
        base = sym.split("/")[0] if "/" in sym else sym
        base_u = str(base).upper()
        if base_u not in supported_bases:
            # Explicit unsupported marker (schema guarantee)
            data[sym] = {
                "symbol": base_u,
                "supported": False,
                "available": False,
                "provider": "deribit",
                "reason": "unsupported_currency",
                "implied_volatility": None,
                "put_call_ratio": None,
                "open_interest": None,
            }
            continue
        try:
            m = fetch_options_metrics(symbol=base_u)
            if isinstance(m, dict) and m:
                data[sym] = m
        except Exception as e:
            _log(f"[ERR-OPS] {sym}: {e}")
    try:
        out = {"updated_at": _utc(), **data}
        out_path = os.path.join(METRICS_DIR, "options_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _log(f"[OPS] Ops metrics updated for {len(data)} symbols")
        return {"ok": True, "num_symbols": len(data)}
    except Exception as e:
        _log(f"[ERR-OPS-WRITE] {e}")
        return {"ok": False, "error": str(e)}


def task_update_liquidation() -> dict:
    """Fetch liquidation heatmaps for each symbol and save results."""
    try:
        from liquidation_data_provider import fetch_liquidation_heatmap  # type: ignore
    except Exception as e:
        _log(f"[ERR-LIQ-IMP] Could not import liquidation_data_provider: {e}")
        return {"ok": False, "error": str(e)}
    symbols = _resolve_symbols()
    data: dict[str, Any] = {}
    for sym in symbols:
        base = sym.split("/")[0] if "/" in sym else sym
        try:
            heatmap = fetch_liquidation_heatmap(symbol=base)
            if isinstance(heatmap, dict) and heatmap:
                data[sym] = heatmap
        except Exception as e:
            _log(f"[ERR-LIQ] {sym}: {e}")
    try:
        out = {"updated_at": _utc(), **data}
        out_path = os.path.join(METRICS_DIR, "liquidation_heatmap.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _log(f"[LIQ] Liquidation heatmaps updated for {len(data)} symbols")
        return {"ok": True, "num_symbols": len(data)}
    except Exception as e:
        _log(f"[ERR-LIQ-WRITE] {e}")
        return {"ok": False, "error": str(e)}


def task_update_whale() -> dict:
    """Fetch whale alerts for each symbol and save results."""
    try:
        from whale_alert_provider import fetch_whale_alerts  # type: ignore
    except Exception as e:
        _log(f"[ERR-WHALE-IMP] Could not import whale_alert_provider: {e}")
        return {"ok": False, "error": str(e)}
    symbols = _resolve_symbols()
    data: dict[str, Any] = {}
    for sym in symbols:
        base = sym.split("/")[0] if "/" in sym else sym
        try:
            alerts = fetch_whale_alerts(token=base)
            if isinstance(alerts, list) and alerts:
                data[sym] = {"whale_alert_count": len(alerts)}
        except Exception as e:
            _log(f"[ERR-WHALE] {sym}: {e}")
    try:
        out = {"updated_at": _utc(), **data}
        out_path = os.path.join(METRICS_DIR, "whale_alerts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _log(f"[WHALE] Whale alerts updated for {len(data)} symbols")
        return {"ok": True, "num_symbols": len(data)}
    except Exception as e:
        _log(f"[ERR-WHALE-WRITE] {e}")
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Performance analysis and cooldown updates
#
# This task computes daily PnL statistics from the trade log and updates
# cooldown parameters accordingly.  It persists a summary into
# metrics/performance_analysis.json.
def task_performance_analysis() -> dict:
    """Analyze trade performance and update cooldowns.

    Returns:
        dict: status and optional data payload
    """
    try:
        from performance_analyzer import analyze_and_update_cooldowns  # type: ignore
    except Exception as e:
        _log(f"[ERR-PERF-IMP] Could not import performance_analyzer: {e}")
        return {"ok": False, "error": str(e)}
    try:
        res = analyze_and_update_cooldowns()
        # Persist the analysis results along with last update timestamp
        _save_json(
            os.path.join(METRICS_DIR, "performance_analysis.json"),
            {"last_update": _utc(), "performance": res or {}},
        )
        _log("[PERF] Günlük performans analizi ve cooldown ayarları güncellendi.")
        return {"ok": True, "data": res}
    except Exception as e:
        _log(f"[ERR-PERF] {e}")
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Hyperparameter search task
#
# This task runs ``hyperparameter_search.py`` to tune the logistic regression
# hyperparameters used in the master confidence calibration.  The helper
# writes its output to ``data/hyperparameters.json``.  Running this script
# can take several minutes; adjust the timeout via the ``TRAIN_TIMEOUT``
# environment variable if necessary.
def task_hyperparameter_search() -> dict:
    """Run the hyperparameter search script and log progress.

    Returns a dictionary indicating success or failure.  Any exceptions
    encountered during the subprocess call are logged and surfaced via the
    'error' field.
    """
    try:
        script_path = os.path.join(ROOT, "hyperparameter_search.py")
        if not os.path.exists(script_path):
            _log("[HPS] hyperparameter_search.py bulunamadı; görev atlandı.")
            return {"ok": False, "error": "script_missing"}
        _log("[HPS] Hiperparametre taraması başlatılıyor...")
        # Run the script using the same Python interpreter; capture output to log
        cmd = [sys.executable, script_path]
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=SUBPROC_ENV,
                capture_output=True,
                text=True,
                timeout=TRAIN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            _log("[HPS] Hiperparametre taraması zaman aşımına uğradı.")
            return {"ok": False, "error": "timeout"}
        if result.returncode != 0:
            _log(f"[HPS] Hiperparametre taraması hata kodu {result.returncode} ile tamamlandı.\n{result.stderr}")
            return {"ok": False, "error": f"returncode_{result.returncode}"}
        _log("[HPS] Hiperparametre taraması başarıyla tamamlandı.")
        # Hiperparametre sonuçlarını config.json'a aktar
        try:
            hp_path = os.path.join(ROOT, "data", "hyperparameters.json")
            cfg_path = os.path.join(ROOT, "config.json")
            if os.path.isfile(hp_path) and os.path.isfile(cfg_path):
                with open(hp_path, "r", encoding="utf-8") as f:
                    hp_data = json.load(f)
                best = hp_data.get("best_params") if isinstance(hp_data, dict) else None
                if isinstance(best, dict):
                    cfg = _load_json(cfg_path, {})
                    # Store logistic regression parameters under a dedicated key
                    cfg.setdefault("logistic_params", {}).update(best)
                    _save_json(cfg_path, cfg)
                    _log(f"[HPS] En iyi hiperparametreler config.json'a yazıldı: {best}")
        except Exception as exc:
            _log(f"[HPS] Hiperparametrelerin config.json'a aktarımı sırasında hata: {exc}")
        return {"ok": True}
    except Exception as e:
        _log(f"[ERR-HPS] {e}")
        return {"ok": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Risk optimisation task
#
# This task invokes ``risk_optimizer.py`` which adjusts risk tiers based on
# recent volatility statistics.  The resulting configuration is saved in
# ``risk_config.json``.  Frequent invocation of this task allows the
# trading bot to respond to changes in market volatility by tuning
# leverage and confidence thresholds.
def task_risk_optimize() -> dict:
    """Run the risk optimiser and update the risk configuration."""
    try:
        script_path = os.path.join(ROOT, "risk_optimizer.py")
        if not os.path.exists(script_path):
            _log("[RISK-OPT] risk_optimizer.py bulunamadı; görev atlandı.")
            return {"ok": False, "error": "script_missing"}
        _log("[RISK-OPT] Risk optimizasyonu başlatılıyor...")
        cmd = [sys.executable, script_path]
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=SUBPROC_ENV,
                capture_output=True,
                text=True,
                timeout=TRAIN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            _log("[RISK-OPT] Risk optimizasyonu zaman aşımına uğradı.")
            return {"ok": False, "error": "timeout"}
        if result.returncode != 0:
            _log(f"[RISK-OPT] Risk optimizasyonu hata kodu {result.returncode} ile tamamlandı.\n{result.stderr}")
            return {"ok": False, "error": f"returncode_{result.returncode}"}
        _log("[RISK-OPT] Risk optimizasyonu başarıyla tamamlandı.")
        # risk_config.json içeriğini config.json'a aktar
        try:
            rc_path = os.path.join(ROOT, "risk_config.json")
            cfg_path = os.path.join(ROOT, "config.json")
            if os.path.isfile(rc_path) and os.path.isfile(cfg_path):
                with open(rc_path, "r", encoding="utf-8") as f:
                    rc_data = json.load(f)
                if isinstance(rc_data, dict):
                    cfg = _load_json(cfg_path, {})
                    # Store risk config under dedicated key
                    cfg["risk_config"] = rc_data
                    _save_json(cfg_path, cfg)
                    _log("[RISK-OPT] risk_config.json içerikleri config.json'a yazıldı.")
        except Exception as exc:
            _log(f"[RISK-OPT] risk_config aktarma hatası: {exc}")
        return {"ok": True}
    except Exception as e:
        _log(f"[ERR-RISK-OPT] {e}")
        return {"ok": False, "error": str(e)}

def task_train_transformer() -> dict:
    try:
        from ml.transformer_model import train_transformer_model  # type: ignore
    except Exception as e:
        _log(f"[ERR-TRANS-IMP] Could not import transformer trainer: {e}")
        return {"ok": False, "error": str(e)}


def task_update_social_trends() -> dict:
    """Refresh the social trends file using the social_scanner module.

    This task invokes ``social_scanner.update_social_trends`` to fetch
    trending coins from CoinMarketCap and LunarCrush (subject to
    available API keys) and writes the result into ``data/social_trends.json``.
    Returns a dict with the result status and the list of updated
    symbols.  Any exceptions are logged and surfaced in the return
    value.
    """
    try:
        from social_scanner import update_social_trends  # type: ignore
        syms = update_social_trends()  # type: ignore[call-arg]
        if isinstance(syms, list):
            return {"ok": True, "symbols": syms}
        return {"ok": True, "symbols": []}
    except Exception as e:
        _log(f"[ERR-SOCIAL-TRENDS] {e}")
        return {"ok": False, "error": str(e)}
    try:
        import math, random, json
        from collections import Counter
        series: list[float] = []
        ohlc_path = os.path.join(METRICS_DIR, "ohlc_history.json")
        try:
            if os.path.isfile(ohlc_path):
                with open(ohlc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rows = data.get("rows") if isinstance(data, dict) else data
                if isinstance(rows, list) and rows:
                    counts = Counter(
                        str(r.get("symbol")) for r in rows if isinstance(r, dict) and r.get("symbol") is not None
                    )
                    if counts:
                        sym, _ = counts.most_common(1)[0]
                        closes = [
                            float(r["close"])
                            for r in rows
                            if isinstance(r, dict) and r.get("symbol") == sym and "close" in r
                        ]
                        if closes:
                            series = closes[-500:]
        except Exception:
            series = []
        if not series:
            series = [100.0 + math.sin(i / 10.0) * 5.0 + random.random() for i in range(500)]
        model_path = os.path.join(MODELS_DIR, "price_transformer.pt")
        train_transformer_model(series, model_path=model_path, context_length=30, epochs=5, batch_size=32)
        _save_json(
            os.path.join(METRICS_DIR, "transformer_train.json"),
            {"last_update": _utc(), "samples": len(series)}
        )
        _log("[TRANSFORMER] Transformer modeli eğitildi ve kaydedildi.")
        return {"ok": True}
    except Exception as e:
        _log(f"[ERR-TRANSFORMER] {e}")
        return {"ok": False, "error": str(e)}

TASKS = {
    "data_update": task_data_update,
    "weight_patch": task_weight_patch,
    # Re‑enable model training tasks.  These spawn subprocesses to train
    # the BiLSTM and RL agents and will respect the timeouts defined at
    # the top of this module.
    "train_bilstm": lambda: task_train_bilstm(),
    "train_rl": lambda: task_train_rl(),
    "risk": task_risk,
    "update_sentiment": lambda: task_update_sentiment(),
    "update_onchain": lambda: task_update_onchain(),
    "model_monitor": lambda: task_model_monitor(),
    "param_optimize": lambda: task_param_optimize(),
    "update_macro": lambda: task_update_macro(),
    "update_external": lambda: task_update_external(),
    "train_transformer": lambda: task_train_transformer(),
    # Update social trends (trending coins) by calling the social_scanner
    "update_social_trends": lambda: task_update_social_trends(),
    # Additional provider update tasks.  These call individual providers
    # (options, liquidation, whale) and store results into metrics files.
    # They are disabled by default in the update plan; you may add them
    # to UPDATE_PLAN if you prefer running them separately.
    "update_ops": lambda: task_update_ops(),
    "update_liquidation": lambda: task_update_liquidation(),
    "update_whale": lambda: task_update_whale(),
    # Daily performance analysis to compute PnL stats and adjust cooldowns
    "performance_analysis": lambda: task_performance_analysis(),
    # Run hyperparameter search (logistic regression calibration)
    "hyperparameter_search": lambda: task_hyperparameter_search(),
    # Run risk tier optimisation
    "risk_optimize": lambda: task_risk_optimize(),
}

def run_update_cycle():
    st = _load_json(STATE_PATH, {"last": {}})
    for name, days in UPDATE_PLAN.items():
        last = st["last"].get(name, "")
        if _due(last, days):
            _log(f"[RUN] {name}...")
            TASKS[name]()
            st["last"][name] = _utc()
            _save_json(STATE_PATH, st)

def start_background(interval_min=60):
    def worker():
        _log("[BG] AutoUpdater servisi başladı.")
        while True:
            try: run_update_cycle()
            except Exception as e: _log(f"[BG-ERR] {e}")
            time.sleep(interval_min * 60)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

if __name__ == "__main__":
    run_update_cycle()

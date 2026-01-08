"""
model_performance_monitor.py
---------------------------

This module monitors the performance of the various AI components used in the
trading bot—namely the BiLSTM classifier, reinforcement learning agent(s),
and large language model (LLM) decision layer.  It provides utilities to
summarise performance metrics from the trade log and AI prediction logs and
to adjust the model weights in ``config.json`` based on these metrics.

The goal is to identify underperforming models and reduce their influence
without completely disabling them.  By periodically running this module,
operators can maintain healthy risk management by ensuring that only the
most effective models have a high impact on the master decision layer.

Example usage::

    from model_performance_monitor import evaluate_models, adjust_weights
    report = evaluate_models()
    print(json.dumps(report, indent=2))
    adjust_weights(report, config_path="config.json")

The functions assume that ``trade_log.json`` contains a list of trade
dictionaries with at least ``pnl_abs`` (absolute PnL) and optionally
``model`` or ``source`` fields that indicate which model produced the trade.
If these fields are missing, metrics will be aggregated under ``overall``.
Likewise, ``metrics/ai_predictions.json`` is expected to be a list of
prediction dictionaries with keys ``model``, ``prediction`` and
``actual``.  See ``ai_batch_manager.py`` for examples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    """Load JSON if the file exists; return None otherwise."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Failed to load JSON from %s: %s", path, exc)
        return None


def _compute_sign_accuracy(predictions: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Compute sign-based accuracy for a list of predictions.

    Predictions are dictionaries with keys ``prediction`` and ``actual``.
    A prediction is considered correct if ``prediction`` and ``actual`` have
    the same sign (both >= 0 or both < 0).  Returns the number of correct
    predictions, total number of predictions, and accuracy (0.0 if none).

    :param predictions: list of dicts with ``prediction`` and ``actual`` keys
    :return: (num_correct, total, accuracy)
    """
    correct = 0
    total = 0
    for record in predictions:
        try:
            pred = float(record.get("prediction", 0.0))
            actual = float(record.get("actual", 0.0))
        except Exception:
            continue
        if (pred >= 0 and actual >= 0) or (pred < 0 and actual < 0):
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else 0.0
    return correct, total, acc


def evaluate_models(
    trade_log_path: Path = Path("trade_log.json"),
    ai_pred_path: Path = Path("metrics/ai_predictions.json"),
) -> Dict[str, Any]:
    """Compute a performance report for each AI component.

    :param trade_log_path: path to the trade log JSON file.
    :param ai_pred_path: path to the AI predictions JSON file.
    :return: nested dictionary of performance metrics.
    """
    report: Dict[str, Any] = {
        "overall": {},
        "bilstm": {},
        "rl": {},
        "chatgpt": {},
    }
    # Load trade log and AI predictions
    trade_log = _load_json(trade_log_path)
    ai_preds = _load_json(ai_pred_path)
    # Process trade log
    if isinstance(trade_log, list):
        total_pnl = 0.0
        total_trades = 0
        model_pnl: Dict[str, float] = {}
        model_count: Dict[str, int] = {}
        for trade in trade_log:
            pnl_abs = float(trade.get("pnl_abs", 0.0))
            total_pnl += pnl_abs
            total_trades += 1
            model_name = str(trade.get("model") or trade.get("source") or "unknown").lower()
            model_pnl[model_name] = model_pnl.get(model_name, 0.0) + pnl_abs
            model_count[model_name] = model_count.get(model_name, 0) + 1
        report["overall"] = {
            "num_trades": total_trades,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": (total_pnl / total_trades) if total_trades else None,
        }
        # summarise RL PnL if any
        rl_pnl = 0.0
        rl_count = 0
        for m, pnl in model_pnl.items():
            if m.startswith("rl") or "ppo" in m:
                rl_pnl += pnl
                rl_count += model_count.get(m, 0)
        if rl_count > 0:
            report["rl"] = {
                "num_trades": rl_count,
                "total_pnl": rl_pnl,
                "avg_pnl": rl_pnl / rl_count,
            }
    else:
        report["overall"] = None

    # Process AI predictions
    if isinstance(ai_preds, list):
        preds_by_model: Dict[str, List[Dict[str, Any]]] = {}
        for rec in ai_preds:
            m = str(rec.get("model", "unknown")).lower()
            preds_by_model.setdefault(m, []).append(rec)
        for m, preds in preds_by_model.items():
            correct, total, acc = _compute_sign_accuracy(preds)
            report.setdefault(m, {})["sign_accuracy"] = acc
            report[m]["num_predictions"] = total
            report[m]["num_correct"] = correct
    return report


def adjust_weights(report: Dict[str, Any], config_path: Path = Path("config.json")) -> bool:
    """Adjust model weights in the configuration based on performance.

    The function reads ``config.json`` and reduces the weight of underperforming
    models according to simple heuristics:

      * If the RL agent has a negative average PnL, reduce ``weights.rl`` by 10%.
      * If the BiLSTM accuracy is below 0.5, reduce ``weights.bilstm`` by 15%.
      * If the ChatGPT/LLM sign accuracy is below 0.5 (if provided), reduce
        ``weights.chatgpt`` by 10%.

    Weights are clipped at zero.  The modified configuration is written back
    to disk.  Returns ``True`` if the file was updated.

    :param report: performance report returned by ``evaluate_models``
    :param config_path: path to the JSON config file
    :return: True if the config file was updated, else False
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logger.warning("Could not read config: %s", exc)
        return False
    weights = config.setdefault("weights", {})
    modified = False
    # RL adjustment
    rl_metrics = report.get("rl")
    if isinstance(rl_metrics, dict):
        avg_pnl = rl_metrics.get("avg_pnl")
        if isinstance(avg_pnl, (float, int)) and avg_pnl < 0:
            old = weights.get("rl", 1.0)
            new = max(0.0, old * 0.9)
            if new != old:
                weights["rl"] = new
                logger.info("Reduced RL weight from %.3f to %.3f due to negative PnL", old, new)
                modified = True
    # BiLSTM adjustment
    bilstm_metrics = report.get("bilstm")
    if isinstance(bilstm_metrics, dict):
        acc = bilstm_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < 0.5:
            old = weights.get("bilstm", 1.0)
            new = max(0.0, old * 0.85)
            if new != old:
                weights["bilstm"] = new
                logger.info("Reduced BiLSTM weight from %.3f to %.3f due to low accuracy", old, new)
                modified = True
    # ChatGPT/LLM adjustment
    chat_metrics = report.get("chatgpt")
    if isinstance(chat_metrics, dict):
        acc = chat_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < 0.5:
            old = weights.get("chatgpt", 1.0)
            new = max(0.0, old * 0.9)
            if new != old:
                weights["chatgpt"] = new
                logger.info("Reduced ChatGPT weight from %.3f to %.3f due to low accuracy", old, new)
                modified = True
    if modified:
        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as exc:
            logger.warning("Failed to write updated config: %s", exc)
            return False
    return False


def deactivate_underperforming_models(
    report: Dict[str, Any],
    config_path: Path = Path("config.json"),
    pnl_threshold: float = -0.05,
    accuracy_threshold: float = 0.25,
) -> bool:
    """Disable models that persistently underperform by setting their weights to zero.

    This function is a more aggressive variant of ``adjust_weights``.  It
    examines the performance report and, if a model's metrics fall
    below the specified thresholds, sets its weight in the configuration
    to zero and marks it as disabled.  Disabled models are recorded in
    the ``disabled_models`` list within the config file.

    :param report: performance report returned by ``evaluate_models``
    :param config_path: path to the JSON config file
    :param pnl_threshold: minimum acceptable average PnL for RL models (more tolerant default -0.05)
    :param accuracy_threshold: minimum acceptable sign accuracy for classifiers (BiLSTM/LLM).  Lower values are more tolerant.
    :return: True if the config was updated, else False
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logger.warning("Could not read config for deactivation: %s", exc)
        return False
    weights = config.setdefault("weights", {})
    disabled = config.setdefault("disabled_models", [])
    modified = False
    # RL deactivation
    rl_metrics = report.get("rl")
    if isinstance(rl_metrics, dict):
        avg_pnl = rl_metrics.get("avg_pnl")
        if isinstance(avg_pnl, (float, int)) and avg_pnl < pnl_threshold:
            if weights.get("rl", 0.0) > 0.0:
                weights["rl"] = 0.0
                if "rl" not in disabled:
                    disabled.append("rl")
                logger.info(
                    "Deactivating RL model due to low average PnL %.4f (threshold %.4f)",
                    avg_pnl,
                    pnl_threshold,
                )
                modified = True
    # BiLSTM deactivation
    bilstm_metrics = report.get("bilstm")
    if isinstance(bilstm_metrics, dict):
        acc = bilstm_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < accuracy_threshold:
            if weights.get("bilstm", 0.0) > 0.0:
                weights["bilstm"] = 0.0
                if "bilstm" not in disabled:
                    disabled.append("bilstm")
                logger.info(
                    "Deactivating BiLSTM model due to low accuracy %.4f (threshold %.4f)",
                    acc,
                    accuracy_threshold,
                )
                modified = True
    # LLM/ChatGPT deactivation
    chat_metrics = report.get("chatgpt")
    if isinstance(chat_metrics, dict):
        acc = chat_metrics.get("sign_accuracy")
        if isinstance(acc, (float, int)) and acc < accuracy_threshold:
            if weights.get("chatgpt", 0.0) > 0.0:
                weights["chatgpt"] = 0.0
                if "chatgpt" not in disabled:
                    disabled.append("chatgpt")
                logger.info(
                    "Deactivating ChatGPT model due to low accuracy %.4f (threshold %.4f)",
                    acc,
                    accuracy_threshold,
                )
                modified = True
    if modified:
        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as exc:
            logger.warning("Failed to write updated config during deactivation: %s", exc)
            return False
    return False


if __name__ == "__main__":  # pragma: no cover
    rep = evaluate_models()
    print(json.dumps(rep, indent=2))
    adjust_weights(rep)
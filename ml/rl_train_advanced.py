"""
rl_train_advanced.py
--------------------

This script extends the basic PPO training setup to support multiple
reinforcement learning algorithms from the ``stable_baselines3`` library
including PPO (Proximal Policy Optimisation), A2C (Advantage Actor-Critic)
and DQN (Deep Q-Network).  It allows selection of the algorithm via a
command-line argument and stores the resulting model under a file name
corresponding to the chosen algorithm.

The environment class ``CryptoEnv`` should implement the OpenAI Gym API
(``reset`` and ``step`` methods) and is assumed to be defined in
``ml/rl_env.py``.  This script also logs basic training metrics to a
``metrics/rl_metrics.json`` file for later analysis.

Example usage::

    python3 -m ml.rl_train_advanced --algo ppo --steps 100_000 --env_config configs/env.json

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

# Use gymnasium when available; fallback to classic gym
try:
    import gymnasium as gym  # type: ignore
except Exception:
    # gymnasium may not be installed; fall back to gym
    import gym  # type: ignore
import numpy as np

# Stable‑Baselines3 import
# The RL algorithms live in stable_baselines3, but this library may not
# be installed in all environments.  Attempt to import them and fall
# back to dummy placeholders if unavailable.  When the imports fail,
# training is skipped gracefully so that the rest of the bot continues
# operating without raising runtime errors.
try:
    from stable_baselines3 import PPO, A2C, DQN  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
except Exception:  # pragma: no cover
    PPO = None  # type: ignore
    A2C = None  # type: ignore
    DQN = None  # type: ignore
    class DummyVecEnv:  # type: ignore
        """Fallback DummyVecEnv when stable_baselines3 is missing."""
        def __init__(self, *args, **kwargs) -> None:
            pass

# Import the multi‑coin trading environment.  The original version of this
# script referred to a ``CryptoEnv`` class, but the environment is actually
# defined as ``MultiCoinTradeEnv`` in ``ml/rl_env.py``.  Import it and alias
# it back to CryptoEnv for backwards compatibility.
# MultiCoinTradeEnv requires a multi-coin OHLC dataframe (full_df). The legacy
# trainer already knows how to build this dataframe from
# metrics/ohlc_history.json via `load_env_from_metrics_multi`, so we reuse it
# here to keep behaviour consistent and avoid runtime errors.
from ml.rl_env import load_env_from_metrics_multi

logger = logging.getLogger(__name__)


ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agent with configurable algorithm")
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()),
                        help="Which algorithm to use (ppo, a2c, dqn)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Number of training steps")
    parser.add_argument("--env_config", type=str, default=None,
                        help="Path to JSON configuration for the environment")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save trained models")
    return parser.parse_args()


def load_env(config_path: Optional[str] = None) -> gym.Env:
    """Initialise the multi-coin trading environment.

    `MultiCoinTradeEnv` requires a multi-coin OHLC dataframe (`full_df`).
    The simplest and most robust way to obtain this dataframe is to reuse the
    same builder used by the legacy trainer (`load_env_from_metrics_multi`),
    which reads `metrics/ohlc_history.json`.

    The optional `config_path` may override a subset of environment parameters.
    Supported keys:
      - window (int)
      - max_reward (float)
      - fee (float)
    """

    env_config: Dict[str, Any] = {}
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                env_config = json.load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load env config: %s", exc)

    window = int(env_config.get("window", 60))
    max_reward = float(env_config.get("max_reward", 0.05))
    fee = float(env_config.get("fee", 0.0006))

    env = load_env_from_metrics_multi(window=window, max_reward=max_reward)
    # Allow fee override without breaking the constructor signature.
    try:
        env.fee = fee
    except Exception:
        pass
    return env


def train_agent(algo: str, total_steps: int, env: gym.Env, model_dir: str) -> str:
    """Train the specified RL agent and return the path to the saved model."""
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algorithm: {algo}")
    algo_cls = ALGOS[algo]
    # If stable_baselines3 could not be imported, skip training
    if algo_cls is None:
        logger.warning(
            "Stable‑Baselines3 not available; skipping RL training for %s."
            " Set up stable_baselines3>=2.5.0 to enable advanced RL.",
            algo.upper(),
        )
        return ""
    # Wrap environment in a DummyVecEnv for algorithms that require vectorised envs
    vec_env = DummyVecEnv([lambda: env])
    model = algo_cls("MlpPolicy", vec_env, verbose=1)
    logger.info("Training %s agent for %d steps", algo.upper(), total_steps)
    model.learn(total_timesteps=total_steps)
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = model_dir_path / f"{algo}_agent.zip"
    model.save(str(model_path))
    logger.info("Saved %s model to %s", algo.upper(), model_path)
    # log metrics
    metrics = {
        "algorithm": algo,
        "steps": total_steps,
    }
    try:
        metrics_path = Path("metrics/rl_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write RL metrics: %s", exc)
    return str(model_path)


def main() -> None:
    args = parse_args()
    env = load_env(args.env_config)
    train_agent(args.algo, args.steps, env, args.model_dir)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
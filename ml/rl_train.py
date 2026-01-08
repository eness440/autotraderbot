"""
ml/rl_train.py
=================

This script trains a multi-asset reinforcement learning agent using the
Proximal Policy Optimization (PPO) algorithm from `stable_baselines3`.  It
expects a custom environment provided by ``rl_env.load_env_from_metrics_multi``
which yields observations based on historical market data and metrics.  The
trained policy is saved under the ``models/`` directory (e.g.
``models/ppo_multi.zip``) along with a normalisation wrapper (``vecnormalize_multi.pkl``).

Command line arguments:
    --window:  The number of lookback steps (e.g. 60) for the environment
    --timesteps:  The total number of training timesteps (default 1e6).  For
        quick refreshes or testing, you may specify a lower value (e.g.
        10000) when invoking via the auto-updater.
    --device:  "cuda" or "cpu" to control where the model runs.  Defaults to
        "cuda" for best performance but falls back gracefully if no GPU is
        available.

Example:
    ``python rl_train.py --window 60 --timesteps 100000 --device cpu``

Note:
    PPO is chosen for its stability and strong performance on continuous
    control tasks.  Should you wish to experiment with other algorithms
    (e.g. DQN, A2C), you can modify this script accordingly, but the rest
    of the system currently assumes PPO checkpoints.
"""
import argparse
import pathlib
import json
import os
import time

# Attempt to import stable_baselines3.  If unavailable, training will be skipped.
try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore
    from stable_baselines3.common.utils import get_linear_fn  # type: ignore
except Exception:
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    VecNormalize = None  # type: ignore
    def get_linear_fn(*args, **kwargs):  # type: ignore
        return None

from .rl_env import load_env_from_metrics_multi

MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- ATOMIC SAVE HELPERS ---------------- #
def _atomic_replace(src: pathlib.Path, dst: pathlib.Path, retries: int = 40, delay_s: float = 0.25) -> None:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    if last_err is not None:
        raise last_err


def _tmp_path_for(dst: pathlib.Path) -> pathlib.Path:
    """
    SB3, path .zip değilse otomatik .zip ekleyebildiği için:
    tmp adı da aynı suffix ile bitsin (örn: ppo_multi.tmp_x.zip)
    """
    stamp = f"{int(time.time())}_{os.getpid()}"
    return dst.with_name(f"{dst.stem}.tmp_{stamp}{dst.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--device", type=str, default="cuda")  # istersen "cpu" yapabilirsin
    args = ap.parse_args()

    # Eğer PPO veya VecNormalize kullanılamıyorsa, eğitim atlanır.
    if PPO is None or DummyVecEnv is None or VecNormalize is None:
        print("[RL-TRAIN] stable_baselines3 bulunamadı veya yüklenemedi. RL eğitimi atlanıyor.")
        return

    # Ortamı bir kere oluşturup kaç coin olduğunu görelim
    temp_env = load_env_from_metrics_multi(window=args.window)
    coin_count = getattr(temp_env, "n_symbols", None)
    if coin_count is None:
        # Eski env ile uyumluluk için
        try:
            coin_count = temp_env.full_df["symbol"].nunique()
        except Exception:
            coin_count = 1

    print(f"🚀 RL Eğitimi Başlatılıyor... (Toplam Coin: {coin_count})")

    def make_env():
        # Her worker kendi MultiCoinTradeEnv kopyasını alsın
        return load_env_from_metrics_multi(window=args.window)

    # 1 adet vektörize env (istersen arttırabilirsin)
    env = DummyVecEnv([make_env])

    #
    # ==== Learning rate schedule and hyperparameters ====
    #
    # The original script used ``get_linear_fn`` from stable_baselines3 to
    # generate a simple linear schedule between a start and end learning rate.
    # Recent versions of stable_baselines3 have deprecated this helper in
    # favour of ``LinearSchedule`` or custom callables.  To avoid relying on
    # deprecated functions, we implement a small linear schedule here.  The
    # schedule takes a ``progress`` argument in the range [0,1], where 1
    # corresponds to the start of training and 0 to the end.  It returns
    # an interpolated learning rate between ``START_LR`` and ``END_LR``.
    # A lower learning rate and longer rollouts/batches help stabilise PPO
    # updates and keep the KL divergence and clip fraction within a healthy
    # range.  See user feedback in the review notes for details.
    START_LR = 2.5e-4  # initial learning rate (reduced from 3e-4)
    END_LR = START_LR * 0.1  # decay to 10% of the starting LR

    def lr_schedule(progress: float) -> float:
        """Linear decay of the learning rate over training progress."""
        # progress==1 at the start of training, 0 at the end
        return float(END_LR + progress * (START_LR - END_LR))


    model_path = MODELS_DIR / "ppo_multi.zip"
    stats_path = MODELS_DIR / "vecnormalize_multi.pkl"

    # Resume vs cold start.  When resuming, verify that the saved model's
    # observation and action spaces match the current environment.  If the
    # observation dimensionality has changed (for example due to a different
    # window size or feature set), loading the checkpoint may lead to
    # shape mismatches in the policy network.  In such cases, start from
    # scratch and ignore the old weights.  Also use more conservative
    # hyperparameters to improve PPO stability: longer rollout (n_steps),
    # larger batch size and a smaller clip range and entropy coefficient.
    if model_path.exists() and stats_path.exists():
        print(f"🔄 Mevcut model bulundu: {model_path}")
        print("   Eğitim kaldığı yerden devam edecek.")

        # Load normalisation statistics
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True

        try:
            # Attempt to load the saved model
            loaded_model = PPO.load(
                model_path,
                env=env,
                device=args.device,
                custom_objects={"learning_rate": lr_schedule},
            )
            # Compare observation and action spaces
            obs_shape_current = getattr(env.observation_space, "shape", None)
            obs_shape_loaded = getattr(loaded_model.observation_space, "shape", None)
            act_space_current = getattr(env.action_space, "n", None) or getattr(env.action_space, "shape", None)
            act_space_loaded = getattr(loaded_model.action_space, "n", None) or getattr(loaded_model.action_space, "shape", None)
            if obs_shape_current != obs_shape_loaded or act_space_current != act_space_loaded:
                print(
                    "⚠️ Uyarı: Kayıtlı modelin gözlem/aksiyon boyutu mevcut ortamla uyuşmuyor. "
                    "Parametre seti değişmiş olabilir, bu yüzden yeni bir model başlatılıyor."
                )
                raise ValueError("Model/Env dimension mismatch")
            model = loaded_model
        except Exception as err:
            print(f"🔁 Model yüklenemedi: {err}. Sıfırdan başlayacak...")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=args.device,
                learning_rate=lr_schedule,
                n_steps=4096,
                batch_size=256,
                ent_coef=0.002,
                vf_coef=0.5,
                n_epochs=10,
                clip_range=0.15,
                target_kl=0.03,
                max_grad_norm=0.5,
            )
    else:
        print("🆕 Sıfırdan eğitim başlıyor.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=args.device,
            learning_rate=lr_schedule,
            n_steps=4096,
            batch_size=256,
            ent_coef=0.002,
            vf_coef=0.5,
            n_epochs=10,
            clip_range=0.15,
            target_kl=0.03,
            max_grad_norm=0.5,
        )

    # Eğitim
    print(f"🏋️‍♂️ Eğitim döngüsü: {args.timesteps:,} adım.")
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)

    # Kayıt (atomic)
    tmp_model_path = _tmp_path_for(model_path)
    tmp_stats_path = _tmp_path_for(stats_path)

    try:
        model.save(str(tmp_model_path))
        env.save(str(tmp_stats_path))

        # Önce stats sonra model (ikisi de “tutarlı” şekilde güncellensin)
        _atomic_replace(tmp_stats_path, stats_path)
        _atomic_replace(tmp_model_path, model_path)

        print(f"✅ RL Eğitimi tamamlandı ve kaydedildi -> {model_path}")
    except Exception as e:
        for p in (tmp_model_path, tmp_stats_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        raise e

    # ----- Metrics Logging -----
    # After training completes, record the last update time so that the
    # Streamlit dashboard can display when the RL model was last refreshed.
    try:
        from datetime import datetime, timezone
        # Determine the metrics directory relative to this script
        root_dir = pathlib.Path(__file__).resolve().parents[1]
        metrics_dir = root_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rl_metrics_path = metrics_dir / "rl_metrics.json"
        with open(rl_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "timesteps": int(args.timesteps),
                    "window": int(args.window),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] RL metrics kaydedildi -> {rl_metrics_path}")
    except Exception as e:
        print(f"[WARN] RL metrics kaydedilemedi: {e}")


if __name__ == "__main__":
    main()

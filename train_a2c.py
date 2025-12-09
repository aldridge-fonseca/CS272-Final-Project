import os
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import LinearSchedule
from group5_custom_env import register_group5_env

register_group5_env()

os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

print("=" * 70)
print("Training A2C Agent on Group 5 Custom Environment")
print("=" * 70)

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SAVE_FREQ = 25_000
N_EVAL_EPISODES = 10

print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
train_env = Monitor(train_env, './logs/a2c/train')
train_env = DummyVecEnv([lambda: train_env])

print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/a2c/eval')
eval_env = DummyVecEnv([lambda: eval_env])

print("\nInitializing A2C agent...")
print("  Policy: MlpPolicy")
print("  Learning rate: Linear schedule (3e-4 -> 1e-4)")
print("  Gamma: 0.99")
print("  N steps: 5")

learning_rate_schedule = LinearSchedule(3e-4, 1e-4, end_fraction=0.1)

model = A2C(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/a2c/tensorboard",
    learning_rate=learning_rate_schedule,
    gamma=0.99,
    n_steps=5,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

print("\nSetting up callbacks...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/a2c/',
    log_path='./logs/a2c/eval',
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./models/a2c/checkpoints/',
    name_prefix='a2c_checkpoint',
)

print("\n" + "=" * 70)
print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("=" * 70)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)

print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)

final_model_path = './models/a2c_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

monitor_file = './logs/a2c/train/monitor.csv'
if os.path.exists(monitor_file):
    try:
        df = pd.read_csv(monitor_file, skiprows=1)
        plt.figure(figsize=(10, 5))
        plt.plot(df['r'], alpha=0.3, label='Episode Reward')
        rolling_mean = df['r'].rolling(window=50).mean()
        plt.plot(rolling_mean, label='Rolling Mean (50 episodes)', color='orange')
        plt.title("A2C Learning Curve")
        plt.xlabel("Training Episodes")
        plt.ylabel("Mean Episodic Reward (Return)")
        plt.grid(True)
        plt.legend()
        plt.savefig("./plots/learning_curve_a2c.png")
        print("Learning curve saved to ./plots/learning_curve_a2c.png")
    except Exception as e:
        print(f"Could not plot learning curve: {e}")

print("\nTraining complete!")

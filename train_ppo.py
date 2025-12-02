"""
Train a PPO (Proximal Policy Optimization) agent on the Group 5 Custom Environment.

This script:
1. Creates the custom environment
2. Trains a PPO agent using Stable-Baselines3
3. Saves the trained model
4. Logs training metrics to TensorBoard

Usage:
    python train_ppo.py

The trained model will be saved to ./models/ppo_group5_env.zip
Training logs will be saved to ./logs/ppo/
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import LinearSchedule
from group5_custom_env import register_group5_env

# Register the custom environment
register_group5_env()

# Create directories for models and logs
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("=" * 70)
print("Training PPO Agent on Group 5 Custom Environment")
print("=" * 70)

# Training configuration
TOTAL_TIMESTEPS = 50_000  # Total training steps (reduced for quick testing)
EVAL_FREQ = 5_000  # Evaluate every N steps
SAVE_FREQ = 25_000  # Save checkpoint every N steps
N_EVAL_EPISODES = 10  # Number of episodes for evaluation

# Create training environment
print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
train_env = Monitor(train_env, './logs/ppo/train')
train_env = DummyVecEnv([lambda: train_env])

# Create evaluation environment
print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/ppo/eval')
eval_env = DummyVecEnv([lambda: eval_env])

# Create PPO agent with adaptive learning rate
print("\nInitializing PPO agent...")
print("  Policy: MlpPolicy")
print("  Learning rate: Linear schedule (3e-4 -> 1e-4)")
print("  Gamma: 0.99 (default)")
print("  Batch size: 64 (default)")
print("  N steps: 2048 (default)")

# Linear learning rate schedule: starts at 3e-4, decays to 1e-4 over training
# LinearSchedule(start, end, end_fraction): end_fraction=0.1 means end is reached at 10% remaining
learning_rate_schedule = LinearSchedule(3e-4, 1e-4, end_fraction=0.1)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/ppo/tensorboard",
    learning_rate=learning_rate_schedule,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
)

# Setup callbacks
print("\nSetting up callbacks...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/ppo/',
    log_path='./logs/ppo/eval',
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./models/ppo/checkpoints/',
    name_prefix='ppo_checkpoint',
)

# Start training
print("\n" + "=" * 70)
print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("=" * 70)
print("\nTraining progress will be logged to:")
print("  - TensorBoard: ./logs/ppo/tensorboard/")
print("  - Evaluation logs: ./logs/ppo/eval/")
print("\nTo view TensorBoard, run:")
print("  tensorboard --logdir ./logs/ppo/tensorboard/")
print("\n" + "-" * 70)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)

# Save final model
print("\n" + "=" * 70)
print("Training complete! Saving final model...")
print("=" * 70)

final_model_path = './models/ppo_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

# Print summary
print("\n" + "=" * 70)
print("Training Summary")
print("=" * 70)
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Model saved to: {final_model_path}")
print(f"  Best model saved to: ./models/ppo/best_model.zip")
print(f"  Checkpoints saved to: ./models/ppo/checkpoints/")
print(f"  TensorBoard logs: ./logs/ppo/tensorboard/")
print("\nTo evaluate the trained agent, run:")
print("  python compare_agents.py")
print("=" * 70)


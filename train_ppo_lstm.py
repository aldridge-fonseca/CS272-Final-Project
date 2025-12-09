import os
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import LinearSchedule
from group5_custom_env import register_group5_env

register_group5_env()

os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("=" * 70)
print("Training PPO with LSTM (Recurrent Policy)")
print("=" * 70)

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SAVE_FREQ = 25_000
N_EVAL_EPISODES = 10

print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
train_env = Monitor(train_env, './logs/ppo_lstm/train')
train_env = DummyVecEnv([lambda: train_env])

print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/ppo_lstm/eval')
eval_env = DummyVecEnv([lambda: eval_env])

print("\nInitializing RecurrentPPO with LSTM...")

learning_rate_schedule = LinearSchedule(3e-4, 1e-4, end_fraction=0.1)

model = RecurrentPPO(
    "MlpLstmPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/ppo_lstm/tensorboard",
    learning_rate=learning_rate_schedule,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    policy_kwargs=dict(
        lstm_hidden_size=256,
        n_lstm_layers=1,
        net_arch=[],
        enable_critic_lstm=True,
    ),
)

print("\nSetting up callbacks...")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/ppo_lstm/',
    log_path='./logs/ppo_lstm/eval',
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./models/ppo_lstm/checkpoints/',
    name_prefix='ppo_lstm_checkpoint',
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

final_model_path = './models/ppo_lstm_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

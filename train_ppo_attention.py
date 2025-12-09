import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import LinearSchedule
from group5_custom_env import register_group5_env

register_group5_env()

os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("=" * 70)
print("Training PPO with Self-Attention Mechanism")
print("=" * 70)


class SelfAttentionExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if len(observation_space.shape) == 1:
            input_dim = observation_space.shape[0]
        else:
            input_dim = int(np.prod(observation_space.shape))
        
        self.embedding_dim = 128
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU()
        )
        
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        self.output_norm = nn.LayerNorm(features_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.shape) > 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)
        
        embedded = self.input_embedding(observations)
        embedded = embedded.unsqueeze(1)
        
        attended, attention_weights = self.attention(
            embedded, embedded, embedded,
            need_weights=True
        )
        
        attended = self.attention_norm(embedded + attended)
        attended = attended.squeeze(1)
        
        features = self.ffn(attended)
        features = self.output_norm(features)
        
        return features


TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SAVE_FREQ = 25_000
N_EVAL_EPISODES = 10

print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
train_env = Monitor(train_env, './logs/ppo_attention/train')
train_env = DummyVecEnv([lambda: train_env])

print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/ppo_attention/eval')
eval_env = DummyVecEnv([lambda: eval_env])

print("\nInitializing PPO with Self-Attention...")

learning_rate_schedule = LinearSchedule(3e-4, 1e-4, end_fraction=0.1)

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/ppo_attention/tensorboard",
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
        features_extractor_class=SelfAttentionExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    ),
)

print("\nSetting up callbacks...")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/ppo_attention/',
    log_path='./logs/ppo_attention/eval',
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./models/ppo_attention/checkpoints/',
    name_prefix='ppo_attention_checkpoint',
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

final_model_path = './models/ppo_attention_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

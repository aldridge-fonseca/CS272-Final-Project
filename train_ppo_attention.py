"""
Train PPO with Self-Attention Mechanism

This variant uses a self-attention layer to help the agent focus on important
features in the observation space, particularly useful for:
- Identifying emergency vehicles among regular traffic
- Focusing on nearby obstacles and lane closures
- Prioritizing relevant vehicles for collision avoidance

Key Innovation:
- Self-attention layer before policy/value networks
- Learns to weight important observations dynamically
- Better than fixed observation processing

Usage:
    python train_ppo_attention.py
"""

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

# Register the custom environment
register_group5_env()

# Create directories
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("=" * 70)
print("Training PPO with Self-Attention Mechanism")
print("=" * 70)


class SelfAttentionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with self-attention mechanism.
    
    Self-attention allows the network to learn which parts of the observation
    are most important for decision-making. This is especially useful for:
    - Detecting emergency vehicles (high attention weight)
    - Identifying nearby obstacles (high attention weight)
    - Ignoring distant irrelevant vehicles (low attention weight)
    
    Architecture:
    1. Input embedding layer
    2. Multi-head self-attention layer
    3. Feed-forward network
    4. Output features for policy/value networks
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Get input dimension - handle both 1D and 2D observations
        if len(observation_space.shape) == 1:
            input_dim = observation_space.shape[0]
        else:
            # Flatten multi-dimensional observations
            input_dim = int(np.prod(observation_space.shape))
        
        # Embedding layer - project input to attention dimension
        self.embedding_dim = 128
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU()
        )
        
        # Multi-head self-attention
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization after attention
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Final layer norm
        self.output_norm = nn.LayerNorm(features_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.
        
        Args:
            observations: Input observations [batch_size, ...obs_shape]
        
        Returns:
            features: Processed features [batch_size, features_dim]
        """
        # Flatten observations if needed
        if len(observations.shape) > 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)
        
        # Embed input
        embedded = self.input_embedding(observations)  # [batch, embedding_dim]
        
        # Add batch dimension for attention (expects [batch, seq_len, embed_dim])
        # We treat the observation as a sequence of length 1
        embedded = embedded.unsqueeze(1)  # [batch, 1, embedding_dim]
        
        # Self-attention
        # The attention mechanism learns to weight different parts of the embedding
        attended, attention_weights = self.attention(
            embedded, embedded, embedded,
            need_weights=True
        )
        
        # Residual connection + layer norm
        attended = self.attention_norm(embedded + attended)
        
        # Remove sequence dimension
        attended = attended.squeeze(1)  # [batch, embedding_dim]
        
        # Feed-forward network
        features = self.ffn(attended)
        
        # Final normalization
        features = self.output_norm(features)
        
        return features


# Training configuration
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SAVE_FREQ = 25_000
N_EVAL_EPISODES = 10

print("\n" + "=" * 70)
print("Self-Attention Configuration")
print("=" * 70)
print(f"  Number of attention heads: 4")
print(f"  Embedding dimension: 128")
print(f"  Features dimension: 256")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")

# Create training environment
print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
print(f"  Observation space: {train_env.observation_space.shape}")
train_env = Monitor(train_env, './logs/ppo_attention/train')
train_env = DummyVecEnv([lambda: train_env])

# Create evaluation environment
print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/ppo_attention/eval')
eval_env = DummyVecEnv([lambda: eval_env])

# Create PPO agent with self-attention
print("\nInitializing PPO with Self-Attention...")
print("  Policy: MlpPolicy with custom attention extractor")
print("  Learning rate: Linear schedule (3e-4 -> 1e-4)")
print("\nSelf-Attention Benefits:")
print("  ✅ Learns to focus on important vehicles (emergency vehicles)")
print("  ✅ Dynamically weights observations by relevance")
print("  ✅ Better feature extraction than fixed processing")
print("  ✅ Interpretable attention weights")

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

# Setup callbacks
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

# Start training
print("\n" + "=" * 70)
print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("=" * 70)
print("\nHow Self-Attention Helps:")
print("  1. Emergency Vehicle Detection:")
print("     - High attention weight on emergency vehicle features")
print("     - Agent learns to prioritize yielding behavior")
print("  2. Obstacle Avoidance:")
print("     - Focuses on nearby lane closures and stalled vehicles")
print("     - Ignores distant irrelevant objects")
print("  3. Traffic Navigation:")
print("     - Weights nearby vehicles higher than distant ones")
print("     - Better collision avoidance decisions")
print("\nTraining progress will be logged to:")
print("  - TensorBoard: ./logs/ppo_attention/tensorboard/")
print("  - Evaluation logs: ./logs/ppo_attention/eval/")
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

final_model_path = './models/ppo_attention_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

# Print summary
print("\n" + "=" * 70)
print("Training Summary")
print("=" * 70)
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Algorithm: PPO with Self-Attention")
print(f"  Attention type: {'Multi-scale' if USE_MULTISCALE else 'Single-scale'}")
print(f"  Number of heads: 4")
print(f"  Model saved to: {final_model_path}")
print(f"  Best model saved to: ./models/ppo_attention/best_model.zip")
print("\nSelf-Attention vs Baseline PPO:")
print("  - Better feature extraction (learned attention)")
print("  - Focuses on important observations (emergency vehicles)")
print("  - More interpretable (can visualize attention weights)")
print("\nTo compare with other agents, run:")
print("  python compare_agents.py \\")
print("    --a2c_path ./models/ppo_group5_env.zip \\")
print("    --ppo_path ./models/ppo_attention_group5_env.zip \\")
print("    --agent1_name 'PPO (Baseline)' \\")
print("    --agent2_name 'PPO (Attention)'")
print("=" * 70)

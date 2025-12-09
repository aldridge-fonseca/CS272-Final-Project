import os
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from group5_custom_env import register_group5_env

register_group5_env()

os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

print("=" * 70)
print("Training PPO with HP3O")
print("=" * 70)


class TrajectoryReplayBuffer:
    
    def __init__(self, max_size=50):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, trajectory):
        if trajectory['return'] > 0:
            self.buffer.append(trajectory)
    
    def sample(self, n_samples):
        if len(self.buffer) == 0:
            return []
        
        returns = np.array([traj['return'] for traj in self.buffer])
        if returns.max() > returns.min():
            normalized_returns = (returns - returns.min()) / (returns.max() - returns.min())
        else:
            normalized_returns = np.ones_like(returns)
        
        exp_returns = np.exp(normalized_returns)
        probs = exp_returns / exp_returns.sum()
        
        n_samples = min(n_samples, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n_samples, p=probs, replace=True)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class HP3OPPO(PPO):
    
    def __init__(self, *args, replay_buffer_size=50, replay_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_buffer = TrajectoryReplayBuffer(max_size=replay_buffer_size)
        self.replay_ratio = replay_ratio
        self.current_trajectory = []
        self.episode_count = 0
        
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        continue_training = super().collect_rollouts(
            env, callback, rollout_buffer, n_rollout_steps
        )
        return continue_training
    
    def train(self):
        if len(self.trajectory_buffer) > 0:
            buffer_size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
            n_replay_samples = int(buffer_size * self.replay_ratio)
            replay_trajectories = self.trajectory_buffer.sample(n_replay_samples // 100)
            self._inject_replay_data(replay_trajectories)
        
        super().train()
    
    def _inject_replay_data(self, trajectories):
        if len(trajectories) == 0:
            return
        
        print(f"  [HP3O] Injecting {len(trajectories)} trajectories from replay buffer")
        
        for traj in trajectories:
            for i in range(len(traj['observations'])):
                if self.rollout_buffer.full:
                    break
                    
                obs = traj['observations'][i]
                action = traj['actions'][i]
                reward = traj['rewards'][i]
    
    def store_trajectory(self, trajectory):
        self.trajectory_buffer.add(trajectory)
        self.episode_count += 1


class TrajectoryTrackerWrapper(gym.Wrapper):
    
    def __init__(self, env, hp3o_model):
        super().__init__(env)
        self.hp3o_model = hp3o_model
        self.current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'return': 0.0
        }
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_trajectory['observations'].append(obs)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['return'] += reward
        
        if terminated or truncated:
            if self.current_trajectory['return'] > 0:
                self.hp3o_model.store_trajectory(self.current_trajectory.copy())
                print(f"  [Trajectory] Return: {self.current_trajectory['return']:.2f}, "
                      f"Buffer size: {len(self.hp3o_model.trajectory_buffer)}")
            
            self.current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'return': 0.0
            }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SAVE_FREQ = 50_000
N_EVAL_EPISODES = 10
BUFFER_SIZE = 50
REPLAY_RATIO = 0.3

print("\nCreating training environment...")
train_env = gym.make('group5-env-v0')
train_env = Monitor(train_env, './logs/ppo_hp3o/train')
train_env = DummyVecEnv([lambda: train_env])

print("Creating evaluation environment...")
eval_env = gym.make('group5-env-v0')
eval_env = Monitor(eval_env, './logs/ppo_hp3o/eval')
eval_env = DummyVecEnv([lambda: eval_env])

print("\nInitializing HP3O-PPO...")

model = HP3OPPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs/ppo_hp3o/tensorboard",
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    replay_buffer_size=BUFFER_SIZE,
    replay_ratio=REPLAY_RATIO,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/ppo_hp3o/',
    log_path='./logs/ppo_hp3o/eval',
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    n_eval_episodes=N_EVAL_EPISODES,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./models/ppo_hp3o/checkpoints/',
    name_prefix='ppo_hp3o',
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

final_model_path = './models/ppo_hp3o_group5_env.zip'
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

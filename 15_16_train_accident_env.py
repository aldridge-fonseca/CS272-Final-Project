"""
Training script for Team 6's AccidentEnv (accident-v0)
Experiment IDs: 15 (learning curve), 16 (performance test)

Uses PPO with tuned hyperparameters and parallel environments.
Generates learning curve and violin plot for evaluation.
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from custom_env import AccidentEnv

# Output directories
SAVE_DIR = "./results_opponent_env"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
LOG_DIR = os.path.join(SAVE_DIR, "logs")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Training hyperparameters
TOTAL_TIMESTEPS = 500_000
N_ENVS = 16  # parallel envs
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
LOG_EPISODE_FREQ = 1000
TEST_EPISODES = 500  # for violin plot

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


def make_env(rank, seed=0):
    """Factory function for creating training environments."""
    def _init():
        env = AccidentEnv(config={
            "observation": {"type": "LidarObservation"},
            "duration": 20,
        })
        env = Monitor(env, os.path.join(LOG_DIR, f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def make_eval_env():
    """Create a single environment for evaluation."""
    env = AccidentEnv(config={
        "observation": {"type": "LidarObservation"},
        "duration": 20,
    })
    env = Monitor(env, os.path.join(LOG_DIR, "eval"))
    return env


class RewardLoggingCallback(BaseCallback):
    """Logs episode rewards during training for plotting learning curve."""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        self.current_rewards = None
        self.log_freq = log_freq
        self.last_log_episode = 0
        
    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)
        print(f"\nTraining started with {self.training_env.num_envs} parallel environments")
        print(f"Logging every {self.log_freq} episodes\n")
        
    def _on_step(self):
        self.current_rewards += self.locals["rewards"]
        
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.timesteps_log.append(self.num_timesteps)
                self.current_rewards[i] = 0
        
        total_episodes = len(self.episode_rewards)
        if total_episodes > 0 and total_episodes >= self.last_log_episode + self.log_freq:
            recent_rewards = self.episode_rewards[-self.log_freq:]
            print(f"[Episode {total_episodes}] Mean: {np.mean(recent_rewards):.3f} | "
                  f"Min: {np.min(recent_rewards):.3f} | Max: {np.max(recent_rewards):.3f}")
            self.last_log_episode = total_episodes
                
        return True
    
    def get_results(self):
        return {"rewards": self.episode_rewards, "timesteps": self.timesteps_log}


class ProgressCallback(BaseCallback):
    """Prints training progress periodically."""
    
    def __init__(self, total_timesteps, print_freq=10000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_print = 0
        
    def _on_step(self):
        if self.num_timesteps - self.last_print >= self.print_freq:
            progress = 100 * self.num_timesteps / self.total_timesteps
            print(f"Progress: {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%)")
            self.last_print = self.num_timesteps
        return True


def plot_learning_curve(rewards, timesteps, window=100, save_path=None):
    """Plot training rewards over episodes with moving average."""
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(rewards))
    
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(rewards)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window} ep)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episodic Reward')
    ax.set_title('ID-15: Learning Curve - AccidentEnv (LidarObservation)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_violin(rewards, save_path=None):
    """Plot violin plot showing reward distribution from test episodes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    parts = ax.violinplot([rewards], positions=[1], showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    median_reward = np.median(rewards)
    
    stats_text = f'Mean: {mean_reward:.3f}\nStd: {std_reward:.3f}\nMedian: {median_reward:.3f}'
    ax.text(1.3, mean_reward, stats_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Episodic Reward')
    ax.set_title(f'ID-16: Performance Test - AccidentEnv ({len(rewards)} episodes)')
    ax.set_xticks([1])
    ax.set_xticklabels(['PPO Agent'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return mean_reward, std_reward


def train():
    """Main training loop: trains PPO, saves model, generates plots."""
    print("=" * 50)
    print("Training PPO on AccidentEnv")
    print("=" * 50)
    
    # Set up parallel training environments
    print(f"\nCreating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Network architecture: 3 hidden layers
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.ReLU
    )
    
    # Learning rate decay from 3e-4 to 1e-5
    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining + 1e-5 * (1 - progress_remaining)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=256,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="auto"
    )
    
    print(f"\nHyperparameters: n_steps=4096, batch=256, epochs=15, gamma=0.995")
    print(f"Network: [256, 256, 128], LR: 3e-4 -> 1e-5")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}\n")
    
    reward_callback = RewardLoggingCallback(log_freq=LOG_EPISODE_FREQ)
    progress_callback = ProgressCallback(TOTAL_TIMESTEPS)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,
        save_path=MODEL_DIR,
        name_prefix="ppo_accident_env"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    
    callbacks = [reward_callback, progress_callback, checkpoint_callback, eval_callback]
    
    # Train the model
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )
    
    print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    final_model_path = os.path.join(MODEL_DIR, "ppo_accident_env_final")
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    results = reward_callback.get_results()
    np.save(os.path.join(SAVE_DIR, "training_rewards.npy"), results["rewards"])
    np.save(os.path.join(SAVE_DIR, "training_timesteps.npy"), results["timesteps"])
    
    print("\nGenerating learning curve...")
    plot_learning_curve(
        results["rewards"], 
        results["timesteps"],
        save_path=os.path.join(PLOT_DIR, "ID-15_learning_curve_opponent_custom_env.png")
    )
    
    env.close()
    eval_env.close()
    
    # Run test episodes for performance evaluation (ID-16)
    print(f"\nRunning {TEST_EPISODES} test episodes...")
    
    best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        test_model = PPO.load(best_model_path)
        print("Using best model")
    else:
        test_model = model
        print("Using final model")
    
    test_env = make_eval_env()
    test_rewards = []
    
    for ep in range(TEST_EPISODES):
        obs, _ = test_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = test_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        
        if (ep + 1) % 100 == 0:
            print(f"  {ep + 1}/{TEST_EPISODES} episodes | Mean: {np.mean(test_rewards):.3f}")
    
    test_env.close()
    np.save(os.path.join(SAVE_DIR, "test_rewards.npy"), test_rewards)
    
    print("\nGenerating violin plot...")
    mean_reward, std_reward = plot_violin(
        test_rewards,
        save_path=os.path.join(PLOT_DIR, "ID-16_performance_test_opponent_custom_env.png")
    )
    
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Training episodes: {len(results['rewards'])}")
    print(f"Test mean reward: {mean_reward:.3f} (+/- {std_reward:.3f})")


if __name__ == "__main__":
    train()

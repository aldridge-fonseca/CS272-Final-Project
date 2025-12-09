import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models_dir = "./models/"
model_path = os.path.join(models_dir, "intersection_lidar_ppo.zip")

model = PPO.load(model_path)

print("\nStarting performance evaluation (500 episodes)...")
eval_env = gym.make("intersection-v0", config={
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
    }
})

episode_rewards = []
num_episodes = 500

for episode in range(num_episodes):
    obs, info = eval_env.reset()
    episode_reward = 0
    done = truncated = False
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        episode_reward += reward
        
    episode_rewards.append(episode_reward)
    
    if (episode + 1) % 50 == 0:
        print(f"Evaluated {episode + 1}/{num_episodes} episodes")

eval_env.close()

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

parts = plt.violinplot(episode_rewards, showmeans=True, showmedians=True)

plt.title("Performance Test - 500 Episodes", fontsize=14, fontweight='bold')
plt.ylabel("Episodic Reward (Return)", fontsize=12)
plt.xticks([1], ['PPO with separate Actor-Critic trained for 100k timesteps'])

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
median_reward = np.median(episode_rewards)

plt.axhline(mean_reward, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Mean')
plt.axhline(mean_reward + std_reward, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
plt.axhline(mean_reward - std_reward, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

stats_text = f"Mean: {mean_reward:.2f}\nStd Dev: {std_reward:.2f}\nMedian: {median_reward:.2f}"
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

plt.legend(loc='upper right')

plt.grid(True, alpha=0.3, axis='y')
os.makedirs("./plots", exist_ok=True)
plot_path = "./plots/10_intersection_lidar_performance_test.png"
plt.savefig(plot_path, dpi=300)
print(f"Violin plot saved to {plot_path}")
print(f"\nPerformance Summary:")
print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Median Reward: {median_reward:.2f}")

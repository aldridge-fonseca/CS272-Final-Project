import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

log_path = "./logs/"
models_dir = "./models/"
model_path = os.path.join(models_dir, "intersection_lidar_ppo.zip")
monitor_log_dir = os.path.join(log_path, "monitor_logs/")

os.makedirs(monitor_log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs("./plots", exist_ok=True)

train_env = gym.make("intersection-v0", config={
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
    }
})

train_env = Monitor(train_env, monitor_log_dir)

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=linear_schedule(3e-4),
    n_steps=4096,
    batch_size=128,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]),
    verbose=1,
    tensorboard_log=log_path,
    ent_coef=0.1,
    device="auto"
)

print("Starting training (100,000 timesteps)...")
model.learn(total_timesteps=100000)

model.save(model_path)
print(f"Model saved to {model_path}")

monitor_file = os.path.join(monitor_log_dir, "monitor.csv")
if os.path.exists(monitor_file):
    df = pd.read_csv(monitor_file, skiprows=1)
    plt.figure(figsize=(10, 5))
    plt.plot(df['r'], alpha=0.3, label='Episode Reward')
    rolling_mean = df['r'].rolling(window=50).mean()
    plt.plot(rolling_mean, label='Rolling Mean (50 episodes)', color='orange')
    plt.title("Intersection (Lidar) Learning Curve")
    plt.xlabel("Training Episodes")
    plt.ylabel("Mean Episodic Reward (Return)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./plots/9_intersection_lidar_learning_curve.png")
    print("Learning curve saved to ./plots/9_intersection_lidar_learning_curve.png")

print("\nTraining complete! Run 10_Intersection_LidarObs.py to evaluate performance.")

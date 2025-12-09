
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from group5_custom_env import register_group5_env
import matplotlib.pyplot as plt
import seaborn as sns

register_group5_env()

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

os.makedirs('./plots', exist_ok=True)

print("="*70)
print("GENERATING PRESENTATION PLOTS")
print("="*70)

# ============================================================================
# ============================================================================

print("\n[1/3] Evaluating Random Policy (500 episodes)...")
env = gym.make('group5-env-v0')

random_successes = 0
random_collisions = 0
random_rewards = []

for episode in range(500):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    random_rewards.append(episode_reward)
    if env.unwrapped.vehicle.position[0] >= 625:
        random_successes += 1
    if env.unwrapped.vehicle.crashed:
        random_collisions += 1
    
    if (episode + 1) % 50 == 0:
        print(f"  Progress: {episode + 1}/500", end='\r')

print(f"  Progress: 500/500 ✓")
env.close()

random_success_rate = random_successes / 500 * 100
random_collision_rate = random_collisions / 500 * 100
random_avg_reward = np.mean(random_rewards)
random_std_reward = np.std(random_rewards)

print(f"  Success: {random_success_rate:.1f}%, Collision: {random_collision_rate:.1f}%")

# ============================================================================
# ============================================================================

print("\n[2/3] Evaluating PPO Baseline (500 episodes)...")

model1_path = './models/ppo_group5_env.zip'
if not os.path.exists(model1_path):
    print(f"  ❌ ERROR: Model not found: {model1_path}")
    print(f"  Please train the baseline model first.")
    print(f"  Exiting...")
    exit(1)

model1 = PPO.load(model1_path)
env1 = gym.make('group5-env-v0')
env1 = Monitor(env1)

baseline_successes = 0
baseline_collisions = 0
baseline_rewards = []
baseline_emergency_encounters = 0
baseline_emergency_yielded = 0

for episode in range(500):
    obs, info = env1.reset()
    done = False
    episode_reward = 0
    
    emergency_spawned = any(hasattr(v, 'is_emergency') and v.is_emergency 
                           for v in env1.unwrapped.road.vehicles)
    if emergency_spawned:
        baseline_emergency_encounters += 1
    
    yielded = False
    while not done:
        action, _ = model1.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env1.step(action)
        episode_reward += reward
        done = terminated or truncated
        
        if emergency_spawned and not yielded:
            ego = env1.unwrapped.vehicle
            my_lane = ego.lane_index[2] if len(ego.lane_index) > 2 else 0
            for v in env1.unwrapped.road.vehicles:
                if hasattr(v, 'is_emergency') and v.is_emergency:
                    v_lane = v.lane_index[2] if len(v.lane_index) > 2 else 0
                    if v_lane == my_lane and v.position[0] > ego.position[0]:
                        yielded = True
                        break
    
    if yielded:
        baseline_emergency_yielded += 1
    
    baseline_rewards.append(episode_reward)
    if env1.unwrapped.vehicle.position[0] >= 625:
        baseline_successes += 1
    if env1.unwrapped.vehicle.crashed:
        baseline_collisions += 1
    
    if (episode + 1) % 50 == 0:
        print(f"  Progress: {episode + 1}/500", end='\r')

print(f"  Progress: 500/500 ✓")
env1.close()

baseline_success_rate = baseline_successes / 500 * 100
baseline_collision_rate = baseline_collisions / 500 * 100
baseline_avg_reward = np.mean(baseline_rewards)
baseline_std_reward = np.std(baseline_rewards)
baseline_emergency_yield = (baseline_emergency_yielded / baseline_emergency_encounters * 100) if baseline_emergency_encounters > 0 else 0

print(f"  Success: {baseline_success_rate:.1f}%, Collision: {baseline_collision_rate:.1f}%")

# ============================================================================
# ============================================================================

print("\n[3/3] Evaluating PPO + Self-Attention (500 episodes)...")

model2_path = './models/ppo_attention_group5_env.zip'
if not os.path.exists(model2_path):
    print(f"  ❌ ERROR: Model not found: {model2_path}")
    print(f"  Please train the self-attention model first.")
    print(f"  Exiting...")
    exit(1)

    model2 = PPO.load(model2_path)
    env2 = gym.make('group5-env-v0')
    env2 = Monitor(env2)
    
    attention_successes = 0
    attention_collisions = 0
    attention_rewards = []
    attention_emergency_encounters = 0
    attention_emergency_yielded = 0
    
    for episode in range(500):
        obs, info = env2.reset()
        done = False
        episode_reward = 0
        
        emergency_spawned = any(hasattr(v, 'is_emergency') and v.is_emergency 
                               for v in env2.unwrapped.road.vehicles)
        if emergency_spawned:
            attention_emergency_encounters += 1
        
        yielded = False
        while not done:
            action, _ = model2.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env2.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if emergency_spawned and not yielded:
                ego = env2.unwrapped.vehicle
                my_lane = ego.lane_index[2] if len(ego.lane_index) > 2 else 0
                for v in env2.unwrapped.road.vehicles:
                    if hasattr(v, 'is_emergency') and v.is_emergency:
                        ev_lane = v.lane_index[2] if len(v.lane_index) > 2 else 0
                        if my_lane != ev_lane:
                            yielded = True
                            break
        
        if yielded:
            attention_emergency_yielded += 1
        
        attention_rewards.append(episode_reward)
        if env2.unwrapped.vehicle.position[0] >= 625:
            attention_successes += 1
        if env2.unwrapped.vehicle.crashed:
            attention_collisions += 1
        
        if (episode + 1) % 50 == 0:
            print(f"  Progress: {episode + 1}/500", end='\r')
    
    print(f"  Progress: 500/500 ✓")
    env2.close()
    
    attention_success_rate = attention_successes / 500 * 100
    attention_collision_rate = attention_collisions / 500 * 100
    attention_avg_reward = np.mean(attention_rewards)
    attention_std_reward = np.std(attention_rewards)
    attention_emergency_yield = (attention_emergency_yielded / attention_emergency_encounters * 100) if attention_emergency_encounters > 0 else 0

print(f"  Success: {attention_success_rate:.1f}%, Collision: {attention_collision_rate:.1f}%")

# ============================================================================
# ============================================================================

print(f"\n{'='*70}")
print("GENERATING PLOTS")
print(f"{'='*70}")

models = ['Random\nPolicy', 'PPO\nBaseline', 'PPO +\nSelf-Attention']
colors = ['#95a5a6', '#3498db', '#e74c3c']

success_rates = [random_success_rate, baseline_success_rate, attention_success_rate]
collision_rates = [random_collision_rate, baseline_collision_rate, attention_collision_rate]
avg_rewards = [random_avg_reward, baseline_avg_reward, attention_avg_reward]
std_rewards = [random_std_reward, baseline_std_reward, attention_std_reward]
emergency_yielding = [0.0, baseline_emergency_yield, attention_emergency_yield]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

bars1 = ax1.bar(models, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars1, success_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + max(success_rates)*0.02,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(success_rates) * 1.25])
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(models, collision_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars2, collision_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(collision_rates)*0.02,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Collision Rate', fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(collision_rates) * 1.15])
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./plots/final_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/final_comparison.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Success\nRate', 'Collision\nReduction', 'Emergency\nYielding']

success_improvement = ((attention_success_rate - baseline_success_rate) / baseline_success_rate * 100) if baseline_success_rate > 0 else 0
collision_improvement = ((baseline_collision_rate - attention_collision_rate) / baseline_collision_rate * 100)
yielding_improvement = ((attention_emergency_yield - baseline_emergency_yield) / baseline_emergency_yield * 100) if baseline_emergency_yield > 0 else 0

improvements = [success_improvement, collision_improvement, yielding_improvement]
bar_colors = ['green' if x > 0 else 'red' for x in improvements]

bars = ax.bar(metrics, improvements, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, improvements):
    height = bar.get_height()
    y_offset = max(abs(max(improvements)), abs(min(improvements))) * 0.05
    y_pos = height + y_offset if height >= 0 else height - y_offset
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('Self-Attention Improvement Over Baseline', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/final_improvements.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/final_improvements.png")
plt.close()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\nRandom Policy:")
print(f"  Success: {random_success_rate:.1f}%, Collision: {random_collision_rate:.1f}%")
print(f"\nPPO Baseline:")
print(f"  Success: {baseline_success_rate:.1f}%, Collision: {baseline_collision_rate:.1f}%")
print(f"\nPPO + Self-Attention:")
print(f"  Success: {attention_success_rate:.1f}%, Collision: {attention_collision_rate:.1f}%")
print(f"\nImprovements:")
print(f"  Success: {success_improvement:+.1f}%")
print(f"  Collision Reduction: {collision_improvement:+.1f}%")
print(f"  Emergency Yielding: {yielding_improvement:+.1f}%")
print(f"\n{'='*70}")
print("✅ COMPLETE! Use these plots for your presentation.")
print(f"{'='*70}")

"""
Comprehensive Model Evaluation Script

Evaluates PPO Baseline and PPO with Self-Attention on all key metrics:
- Success rate (reaching 625m)
- Collision rate
- Off-road rate
- Average reward
- Episode length
- Emergency vehicle encounters
- Emergency vehicle yielding rate

Usage:
    python evaluate_models.py
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from group5_custom_env import register_group5_env
import matplotlib.pyplot as plt
import seaborn as sns

# Register environment
register_group5_env()

# Set style
sns.set_style("whitegrid")


def evaluate_model(model, env, num_episodes=500, model_name="Model"):
    """
    Comprehensive evaluation of a trained model.
    
    Returns:
        dict with all metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    print(f"Running {num_episodes} episodes (deterministic)...")
    
    # Metrics to track
    episode_rewards = []
    episode_lengths = []
    
    successes = 0
    collisions = 0
    off_road = 0
    timeouts = 0
    
    emergency_encounters = 0
    emergency_yielded = 0
    emergency_blocked = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        # Check if emergency vehicle spawned
        emergency_spawned = any(
            hasattr(v, 'is_emergency') and v.is_emergency 
            for v in env.unwrapped.road.vehicles
        )
        if emergency_spawned:
            emergency_encounters += 1
        
        # Track if we blocked emergency vehicle during episode
        blocked_emergency = False
        yielded_properly = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Check emergency vehicle interaction
            if emergency_spawned:
                ego = env.unwrapped.vehicle
                my_lane = ego.lane_index[2] if len(ego.lane_index) > 2 else 0
                my_position = ego.position[0]
                
                for vehicle in env.unwrapped.road.vehicles:
                    if not hasattr(vehicle, 'is_emergency') or not vehicle.is_emergency:
                        continue
                    
                    ev_lane = vehicle.lane_index[2] if len(vehicle.lane_index) > 2 else 0
                    ev_position = vehicle.position[0]
                    distance_behind = my_position - ev_position
                    
                    # Emergency vehicle behind us within 60m
                    if 0 < distance_behind < 60:
                        if ev_lane == my_lane:
                            blocked_emergency = True
                        else:
                            yielded_properly = True
        
        # Determine episode outcome
        ego = env.unwrapped.vehicle
        if ego.crashed:
            collisions += 1
        elif not ego.on_road:
            off_road += 1
        elif ego.position[0] >= env.unwrapped.config["road_length"]:
            successes += 1
        else:
            timeouts += 1
        
        # Emergency vehicle handling
        if emergency_spawned:
            if blocked_emergency:
                emergency_blocked += 1
            elif yielded_properly:
                emergency_yielded += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Progress indicator
        if (episode + 1) % 50 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes", end='\r')
    
    print(f"  Progress: {num_episodes}/{num_episodes} episodes ✓")
    
    # Calculate statistics
    results = {
        'model_name': model_name,
        'num_episodes': num_episodes,
        
        # Episode outcomes
        'successes': successes,
        'collisions': collisions,
        'off_road': off_road,
        'timeouts': timeouts,
        
        # Rates
        'success_rate': successes / num_episodes * 100,
        'collision_rate': collisions / num_episodes * 100,
        'off_road_rate': off_road / num_episodes * 100,
        'timeout_rate': timeouts / num_episodes * 100,
        
        # Rewards
        'episode_rewards': episode_rewards,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        
        # Episode lengths
        'episode_lengths': episode_lengths,
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        
        # Emergency vehicle handling
        'emergency_encounters': emergency_encounters,
        'emergency_yielded': emergency_yielded,
        'emergency_blocked': emergency_blocked,
        'emergency_yield_rate': (emergency_yielded / emergency_encounters * 100) if emergency_encounters > 0 else 0,
        'emergency_block_rate': (emergency_blocked / emergency_encounters * 100) if emergency_encounters > 0 else 0,
    }
    
    return results


def print_results(results):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results['model_name']}")
    print(f"{'='*70}")
    
    print(f"\n📊 Episode Outcomes ({results['num_episodes']} episodes):")
    print(f"  Successes:  {results['successes']:3d} ({results['success_rate']:5.1f}%)")
    print(f"  Collisions: {results['collisions']:3d} ({results['collision_rate']:5.1f}%)")
    print(f"  Off-road:   {results['off_road']:3d} ({results['off_road_rate']:5.1f}%)")
    print(f"  Timeouts:   {results['timeouts']:3d} ({results['timeout_rate']:5.1f}%)")
    
    print(f"\n💰 Rewards:")
    print(f"  Mean:   {results['mean_reward']:7.2f} ± {results['std_reward']:6.2f}")
    print(f"  Median: {results['median_reward']:7.2f}")
    print(f"  Range:  [{results['min_reward']:6.2f}, {results['max_reward']:6.2f}]")
    
    print(f"\n⏱️  Episode Length:")
    print(f"  Mean:   {results['mean_length']:6.1f} ± {results['std_length']:5.1f} steps")
    
    print(f"\n🚨 Emergency Vehicle Handling:")
    print(f"  Encounters: {results['emergency_encounters']:3d} episodes")
    print(f"  Yielded:    {results['emergency_yielded']:3d} ({results['emergency_yield_rate']:5.1f}%)")
    print(f"  Blocked:    {results['emergency_blocked']:3d} ({results['emergency_block_rate']:5.1f}%)")


def compare_results(results1, results2):
    """Compare two models and print differences."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {results1['model_name']} vs {results2['model_name']}")
    print(f"{'='*70}")
    
    print(f"\n📊 Success Rate:")
    print(f"  {results1['model_name']:20s}: {results1['success_rate']:5.1f}%")
    print(f"  {results2['model_name']:20s}: {results2['success_rate']:5.1f}%")
    diff = results2['success_rate'] - results1['success_rate']
    print(f"  Difference: {diff:+5.1f}% {'✅' if diff > 0 else '❌'}")
    
    print(f"\n💥 Collision Rate:")
    print(f"  {results1['model_name']:20s}: {results1['collision_rate']:5.1f}%")
    print(f"  {results2['model_name']:20s}: {results2['collision_rate']:5.1f}%")
    diff = results1['collision_rate'] - results2['collision_rate']  # Lower is better
    print(f"  Reduction: {diff:+5.1f}% {'✅' if diff > 0 else '❌'}")
    
    print(f"\n💰 Average Reward:")
    print(f"  {results1['model_name']:20s}: {results1['mean_reward']:7.2f}")
    print(f"  {results2['model_name']:20s}: {results2['mean_reward']:7.2f}")
    diff = results2['mean_reward'] - results1['mean_reward']
    print(f"  Improvement: {diff:+7.2f} {'✅' if diff > 0 else '❌'}")
    
    print(f"\n🚨 Emergency Vehicle Yielding:")
    print(f"  {results1['model_name']:20s}: {results1['emergency_yield_rate']:5.1f}%")
    print(f"  {results2['model_name']:20s}: {results2['emergency_yield_rate']:5.1f}%")
    diff = results2['emergency_yield_rate'] - results1['emergency_yield_rate']
    print(f"  Improvement: {diff:+5.1f}% {'✅' if diff > 0 else '❌'}")
    
    # Overall winner
    print(f"\n{'='*70}")
    score1 = 0
    score2 = 0
    
    if results1['success_rate'] > results2['success_rate']:
        score1 += 1
    else:
        score2 += 1
    
    if results1['collision_rate'] < results2['collision_rate']:
        score1 += 1
    else:
        score2 += 1
    
    if results1['mean_reward'] > results2['mean_reward']:
        score1 += 1
    else:
        score2 += 1
    
    if results1['emergency_yield_rate'] > results2['emergency_yield_rate']:
        score1 += 1
    else:
        score2 += 1
    
    print(f"Overall Score: {results1['model_name']} {score1} - {score2} {results2['model_name']}")
    if score1 > score2:
        print(f"🏆 Winner: {results1['model_name']}")
    elif score2 > score1:
        print(f"🏆 Winner: {results2['model_name']}")
    else:
        print(f"🤝 Tie")
    print(f"{'='*70}")


def plot_comparison(results1, results2, save_dir='./plots'):
    """Create comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Success and Collision Rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [results1['model_name'], results2['model_name']]
    success_rates = [results1['success_rate'], results2['success_rate']]
    collision_rates = [results1['collision_rate'], results2['collision_rate']]
    
    ax1.bar(models, success_rates, color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11)
    
    ax2.bar(models, collision_rates, color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Collision Rate (%)', fontsize=12)
    ax2.set_title('Collision Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(collision_rates) * 1.2])
    for i, v in enumerate(collision_rates):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_rates.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_dir}/comparison_rates.png")
    plt.close()
    
    # 2. Reward Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [results1['episode_rewards'], results2['episode_rewards']]
    positions = [0, 1]
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.7)
    colors = ['#3498db', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=13)
    ax.set_title('Reward Distribution Comparison', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_rewards.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/comparison_rewards.png")
    plt.close()
    
    # 3. Emergency Vehicle Handling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    yield_rates = [results1['emergency_yield_rate'], results2['emergency_yield_rate']]
    
    ax.bar(models, yield_rates, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Emergency Vehicle Yielding Rate (%)', fontsize=12)
    ax.set_title('Emergency Vehicle Handling', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for i, v in enumerate(yield_rates):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_emergency.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/comparison_emergency.png")
    plt.close()


def save_results_to_file(results1, results2, filename='./evaluation_results.txt'):
    """Save results to text file."""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for results in [results1, results2]:
            f.write(f"\n{results['model_name']}\n")
            f.write("-"*70 + "\n")
            f.write(f"Success Rate:     {results['success_rate']:.1f}%\n")
            f.write(f"Collision Rate:   {results['collision_rate']:.1f}%\n")
            f.write(f"Mean Reward:      {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Emergency Yield:  {results['emergency_yield_rate']:.1f}%\n")
            f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("COMPARISON\n")
        f.write("="*70 + "\n")
        f.write(f"Success Rate Diff:    {results2['success_rate'] - results1['success_rate']:+.1f}%\n")
        f.write(f"Collision Rate Diff:  {results1['collision_rate'] - results2['collision_rate']:+.1f}%\n")
        f.write(f"Reward Diff:          {results2['mean_reward'] - results1['mean_reward']:+.2f}\n")
        f.write(f"Emergency Yield Diff: {results2['emergency_yield_rate'] - results1['emergency_yield_rate']:+.1f}%\n")
    
    print(f"\n✅ Saved: {filename}")


def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Model paths
    model1_path = './models/ppo_group5_env.zip'
    model2_path = './models/ppo_attention_group5_env.zip'
    
    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"\n❌ Error: Model not found at {model1_path}")
        print("   Please train the baseline model first:")
        print("   python train_ppo.py")
        return
    
    if not os.path.exists(model2_path):
        print(f"\n❌ Error: Model not found at {model2_path}")
        print("   Please train the attention model first:")
        print("   python train_ppo_attention.py")
        return
    
    # Load models
    print("\nLoading models...")
    model1 = PPO.load(model1_path)
    print(f"  ✅ Loaded: {model1_path}")
    
    model2 = PPO.load(model2_path)
    print(f"  ✅ Loaded: {model2_path}")
    
    # Create environments
    print("\nCreating evaluation environments...")
    env1 = gym.make('group5-env-v0')
    env1 = Monitor(env1)
    
    env2 = gym.make('group5-env-v0')
    env2 = Monitor(env2)
    
    # Evaluate models
    results1 = evaluate_model(model1, env1, num_episodes=500, model_name="PPO Baseline")
    results2 = evaluate_model(model2, env2, num_episodes=500, model_name="PPO + Self-Attention")
    
    # Print results
    print_results(results1)
    print_results(results2)
    
    # Compare
    compare_results(results1, results2)
    
    # Create plots
    print(f"\n{'='*70}")
    print("Generating comparison plots...")
    print(f"{'='*70}")
    plot_comparison(results1, results2)
    
    # Save to file
    save_results_to_file(results1, results2)
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - ./plots/comparison_rates.png")
    print("  - ./plots/comparison_rewards.png")
    print("  - ./plots/comparison_emergency.png")
    print("  - ./evaluation_results.txt")
    print(f"{'='*70}")
    
    # Cleanup
    env1.close()
    env2.close()


if __name__ == '__main__':
    main()

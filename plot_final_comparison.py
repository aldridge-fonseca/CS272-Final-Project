"""
Create publication-ready comparison plots for all three approaches:
- Random Policy (baseline)
- PPO Baseline
- PPO + Self-Attention

Generates clean, presentation-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# Create output directory
os.makedirs('./plots', exist_ok=True)

# ============================================================================
# DATA - Replace with your actual results
# ============================================================================

# Model names
models = ['Random\nPolicy', 'PPO\nBaseline', 'PPO +\nSelf-Attention']
colors = ['#95a5a6', '#3498db', '#e74c3c']  # Gray, Blue, Red

# Success rates (%)
success_rates = [0.0, 10.6, 16.0]  # Replace with actual values

# Collision rates (%)
collision_rates = [85.0, 53.2, 41.8]  # Replace with actual values

# Average rewards
avg_rewards = [-50.0, 42.5, 45.3]  # Replace with actual values
std_rewards = [10.0, 12.3, 11.8]   # Replace with actual values

# Emergency vehicle yielding (%) - only for trained agents
emergency_yielding = [0.0, 60.1, 72.3]  # Replace with actual values

# ============================================================================
# PLOT 1: Success Rate Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, success_rates)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Success Rate: Reaching 625m Goal', fontsize=15, fontweight='bold')
ax.set_ylim([0, max(success_rates) * 1.3])
ax.grid(axis='y', alpha=0.3)

# Add improvement annotations
if success_rates[2] > success_rates[1]:
    improvement = ((success_rates[2] - success_rates[1]) / success_rates[1] * 100)
    ax.annotate(f'+{improvement:.0f}%',
                xy=(2, success_rates[2]), xytext=(1.5, success_rates[2] + 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('./plots/comparison_success_rate.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_success_rate.png")
plt.close()

# ============================================================================
# PLOT 2: Collision Rate Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, collision_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, collision_rates)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Collision Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Collision Rate: Safety Performance', fontsize=15, fontweight='bold')
ax.set_ylim([0, max(collision_rates) * 1.2])
ax.grid(axis='y', alpha=0.3)

# Add improvement annotations
if collision_rates[2] < collision_rates[1]:
    reduction = ((collision_rates[1] - collision_rates[2]) / collision_rates[1] * 100)
    ax.annotate(f'-{reduction:.0f}%',
                xy=(2, collision_rates[2]), xytext=(1.5, collision_rates[2] + 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('./plots/comparison_collision_rate.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_collision_rate.png")
plt.close()

# ============================================================================
# PLOT 3: Combined Success & Collision (Side by Side)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Success rates
bars1 = ax1.bar(models, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars1, success_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(success_rates) * 1.3])
ax1.grid(axis='y', alpha=0.3)

# Collision rates
bars2 = ax2.bar(models, collision_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars2, collision_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Collision Rate', fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(collision_rates) * 1.2])
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Performance Comparison: Success vs Safety', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./plots/comparison_success_collision.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_success_collision.png")
plt.close()

# ============================================================================
# PLOT 4: Average Reward Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, avg_rewards, yerr=std_rewards, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, capsize=5, error_kw={'linewidth': 2})

# Add value labels on bars
for bar, val, std in zip(bars, avg_rewards, std_rewards):
    height = bar.get_height()
    y_pos = height + std + 2 if height > 0 else height - std - 2
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:.1f}',
            ha='center', va='bottom' if height > 0 else 'top', 
            fontsize=12, fontweight='bold')

ax.set_ylabel('Average Episode Reward', fontsize=13, fontweight='bold')
ax.set_title('Average Reward Comparison', fontsize=15, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/comparison_rewards.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_rewards.png")
plt.close()

# ============================================================================
# PLOT 5: Emergency Vehicle Yielding (Trained Agents Only)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Only show trained agents
trained_models = ['PPO\nBaseline', 'PPO +\nSelf-Attention']
trained_yielding = [emergency_yielding[1], emergency_yielding[2]]
trained_colors = [colors[1], colors[2]]

bars = ax.bar(trained_models, trained_yielding, color=trained_colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, trained_yielding):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Emergency Vehicle Yielding Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Emergency Vehicle Handling', fontsize=15, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Add improvement annotation
if trained_yielding[1] > trained_yielding[0]:
    improvement = trained_yielding[1] - trained_yielding[0]
    ax.annotate(f'+{improvement:.1f}%',
                xy=(1, trained_yielding[1]), xytext=(0.5, trained_yielding[1] + 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('./plots/comparison_emergency_yielding.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_emergency_yielding.png")
plt.close()

# ============================================================================
# PLOT 6: Comprehensive Summary (All Metrics)
# ============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Success Rate
bars = ax1.bar(models, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, success_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Success Rate', fontsize=12, fontweight='bold')
ax1.set_ylim([0, max(success_rates) * 1.3])
ax1.grid(axis='y', alpha=0.3)

# 2. Collision Rate
bars = ax2.bar(models, collision_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, collision_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Collision Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Collision Rate', fontsize=12, fontweight='bold')
ax2.set_ylim([0, max(collision_rates) * 1.2])
ax2.grid(axis='y', alpha=0.3)

# 3. Average Reward
bars = ax3.bar(models, avg_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, avg_rewards):
    height = bar.get_height()
    y_pos = height + 2 if height > 0 else height - 2
    ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=10, fontweight='bold')
ax3.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
ax3.set_title('Average Reward', fontsize=12, fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.grid(axis='y', alpha=0.3)

# 4. Emergency Yielding (trained only)
bars = ax4.bar(trained_models, trained_yielding, color=trained_colors, alpha=0.8, 
               edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, trained_yielding):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_ylabel('Emergency Yielding (%)', fontsize=11, fontweight='bold')
ax4.set_title('Emergency Vehicle Handling', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 100])
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('./plots/comparison_all_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_all_metrics.png")
plt.close()

# ============================================================================
# PLOT 7: Improvement Over Baseline (Percentage)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Success\nRate', 'Collision\nReduction', 'Avg\nReward', 'Emergency\nYielding']

# Calculate improvements (Self-Attention vs Baseline)
success_improvement = ((success_rates[2] - success_rates[1]) / success_rates[1] * 100) if success_rates[1] > 0 else 0
collision_improvement = ((collision_rates[1] - collision_rates[2]) / collision_rates[1] * 100)  # Reduction is positive
reward_improvement = ((avg_rewards[2] - avg_rewards[1]) / abs(avg_rewards[1]) * 100) if avg_rewards[1] != 0 else 0
yielding_improvement = ((emergency_yielding[2] - emergency_yielding[1]) / emergency_yielding[1] * 100) if emergency_yielding[1] > 0 else 0

improvements = [success_improvement, collision_improvement, reward_improvement, yielding_improvement]
bar_colors = ['green' if x > 0 else 'red' for x in improvements]

bars = ax.bar(metrics, improvements, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    y_pos = height + 2 if height > 0 else height - 2
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('Self-Attention Improvement Over Baseline', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/comparison_improvements.png', dpi=300, bbox_inches='tight')
print("✅ Saved: ./plots/comparison_improvements.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*70}")
print("PLOT GENERATION COMPLETE")
print(f"{'='*70}")
print("\nGenerated plots:")
print("  1. comparison_success_rate.png       - Success rate comparison")
print("  2. comparison_collision_rate.png     - Collision rate comparison")
print("  3. comparison_success_collision.png  - Side-by-side success & collision")
print("  4. comparison_rewards.png            - Average reward comparison")
print("  5. comparison_emergency_yielding.png - Emergency vehicle handling")
print("  6. comparison_all_metrics.png        - All metrics in one figure")
print("  7. comparison_improvements.png       - Improvement percentages")
print(f"\n{'='*70}")
print("For presentation, use:")
print("  - comparison_success_collision.png (shows both key metrics)")
print("  - comparison_improvements.png (shows your contribution)")
print(f"{'='*70}")

"""
DRL Agent Training Script for Highway Merge Operations
Trains agents on group5-env and custom environment with LidarObservation and GrayscaleObservation
"""

import os
import sys
import numpy as np
import matplotlib
# Set non-interactive backend for saving plots
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (non-interactive)
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import is_image_space
import gymnasium as gym
import csv

# Add group5-env to path
group5_env_path = os.path.join(os.path.dirname(__file__), 'group5-env')
sys.path.insert(0, group5_env_path)

# Import and register group5 environment
try:
    from group5_custom_env import register_group5_env
    register_group5_env()
    print("✓ Group5 environment registered successfully")
except ImportError as e:
    print(f"Warning: Could not import group5 environment: {e}")
    print("Make sure the group5-env folder is in the correct path.")

# Add custom environment to path
custom_env_path = os.path.join(os.path.dirname(__file__), 'cs272-team-6-custom-env-master', 'custom')
sys.path.insert(0, custom_env_path)

# Import and register custom environment
try:
    from custom_env import AccidentEnv
    # Register custom environment as custom-highway-v0
    try:
        # Check if already registered
        gym.envs.registry['custom-highway-v0']
    except KeyError:
        # Not registered, register it
        gym.register(
            id='custom-highway-v0',
            entry_point='custom_env:AccidentEnv',
        )
except ImportError as e:
    print(f"Warning: Could not import custom environment: {e}")
    print("Make sure the custom environment is in the correct path.")
    AccidentEnv = None

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class EpisodeRewardCallback(BaseCallback):
    """Callback to track episodic rewards during training"""
    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward = 0.0
        self.episode_length = 0
        
    def _on_step(self) -> bool:
        # Track rewards from the environment
        rewards = self.locals.get('rewards', [])
        if len(rewards) > 0:
            self.episode_reward += rewards[0]
            self.episode_length += 1
        
        # Check if episode is done
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        # Monitor wrapper adds episode info to the info dict when episode ends
        if len(infos) > 0 and isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if 'episode' in info:
                episode_info = info['episode']
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                # Reset for next episode
                self.episode_reward = 0.0
                self.episode_length = 0
        
        # Also reset if episode is done (even if Monitor didn't provide info)
        if len(dones) > 0 and dones[0]:
            self.episode_reward = 0.0
            self.episode_length = 0
            
        return True


def create_env(env_name, obs_type, render_mode=None):
    """Create and configure environment with specified observation type for merge operations"""
    if env_name == 'group5-env-v0':
        # Configure for merge operations: focus on lane changes and merging behavior
        # Configure observation based on type
        if obs_type == "GrayscaleObservation":
            observation_config = {
                "type": obs_type,
                "observation_shape": (128, 64),  # (width, height)
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140]  # RGB to grayscale weights
            }
        else:
            observation_config = {"type": obs_type}
        
        config = {
            "observation": observation_config,
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 5,
            "vehicles_count": 10,
            "duration": 50,
            "initial_spacing": 2,
            "collision_reward": -1.5,
            "high_speed_reward": 0.3,
            "progress_reward": 0.5,  # Increased from 0.4 to encourage forward progress
            "success_reward": 1.0,
            "lane_change_reward": 0.15,  # Changed from -0.01 to +0.15 to ENCOURAGE lane changes
            "yielding_reward": 0.5,
            "reward_speed_range": [20, 28],
            "normalize_reward": False,
        }
        # Try passing config to gym.make, fallback to unwrapped.configure if needed
        try:
            env = gym.make('group5-env-v0', render_mode=render_mode, config=config)
        except TypeError:
            # If config parameter not supported, use unwrapped.configure
            env = gym.make('group5-env-v0', render_mode=render_mode)
            env.unwrapped.configure(config)
    elif env_name == 'custom-highway-v0':
        # Configure custom environment for merge operations
        # Configure observation based on type
        if obs_type == "GrayscaleObservation":
            observation_config = {
                "type": obs_type,
                "observation_shape": (128, 64),  # (width, height)
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140]  # RGB to grayscale weights
            }
        else:
            observation_config = {"type": obs_type}
        
        config = {
            "observation": observation_config,
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 15,
            "duration": 40,  # Extended duration for merge operations
            "initial_spacing": 2,
            "lane_change_reward": 0.15,  # ENCOURAGE lane changes (was 0, neutral)
            "right_lane_reward": 0.15,  # Increased from 0.1 to encourage merging to right lane
            "high_speed_reward": 0.4,  # Keep speed reward
            "collision_reward": -1.5,  # Strong penalty for crashes
        }
        # Try passing config to gym.make, fallback to unwrapped.configure if needed
        try:
            env = gym.make('custom-highway-v0', render_mode=render_mode, config=config)
        except TypeError:
            # If config parameter not supported, use unwrapped.configure
            env = gym.make('custom-highway-v0', render_mode=render_mode)
            env.unwrapped.configure(config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env


def train_agent(env_name, obs_type, total_timesteps=100000, prefix=""):
    """Train a PPO agent on the specified environment and observation type"""
    print(f"\n{'='*60}")
    print(f"Training {prefix} with {obs_type}")
    print(f"{'='*60}")
    
    # Create environment
    env = create_env(env_name, obs_type)
    log_dir = f"./logs/{prefix}_{obs_type}_train/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, allow_early_resets=True)
    
    # Determine policy type based on actual observation space
    # Check if observation space is an image space (not just by name)
    obs_space = env.observation_space
    try:
        # Check if it's actually an image space
        is_image = is_image_space(obs_space, check_channels=False)
        if is_image:
            policy = "CnnPolicy"
            print(f"Using CnnPolicy for image observation space: {obs_space}")
        else:
            policy = "MlpPolicy"
            print(f"Using MlpPolicy for vector observation space: {obs_space}")
    except Exception as e:
        # Fallback: check observation space shape/dtype
        # Images typically have: (H, W) or (H, W, C) with H, W >= 32
        # Vector observations have: (n,) or (n, m) with small dimensions
        if hasattr(obs_space, 'shape'):
            shape = obs_space.shape
            if len(shape) == 3:
                # 3D: (H, W, C) - likely an image
                policy = "CnnPolicy"
                print(f"Using CnnPolicy (fallback 3D) for observation space: {obs_space}")
            elif len(shape) == 2 and shape[0] >= 32 and shape[1] >= 32:
                # 2D with large dimensions - likely an image (H, W)
                policy = "CnnPolicy"
                print(f"Using CnnPolicy (fallback 2D image) for observation space: {obs_space}")
            else:
                # 1D or 2D with small dimensions - vector observation
                policy = "MlpPolicy"
                print(f"Using MlpPolicy (fallback) for observation space: {obs_space}")
        else:
            # Default to MLP for unknown spaces
            policy = "MlpPolicy"
            print(f"Using MlpPolicy (default) for observation space: {obs_space}")
    
    # Create model
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Increased from 0.01 to encourage more exploration during training
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{prefix}_{obs_type}/"
    )
    
    # Setup callback for tracking episodic rewards
    episode_callback = EpisodeRewardCallback()
    
    # Create eval environment
    eval_env = create_env(env_name, obs_type)
    eval_log_dir = f"./logs/{prefix}_{obs_type}_eval/"
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_env = Monitor(eval_env, eval_log_dir, allow_early_resets=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{prefix}_{obs_type}/",
        log_path=eval_log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    # Check if progress bar dependencies are available
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Note: tqdm and rich not available, progress bar disabled")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[episode_callback, eval_callback],
        progress_bar=use_progress_bar
    )
    
    # Save model
    model_path = f"./models/{prefix}_{obs_type}_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Get learning curve data from Monitor logs
    learning_curve = []
    monitor_file = os.path.join(log_dir, "monitor.csv")
    
    if os.path.exists(monitor_file):
        try:
            with open(monitor_file, 'r') as f:
                lines = f.readlines()
                # Find the header line (first non-comment line)
                header_line_idx = None
                for i, line in enumerate(lines):
                    if not line.strip().startswith('#') and line.strip():
                        header_line_idx = i
                        break
                
                if header_line_idx is not None:
                    # Read CSV starting from header
                    reader = csv.DictReader(lines[header_line_idx:])
                    for row in reader:
                        if 'r' in row and row['r']:
                            try:
                                learning_curve.append(float(row['r']))
                            except (ValueError, KeyError):
                                pass
        except Exception as e:
            print(f"Warning: Could not read from Monitor logs: {e}")
    
    # If Monitor didn't capture enough, use callback data
    if len(learning_curve) < 10 and len(episode_callback.episode_rewards) > 0:
        print(f"Using callback episode rewards for learning curve ({len(episode_callback.episode_rewards)} episodes)")
        learning_curve = episode_callback.episode_rewards
    
    # If still empty, create placeholder
    if len(learning_curve) == 0:
        print("Warning: No episode rewards captured. Creating placeholder learning curve.")
        learning_curve = [0.0] * 100
    
    print(f"Learning curve prepared with {len(learning_curve)} data points")
    
    env.close()
    eval_env.close()
    
    return model, learning_curve


def evaluate_agent(model, env_name, obs_type, n_episodes=500, prefix=""):
    """Evaluate trained agent over n_episodes without exploration"""
    print(f"\nEvaluating {prefix} with {obs_type} over {n_episodes} episodes...")
    
    env = create_env(env_name, obs_type)
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            print(f"  Completed {episode + 1}/{n_episodes} episodes")
    
    env.close()
    
    return episode_rewards


def plot_learning_curve(episode_rewards, prefix, obs_type, save_path):
    """Plot learning curve: Mean episodic reward vs training episodes"""
    try:
        print(f"  Attempting to plot learning curve: {len(episode_rewards)} data points")
        if len(episode_rewards) == 0:
            print(f"  ✗ Warning: No episode rewards to plot for {prefix} - {obs_type}")
            return
        
        # Convert to numpy array if needed
        episode_rewards = np.array(episode_rewards)
        print(f"  Data range: min={episode_rewards.min():.2f}, max={episode_rewards.max():.2f}, mean={episode_rewards.mean():.2f}")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Calculate rolling mean
        window_size = min(100, len(episode_rewards) // 10)
        if window_size < 1:
            window_size = 1
        
        # Calculate rolling mean ensuring it matches the length of episode_rewards
        if len(episode_rewards) > 0:
            # Use pandas rolling if available, otherwise use numpy with proper alignment
            try:
                import pandas as pd
                df = pd.Series(episode_rewards)
                rolling_mean = df.rolling(window=window_size, min_periods=1, center=True).mean().values
            except ImportError:
                # Fallback to numpy convolution with proper padding
                # Pad symmetrically to maintain same length
                pad_width = window_size // 2
                padded_rewards = np.pad(episode_rewards, pad_width, mode='edge')
                rolling_mean = np.convolve(padded_rewards, np.ones(window_size)/window_size, mode='valid')
                # Ensure length matches (trim if necessary)
                if len(rolling_mean) > len(episode_rewards):
                    # Trim from the end
                    rolling_mean = rolling_mean[:len(episode_rewards)]
                elif len(rolling_mean) < len(episode_rewards):
                    # Pad from the end
                    rolling_mean = np.pad(rolling_mean, (0, len(episode_rewards) - len(rolling_mean)), mode='edge')
        else:
            rolling_mean = episode_rewards
        
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
        if len(rolling_mean) > 0 and len(rolling_mean) == len(episodes):
            plt.plot(episodes, rolling_mean, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.xlabel('Training Episodes', fontsize=12)
        plt.ylabel('Mean Episodic Training Reward (Return)', fontsize=12)
        plt.title(f'Learning Curve: {prefix} - {obs_type}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Use absolute path to ensure we save in the right location
        abs_save_path = os.path.abspath(save_path)
        plt.savefig(abs_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        if os.path.exists(abs_save_path):
            file_size = os.path.getsize(abs_save_path)
            print(f"  ✓ Learning curve saved to {abs_save_path} ({file_size} bytes)")
        else:
            print(f"  ✗ Warning: Learning curve file was not created at {abs_save_path}")
    except Exception as e:
        print(f"  ✗ Error plotting learning curve for {prefix} - {obs_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            plt.close('all')  # Make sure to close any open figures
        except:
            pass


def save_performance_test_results(episode_rewards, prefix, obs_type, save_path):
    """Save performance test results as a markdown table"""
    try:
        print(f"  Saving performance test results to markdown: {len(episode_rewards)} data points")
        if len(episode_rewards) == 0:
            print(f"  ✗ Warning: No episode rewards to save for {prefix} - {obs_type}")
            return
        
        # Convert to numpy array if needed
        episode_rewards = np.array(episode_rewards)
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        median_reward = np.median(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        q25 = np.percentile(episode_rewards, 25)
        q75 = np.percentile(episode_rewards, 75)
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Create markdown content
        md_content = f"""# Performance Test Results: {prefix} - {obs_type}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Number of Episodes | {len(episode_rewards)} |
| Mean Reward | {mean_reward:.4f} |
| Standard Deviation | {std_reward:.4f} |
| Median Reward | {median_reward:.4f} |
| Minimum Reward | {min_reward:.4f} |
| Maximum Reward | {max_reward:.4f} |
| 25th Percentile (Q1) | {q25:.4f} |
| 75th Percentile (Q3) | {q75:.4f} |

## Episode-by-Episode Results

| Episode | Reward |
|---------|--------|
"""
        
        # Add all episode rewards to the table
        for i, reward in enumerate(episode_rewards, 1):
            md_content += f"| {i} | {reward:.4f} |\n"
        
        # Use absolute path to ensure we save in the right location
        abs_save_path = os.path.abspath(save_path)
        with open(abs_save_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Verify file was created
        if os.path.exists(abs_save_path):
            file_size = os.path.getsize(abs_save_path)
            print(f"  ✓ Performance test results saved to {abs_save_path} ({file_size} bytes)")
        else:
            print(f"  ✗ Warning: Markdown file was not created at {abs_save_path}")
    except Exception as e:
        print(f"  ✗ Error saving performance test results for {prefix} - {obs_type}: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_violin_performance(episode_rewards, prefix, obs_type, save_path):
    """Plot violin plot: Distribution of episodic rewards over 500 episodes"""
    try:
        print(f"  Attempting to plot violin performance: {len(episode_rewards)} data points")
        if len(episode_rewards) == 0:
            print(f"  ✗ Warning: No episode rewards to plot for {prefix} - {obs_type}")
            return
        
        # Convert to numpy array if needed
        episode_rewards = np.array(episode_rewards)
        print(f"  Data range: min={episode_rewards.min():.2f}, max={episode_rewards.max():.2f}, mean={episode_rewards.mean():.2f}")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Create violin plot
        parts = plt.violinplot([episode_rewards], positions=[0], showmeans=True, showmedians=True)
        
        # Customize violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_alpha(0.7)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('green')
        parts['cmedians'].set_linewidth(2)
        
        # Add statistics text
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        median_reward = np.median(episode_rewards)
        
        stats_text = f'Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}\nMedian: {median_reward:.2f}'
        plt.text(0.5, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)
        
        plt.ylabel('Mean Episodic Training Reward (Return)', fontsize=12)
        plt.title(f'Performance Test: {prefix} - {obs_type}\n(500 Episodes, No Exploration)', 
                  fontsize=14, fontweight='bold')
        plt.xticks([0], [f'{prefix}\n{obs_type}'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Use absolute path to ensure we save in the right location
        abs_save_path = os.path.abspath(save_path)
        plt.savefig(abs_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        if os.path.exists(abs_save_path):
            file_size = os.path.getsize(abs_save_path)
            print(f"  ✓ Violin plot saved to {abs_save_path} ({file_size} bytes)")
        else:
            print(f"  ✗ Warning: Violin plot file was not created at {abs_save_path}")
    except Exception as e:
        print(f"  ✗ Error plotting violin performance for {prefix} - {obs_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            plt.close('all')  # Make sure to close any open figures
        except:
            pass


def main():
    """Main training and evaluation pipeline"""
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    plots_dir = os.path.abspath("./plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    
    # Test plot generation to verify matplotlib is working
    print("Testing plot generation...")
    test_plot_path = os.path.join(plots_dir, "test_plot.png")
    try:
        plt.figure(figsize=(5, 3))
        plt.plot([1, 2, 3], [1, 4, 9], 'b-o')
        plt.title("Test Plot")
        plt.savefig(test_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        if os.path.exists(test_plot_path):
            print(f"✓ Plot test successful: {test_plot_path}")
            # Remove test plot
            try:
                os.remove(test_plot_path)
            except:
                pass
        else:
            print(f"✗ Plot test failed: file not created at {test_plot_path}")
    except Exception as e:
        print(f"✗ Plot test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Configuration
    total_timesteps = 200000  # Increased for proper learning (was 10000 - too low)
    n_eval_episodes = 500  # Standard evaluation episodes
    
    # Training configurations: (env_name, obs_type, prefix)
    configs = [
        ('group5-env-v0', 'LidarObservation', 'group5-env'),
        ('group5-env-v0', 'GrayscaleObservation', 'group5-env'),
        ('custom-highway-v0', 'LidarObservation', 'custom'),
        ('custom-highway-v0', 'GrayscaleObservation', 'custom'),
    ]
    
    # Train and evaluate for each configuration
    for env_name, obs_type, prefix in configs:
        try:
            # Train agent
            model, learning_curve = train_agent(
                env_name=env_name,
                obs_type=obs_type,
                total_timesteps=total_timesteps,
                prefix=prefix
            )
            
            # Plot learning curve
            learning_curve_path = f"./plots/{prefix}_{obs_type}_learning_curve.png"
            print(f"\nPlotting learning curve for {prefix} - {obs_type}...")
            print(f"  Learning curve data points: {len(learning_curve)}")
            plot_learning_curve(learning_curve, prefix, obs_type, learning_curve_path)
            
            # Evaluate agent
            episode_rewards = evaluate_agent(
                model=model,
                env_name=env_name,
                obs_type=obs_type,
                n_episodes=n_eval_episodes,
                prefix=prefix
            )
            
            # Plot violin plot
            violin_path = f"./plots/{prefix}_{obs_type}_performance_test.png"
            print(f"\nPlotting violin performance for {prefix} - {obs_type}...")
            print(f"  Evaluation data points: {len(episode_rewards)}")
            plot_violin_performance(episode_rewards, prefix, obs_type, violin_path)
            
            # Save performance test results as markdown table
            md_path = f"./plots/{prefix}_{obs_type}_performance_test.md"
            print(f"\nSaving performance test results to markdown for {prefix} - {obs_type}...")
            save_performance_test_results(episode_rewards, prefix, obs_type, md_path)
            
            print(f"\n✓ Completed {prefix} - {obs_type}")
            
        except Exception as e:
            print(f"\n✗ Error in {prefix} - {obs_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)
    
    # List actually generated plots
    plots_dir = os.path.abspath("./plots")
    print(f"\nChecking for generated plots in: {plots_dir}")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if plot_files:
            print(f"\n✓ Found {len(plot_files)} plot file(s):")
            for plot_file in sorted(plot_files):
                plot_path = os.path.join(plots_dir, plot_file)
                file_size = os.path.getsize(plot_path) / 1024  # Size in KB
                print(f"  - {plot_file} ({file_size:.1f} KB)")
        else:
            print("\n✗ No plot files found in plots directory")
            print("  Expected files:")
            print("    - Learning curves: ./plots/*_learning_curve.png")
            print("    - Performance tests: ./plots/*_performance_test.png")
    else:
        print(f"\n✗ Plots directory does not exist: {plots_dir}")


if __name__ == "__main__":
    main()


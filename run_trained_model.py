"""
Run Trained DRL Agent with GUI Visualization
Loads and runs pre-trained models with interactive visualization
"""

import os
import sys
import glob
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

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

# Add custom environment to path
custom_env_path = os.path.join(os.path.dirname(__file__), 'cs272-team-6-custom-env-master', 'custom')
sys.path.insert(0, custom_env_path)

# Import and register custom environment
try:
    from custom_env import AccidentEnv
    try:
        gym.envs.registry['custom-highway-v0']
    except KeyError:
        gym.register(
            id='custom-highway-v0',
            entry_point='custom_env:AccidentEnv',
        )
except ImportError as e:
    print(f"Warning: Could not import custom environment: {e}")
    AccidentEnv = None


def create_env(env_name, obs_type, render_mode='human'):
    """Create and configure environment with specified observation type for merge operations"""
    if env_name == 'group5-env-v0':
        # Configure observation based on type
        if obs_type == "GrayscaleObservation":
            observation_config = {
                "type": obs_type,
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140]
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
        try:
            env = gym.make('group5-env-v0', render_mode=render_mode, config=config)
        except TypeError:
            env = gym.make('group5-env-v0', render_mode=render_mode)
            env.unwrapped.configure(config)
    elif env_name == 'custom-highway-v0':
        # Configure observation based on type
        if obs_type == "GrayscaleObservation":
            observation_config = {
                "type": obs_type,
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140]
            }
        else:
            observation_config = {"type": obs_type}
        
        config = {
            "observation": observation_config,
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 15,
            "duration": 40,
            "initial_spacing": 2,
            "lane_change_reward": 0.15,  # ENCOURAGE lane changes (was 0, neutral)
            "right_lane_reward": 0.15,  # Increased from 0.1 to encourage merging to right lane
            "high_speed_reward": 0.4,  # Keep speed reward
            "collision_reward": -1.5,  # Strong penalty for crashes
        }
        try:
            env = gym.make('custom-highway-v0', render_mode=render_mode, config=config)
        except TypeError:
            env = gym.make('custom-highway-v0', render_mode=render_mode)
            env.unwrapped.configure(config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env


def find_available_models():
    """Find all available trained models in the models directory"""
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    
    # Look for final models (format: {prefix}_{obs_type}_final)
    final_models = glob.glob(os.path.join(models_dir, "*_final.zip"))
    for model_path in final_models:
        filename = os.path.basename(model_path)
        # Extract prefix and obs_type from filename
        # Format: {prefix}_{obs_type}_final.zip
        parts = filename.replace("_final.zip", "").split("_")
        if len(parts) >= 2:
            obs_type = parts[-1]  # Last part is observation type
            prefix = "_".join(parts[:-1])  # Everything before is prefix
            
            # Determine env_name from prefix
            if prefix == "group5-env":
                env_name = "group5-env-v0"
            elif prefix == "custom":
                env_name = "custom-highway-v0"
            else:
                continue
            
            available_models.append({
                'path': model_path.replace(".zip", ""),  # Remove .zip for loading
                'prefix': prefix,
                'obs_type': obs_type,
                'env_name': env_name,
                'type': 'final'
            })
    
    # Also look for best models in subdirectories
    best_model_dirs = glob.glob(os.path.join(models_dir, "*", "best_model.zip"))
    for model_path in best_model_dirs:
        dir_name = os.path.basename(os.path.dirname(model_path))
        # Format: {prefix}_{obs_type}/best_model.zip
        parts = dir_name.split("_")
        if len(parts) >= 2:
            obs_type = parts[-1]
            prefix = "_".join(parts[:-1])
            
            if prefix == "group5-env":
                env_name = "group5-env-v0"
            elif prefix == "custom":
                env_name = "custom-highway-v0"
            else:
                continue
            
            available_models.append({
                'path': model_path.replace(".zip", ""),
                'prefix': prefix,
                'obs_type': obs_type,
                'env_name': env_name,
                'type': 'best'
            })
    
    return available_models


def display_available_models(models):
    """Display available models in a numbered list"""
    if not models:
        print("\n✗ No trained models found in ./models/ directory")
        print("  Please train models first using merge_drl_agent.py")
        return
    
    print("\n" + "="*70)
    print("Available Trained Models:")
    print("="*70)
    for i, model in enumerate(models, 1):
        model_type_str = "Final" if model['type'] == 'final' else "Best"
        print(f"{i}. {model['prefix']} - {model['obs_type']} ({model_type_str} model)")
        print(f"   Environment: {model['env_name']}")
        print(f"   Path: {model['path']}")
        print()
    
    print("="*70)


def load_model(model_info, env):
    """Load a trained model and verify compatibility with environment"""
    model_path = model_info['path']
    print(f"\nLoading model from: {model_path}")
    
    try:
        model = PPO.load(model_path, env=env)
        print(f"✓ Model loaded successfully")
        
        # Verify observation space compatibility
        model_obs_space = model.observation_space
        env_obs_space = env.observation_space
        print(f"\nVerifying model compatibility:")
        print(f"  Model observation space: {model_obs_space}")
        print(f"  Environment observation space: {env_obs_space}")
        
        if model_obs_space != env_obs_space:
            print(f"  ⚠ Warning: Observation spaces don't match exactly!")
            print(f"     This might cause issues. Shapes: model={model_obs_space.shape}, env={env_obs_space.shape}")
        else:
            print(f"  ✓ Observation spaces match")
        
        # Verify action space compatibility
        model_action_space = model.action_space
        env_action_space = env.action_space
        print(f"  Model action space: {model_action_space}")
        print(f"  Environment action space: {env_action_space}")
        
        if model_action_space != env_action_space:
            print(f"  ✗ Error: Action spaces don't match!")
            print(f"     Model: {model_action_space}, Environment: {env_action_space}")
            return None
        else:
            print(f"  ✓ Action spaces match")
        
        # Action space info
        if hasattr(env_action_space, 'n'):
            print(f"  Action space size: {env_action_space.n}")
            print(f"  Actions: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER")
        
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_episode(model, env, episode_num, show_info=True, debug_actions=False):
    """Run a single episode with the trained model"""
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    action_counts = {}  # Track which actions are being taken
    action_sequence = []  # Track action sequence for debugging
    
    if show_info:
        print(f"\nEpisode {episode_num} started...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert action to scalar if it's an array
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        action = int(action)
        
        # Track actions
        action_counts[action] = action_counts.get(action, 0) + 1
        if debug_actions and episode_length < 20:  # Show first 20 actions
            action_names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
            action_sequence.append((episode_length, action, action_names.get(action, f"UNKNOWN({action})")))
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
    
    if show_info:
        print(f"Episode {episode_num} completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        if 'crashed' in info:
            print(f"  Crashed: {info['crashed']}")
        
        # Show action statistics
        print(f"  Action distribution:")
        action_names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        for action_id in sorted(action_counts.keys()):
            count = action_counts[action_id]
            percentage = (count / episode_length) * 100
            action_name = action_names.get(action_id, f"UNKNOWN({action_id})")
            print(f"    {action_name} ({action_id}): {count} times ({percentage:.1f}%)")
        
        if debug_actions and action_sequence:
            print(f"  First 20 actions:")
            for step, act, name in action_sequence[:20]:
                print(f"    Step {step}: {name} ({act})")
    
    return episode_reward, episode_length, info


def run_model_interactive(model_info, n_episodes=None, continuous=False):
    """Run model interactively with GUI"""
    print(f"\n{'='*70}")
    print(f"Running: {model_info['prefix']} - {model_info['obs_type']}")
    print(f"{'='*70}")
    
    # Create environment with GUI first (needed for model verification)
    print(f"\nCreating environment with GUI visualization...")
    env = create_env(
        env_name=model_info['env_name'],
        obs_type=model_info['obs_type'],
        render_mode='human'
    )
    print(f"✓ Environment created: {model_info['env_name']}")
    print(f"  Observation type: {model_info['obs_type']}")
    
    # Load model and verify compatibility
    model = load_model(model_info, env)
    if model is None:
        env.close()
        return
    
    print(f"\n{'='*70}")
    print("GUI Window Controls:")
    print("  - Close the window to stop")
    print("  - The agent will run automatically")
    print(f"{'='*70}\n")
    
    episode_rewards = []
    episode_lengths = []
    episode_num = 1
    
    try:
        # Enable action debugging for first episode
        debug_first = True
        
        if continuous:
            print("Running in continuous mode (press Ctrl+C to stop)...")
            while True:
                reward, length, info = run_episode(model, env, episode_num, show_info=True, debug_actions=debug_first)
                episode_rewards.append(reward)
                episode_lengths.append(length)
                episode_num += 1
                debug_first = False  # Only debug first episode
        elif n_episodes:
            print(f"Running {n_episodes} episodes...")
            for i in range(n_episodes):
                reward, length, info = run_episode(model, env, i + 1, show_info=True, debug_actions=(i == 0))
                episode_rewards.append(reward)
                episode_lengths.append(length)
        else:
            # Single episode by default
            reward, length, info = run_episode(model, env, 1, show_info=True, debug_actions=True)
            episode_rewards.append(reward)
            episode_lengths.append(length)
        
        # Print summary statistics
        if len(episode_rewards) > 0:
            print(f"\n{'='*70}")
            print("Summary Statistics:")
            print(f"{'='*70}")
            print(f"Total Episodes: {len(episode_rewards)}")
            print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Min Reward: {np.min(episode_rewards):.2f}")
            print(f"Max Reward: {np.max(episode_rewards):.2f}")
            print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} steps")
            print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed")


def main():
    """Main function to run trained models"""
    print("\n" + "="*70)
    print("DRL Agent Runner - GUI Visualization")
    print("="*70)
    
    # Find available models
    models = find_available_models()
    
    if not models:
        print("\n✗ No trained models found!")
        print("\nTo train models, run:")
        print("  python merge_drl_agent.py")
        return
    
    # Display available models
    display_available_models(models)
    
    # Let user select model
    while True:
        try:
            choice = input(f"\nSelect a model to run (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting...")
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Ask for run mode
    print("\nRun Mode:")
    print("  1. Single episode")
    print("  2. Multiple episodes (specify count)")
    print("  3. Continuous mode (run until stopped)")
    
    while True:
        try:
            mode_choice = input("\nSelect mode (1-3): ").strip()
            if mode_choice == '1':
                n_episodes = None
                continuous = False
                break
            elif mode_choice == '2':
                n_episodes = int(input("Number of episodes: "))
                continuous = False
                break
            elif mode_choice == '3':
                n_episodes = None
                continuous = True
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3")
        except ValueError:
            print("Invalid input. Please enter a number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Run the model
    run_model_interactive(selected_model, n_episodes=n_episodes, continuous=continuous)


if __name__ == "__main__":
    main()


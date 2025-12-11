# DRL Agent for Highway Merge Operations

This project implements a Deep Reinforcement Learning (DRL) agent using Proximal Policy Optimization (PPO) to perform merge operations on two highway environments:

1. **Base highway-env** (`highway-v0`) from Gymnasium
2. **Custom highway environment** (`custom-highway-v0`) from the custom environment package

For each environment, the agent is trained and evaluated with two different observation types:
- **LidarObservation**: Vector-based observation using LiDAR sensors
- **GrayscaleObservation**: Image-based observation using grayscale camera

## Project Structure

```
.
├── train_drl_agent.py          # Main training and evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── plots/                      # Generated plots (created automatically)
├── models/                     # Saved trained models (created automatically)
├── logs/                       # Training logs (created automatically)
├── tensorboard_logs/          # TensorBoard logs (created automatically)
├── cs272-team-6-custom-env-master/  # Custom environment package
│   └── custom/
│       └── custom_env.py       # Custom environment implementation
└── highway env.txt            # Reference for base highway-env
```

## Installation

### Step 1: Install Python Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install gymnasium>=0.29.0 highway-env>=1.8.0 stable-baselines3>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 numpy>=1.24.0 pygame>=2.5.0 pandas>=2.0.0 scipy>=1.10.0 tensorboard>=2.13.0
```

### Step 2: Verify Custom Environment

Ensure the custom environment is in the correct location:
- The `cs272-team-6-custom-env-master` folder should be in the project root
- The `custom_env.py` file should be in `cs272-team-6-custom-env-master/custom/`

## Training and Evaluation

### Step 3: Run Training and Evaluation

Execute the main training script:

```bash
python train_drl_agent.py
```

This will:
1. Train 4 different agent configurations:
   - `highway-v0` with `LidarObservation`
   - `highway-v0` with `GrayscaleObservation`
   - `custom-highway-v0` with `LidarObservation`
   - `custom-highway-v0` with `GrayscaleObservation`

2. For each configuration:
   - Train a PPO agent for 100,000 timesteps
   - Generate a learning curve plot
   - Evaluate the trained model over 500 episodes (no exploration)
   - Generate a performance test violin plot

3. Save all results:
   - Models: `./models/{prefix}_{obs_type}_final/`
   - Learning curves: `./plots/{prefix}_{obs_type}_learning_curve.png`
   - Performance tests: `./plots/{prefix}_{obs_type}_performance_test.png`

### Training Time

Training time depends on your hardware:
- **CPU only**: ~2-4 hours per configuration (4 configurations total)
- **GPU**: ~30-60 minutes per configuration

You can reduce training time by modifying `total_timesteps` in `train_drl_agent.py` (line 298), but this may affect learning quality.

## Generated Outputs

### Plots Directory Structure

After training completes, you'll find the following plots in `./plots/`:

```
plots/
├── highway-env_LidarObservation_learning_curve.png
├── highway-env_LidarObservation_performance_test.png
├── highway-env_GrayscaleObservation_learning_curve.png
├── highway-env_GrayscaleObservation_performance_test.png
├── custom_LidarObservation_learning_curve.png
├── custom_LidarObservation_performance_test.png
├── custom_GrayscaleObservation_learning_curve.png
└── custom_GrayscaleObservation_performance_test.png
```

### Plot Descriptions

1. **Learning Curves**: Show mean episodic training reward vs. training episodes
   - Includes raw episode rewards (transparent) and moving average (solid line)
   - File naming: `{prefix}_{obs_type}_learning_curve.png`

2. **Performance Test Violin Plots**: Show distribution of episodic rewards over 500 evaluation episodes
   - Includes mean, median, and standard deviation statistics
   - File naming: `{prefix}_{obs_type}_performance_test.png`

## Configuration

### Modify Training Parameters

Edit `train_drl_agent.py` to customize:

- **Training timesteps** (line 298): `total_timesteps = 100000`
- **Evaluation episodes** (line 299): `n_eval_episodes = 500`
- **PPO hyperparameters** (lines 114-127): learning rate, batch size, etc.

### Environment Configuration

The environments are configured for merge operations in the `create_env()` function:

- **highway-v0**: Configured with lane change rewards and merge-focused settings
- **custom-highway-v0**: Uses the custom environment's merge/accident avoidance scenario

## Troubleshooting

### Issue: Custom environment not found

**Solution**: Ensure the `cs272-team-6-custom-env-master` folder is in the project root and contains `custom/custom_env.py`

### Issue: Import errors

**Solution**: 
1. Verify all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version (3.8+ recommended)

### Issue: Out of memory errors

**Solution**: 
1. Reduce `total_timesteps` in the training script
2. Reduce `batch_size` in PPO configuration
3. Use CPU instead of GPU if GPU memory is limited

### Issue: Plots not generated

**Solution**: 
1. Check that training completed successfully
2. Verify the `plots/` directory was created
3. Check console output for error messages

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir ./tensorboard_logs
```

Then open `http://localhost:6006` in your browser.

### Console Output

The script prints progress updates:
- Training start/end for each configuration
- Model save locations
- Plot generation confirmations

## Results Summary

After training completes, the script will print a summary of all generated plots and their locations.

## Notes

- The agent focuses on **merge operations**: lane changes, merging behavior, and safe navigation
- All models are saved and can be reloaded for further evaluation
- Training logs are saved in `./logs/` for detailed analysis
- The custom environment (`custom-highway-v0`) simulates a car accident scenario requiring merge/avoidance behavior

## Citation

If you use the custom environment, please cite:
- CS272 Team 6 - Custom Environment (AccidentEnv)
- Highway-Env: https://github.com/eleurent/highway-env
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3


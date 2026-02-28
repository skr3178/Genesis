"""
Training Script for Cartpole RL using Genesis + RSL-RL

This script demonstrates how to train a policy using PPO (Proximal Policy Optimization)
on the cartpole balancing task with Genesis physics simulator.

Usage:
    python cartpole_train.py -e cartpole-exp -B 1024 --max_iterations 300

Requirements:
    pip install rsl-rl-lib==2.2.4
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

# Check for correct rsl-rl-lib version
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from cartpole_env import CartpoleEnv


def get_train_cfg(exp_name, max_iterations):
    """
    Define the training configuration for PPO.
    
    These hyperparameters control how the policy learns. Key ones:
    - learning_rate: How fast the network updates (0.001 is typical)
    - gamma: Discount factor for future rewards (0.99 is standard)
    - lam: GAE lambda for advantage estimation (0.95 is standard)
    - clip_param: PPO clipping parameter (0.2 is standard)
    - num_steps_per_env: How many steps to collect before updating
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 25,
        "empirical_normalization": None,
        "seed": 1,
    }
    
    return train_cfg_dict


def get_cfgs():
    """
    Define all configuration dictionaries for the environment.
    
    Returns:
        env_cfg: Physics and robot configuration
        obs_cfg: Observation scaling
        reward_cfg: Reward function weights
    """
    # Environment configuration
    env_cfg = {
        "num_actions": 1,  # We control only the cart (1D force/velocity)
        # Physics parameters
        "dt": 0.0025,  # 400Hz control frequency
        "episode_length_s": 10.0,  # Each episode is 10 seconds
        # Control parameters
        "kp": 50.0,  # PD controller position gain (reduced for stability)
        "kd": 5.0,   # PD controller velocity gain (reduced for stability)
        "action_scale": 0.5,  # Scale for actions (reduced for stability)
        "clip_actions": 10.0,  # Action clipping (reduced)
        # Termination conditions
        "termination_angle": 0.5,  # ~28 degrees (pole falls)
        "cart_limit": 2.0,  # Cart hits boundary
    }
    
    # Observation configuration
    obs_cfg = {
        "num_obs": 4,  # [cart_pos, cart_vel, pole_angle, pole_vel]
        "obs_scales": {
            "cart_pos": 1.0,
            "cart_vel": 1.0,
            "pole_angle": 1.0,
            "pole_vel": 1.0,
        },
    }
    
    # Reward configuration
    # Positive rewards encourage desired behavior
    # Negative rewards (penalties) discourage undesired behavior
    reward_cfg = {
        "reward_scales": {
            "pole_upright": 10.0,   # Main reward: keep pole upright
            "pole_balance": 0.1,    # Small bonus: minimize pole velocity
            "cart_center": 0.5,     # Penalty: stay near center
            "survival": 1.0,        # Reward: stay alive longer
        },
    }
    
    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser(description="Train cartpole balancing policy")
    parser.add_argument("-e", "--exp_name", type=str, default="cartpole-rl",
                        help="Experiment name for logging")
    parser.add_argument("-B", "--num_envs", type=int, default=1024,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=300,
                        help="Number of training iterations")
    parser.add_argument("-v", "--vis", action="store_true",
                        help="Enable visualization during training (slower)")
    args = parser.parse_args()
    
    # Setup logging directory
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # Clean up and create log directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configurations for later evaluation
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    # ============================================================
    # Initialize Genesis
    # ============================================================
    # backend=gs.gpu uses CUDA for GPU acceleration
    # precision="32" uses float32 (standard)
    # performance_mode=True optimizes for speed
    gs.init(
        backend=gs.cpu,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
    )
    
    # ============================================================
    # Create Environment
    # ============================================================
    print(f"Creating {args.num_envs} parallel environments...")
    env = CartpoleEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    
    # ============================================================
    # Create PPO Runner
    # ============================================================
    # OnPolicyRunner handles:
    # - Collecting rollouts (experience) from the environment
    # - Computing advantages using GAE
    # - Updating the policy and value networks
    # - Logging training statistics
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    print(f"Starting training for {args.max_iterations} iterations...")
    print(f"Logs saved to: {log_dir}")
    print(f"To evaluate: python cartpole_eval.py -e {args.exp_name} --ckpt {args.max_iterations}")
    
    # ============================================================
    # Train the Policy
    # ============================================================
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

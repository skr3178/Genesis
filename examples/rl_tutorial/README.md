# Genesis + Reinforcement Learning Tutorial

This tutorial demonstrates how to create a reinforcement learning pipeline using **Genesis** physics simulator and **RSL-RL** library.

## Overview

The pattern consists of three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   cartpole_env  │────▶│  cartpole_train │────▶│  cartpole_eval  │
│   (Environment) │     │   (Training)    │     │  (Evaluation)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## File Structure

```
examples/rl_tutorial/
├── cartpole_env.py      # Environment definition (Scene, rewards, resets)
├── cartpole_train.py    # Training script (PPO configuration)
├── cartpole_eval.py     # Evaluation script (visualize trained policy)
└── README.md            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Genesis (already in project)
pip install -e .

# Install RSL-RL (specific version required)
pip install rsl-rl-lib==2.2.4
```

### 2. Train a Policy

```bash
cd examples/rl_tutorial
python cartpole_train.py -e my-exp --max_iterations 300
```

**Arguments:**
- `-e, --exp_name`: Experiment name for logging
- `-B, --num_envs`: Number of parallel environments (default: 1024)
- `--max_iterations`: Training iterations (default: 300)
- `-v, --vis`: Enable visualization (slower)

### 3. Evaluate the Trained Policy

```bash
python cartpole_eval.py -e my-exp --ckpt 300
```

## Key Concepts

### 1. Environment Class (`cartpole_env.py`)

The environment encapsulates the simulation and follows the standard Gym interface:

```python
class CartpoleEnv:
    def __init__(self, num_envs, ...):
        # 1. Create Genesis Scene
        self.scene = gs.Scene(...)
        
        # 2. Add entities (robots, objects, ground)
        self.scene.add_entity(gs.morphs.Plane())
        self.cartpole = self.scene.add_entity(gs.morphs.URDF(...))
        
        # 3. Build for parallel simulation
        self.scene.build(n_envs=num_envs)
    
    def step(self, actions):
        # Apply actions, step simulation, compute rewards
        self.scene.step()
        return obs, rewards, dones, extras
    
    def reset(self):
        # Reset environments
        return obs, extras
```

**Important Methods:**
- `step(actions)`: Execute one simulation step
- `reset()`: Reset environments
- `_reward_*()`: Reward functions (automatically discovered)

### 2. Training Configuration

The training config uses PPO (Proximal Policy Optimization):

```python
train_cfg = {
    "algorithm": {
        "class_name": "PPO",
        "learning_rate": 0.001,
        "gamma": 0.99,        # Discount factor
        "lam": 0.95,          # GAE lambda
        "clip_param": 0.2,    # PPO clipping
    },
    "policy": {
        "actor_hidden_dims": [128, 128],
        "critic_hidden_dims": [128, 128],
    },
    "num_steps_per_env": 100,  # Steps before policy update
}
```

### 3. Parallel Environments

Genesis supports massive parallelization:

```python
# All 4096 environments run simultaneously on GPU!
env = CartpoleEnv(num_envs=4096, ...)
```

This is what makes Genesis fast - you can train with thousands of environments in parallel.

## Customizing for Your Robot

To adapt this for your own robot:

### 1. Modify the Environment

```python
# In __init__:
self.robot = self.scene.add_entity(
    gs.morphs.URDF(
        file="urdf/my_robot/robot.urdf",  # Your URDF
        pos=(0, 0, 0.5),
    ),
)

# Get controllable joints
self.motor_dof_idx = [0, 1, 2, ...]  # Your joint indices
```

### 2. Define Observations

```python
def _update_observation(self):
    self.obs_buf = torch.cat([
        self.joint_positions,
        self.joint_velocities,
        self.base_orientation,
        # ... your custom observations
    ], dim=-1)
```

### 3. Define Reward Functions

```python
def _reward_task_completion(self):
    return distance_to_target  # Your reward logic

def _reward_energy_efficiency(self):
    return -torch.sum(self.torques ** 2, dim=-1)
```

## Advanced Patterns from Other Examples

### Go2 Quadruped (`examples/locomotion/`)
- **Velocity tracking**: Commands for forward/sideways/rotation speed
- **PD control**: Position control with kp/kd gains
- **Domain randomization**: Randomize friction, mass, etc.

### Drone Hovering (`examples/drone/`)
- **Direct force control**: Set propeller RPMs directly
- **3D navigation**: Position targets in 3D space

### Franka Manipulation (`examples/manipulation/`)
- **Vision observations**: Camera RGB images
- **Behavior cloning**: Train from demonstrations
- **Inverse kinematics**: Control end-effector pose

## Troubleshooting

### "rsl_rl not found"
```bash
pip uninstall rsl_rl rsl-rl-lib  # Remove any existing
pip install rsl-rl-lib==2.2.4     # Install correct version
```

### Training is slow
- Increase `num_envs` (4096+ for GPU)
- Enable `performance_mode=True` in `gs.init()`
- Disable viewer during training (`show_viewer=False`)

### Policy doesn't learn
- Check reward scales (should be roughly similar magnitude)
- Increase `num_steps_per_env` (more data per update)
- Check observation ranges (should be normalized ~[-1, 1])
- Verify termination conditions aren't too strict

## Resources

- **Genesis Docs**: https://genesis-world.readthedocs.io/
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl
- **PPO Paper**: https://arxiv.org/abs/1707.06347

## Example Training Output

```
Creating 1024 parallel environments...
Starting training for 300 iterations...
Logs saved to: logs/cartpole-rl

################################################################################
#                        Learning iteration 1/300                        #
################################################################################
Episode Rewards: {...}
Mean reward: 8.45
...

Training complete!
```

Then evaluate:
```
python cartpole_eval.py -e cartpole-rl --ckpt 300
# Episode 1: length=500, reward=450.23
# Episode 2: length=500, reward=460.12
```

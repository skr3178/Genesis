"""
Simple Cartpole Environment for Reinforcement Learning with Genesis

This demonstrates the standard pattern for creating RL environments in Genesis:
1. Create a Scene with simulation options
2. Add entities (cart, pole, ground)
3. Define step(), reset(), and reward functions
4. Support parallel environments via n_envs
"""

import torch
import math
import genesis as gs


class CartpoleEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        """
        Initialize the Cartpole environment.
        
        Args:
            num_envs: Number of parallel environments to simulate
            env_cfg: Environment configuration dict
            obs_cfg: Observation configuration dict  
            reward_cfg: Reward configuration dict
            show_viewer: Whether to show the 3D viewer
        """
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        
        # Simulation parameters
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        
        # Store configs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        
        # ============================================================
        # Create the Genesis Scene
        # ============================================================
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=16,  # Increased for stability with perturbed initial states
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,  # Simpler without collision for cartpole
                enable_joint_limit=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
            renderer=None,  # Disable renderer for headless training
        )
        
        # ============================================================
        # Add Entities to the Scene
        # ============================================================
        
        # Ground plane
        self.scene.add_entity(gs.morphs.Plane())
        
        # Cartpole robot with sliding cart and rotating pole
        self.cartpole = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/cartpole/urdf/cartpole.urdf",
                pos=(0.0, 0.0, 0.0),
            ),
        )
        
        # ============================================================
        # Build the Scene (required before simulation)
        # ============================================================
        # Patch to skip rasterizer build on headless systems (no EGL)
        if not show_viewer:
            original_vis_build = self.scene._visualizer.build
            def patched_vis_build():
                self.scene._visualizer._context.build(self.scene)
                if self.scene._visualizer._viewer is not None:
                    self.scene._visualizer._viewer.build(self.scene)
                # Skip rasterizer.build() - causes EGL error on headless systems
                if self.scene._visualizer._raytracer is not None:
                    self.scene._visualizer._raytracer.build(self.scene)
                for camera in self.scene._visualizer._cameras:
                    camera.build()
                if self.scene._visualizer._batch_renderer is not None:
                    self.scene._visualizer._batch_renderer.build()
                self.scene._visualizer._is_built = True
                self.scene._visualizer.reset()
            self.scene._visualizer.build = patched_vis_build
        
        self.scene.build(n_envs=num_envs)
        
        # ============================================================
        # Get Joint Indices for Control
        # ============================================================
        # Note: Genesis adds a root_joint (6 DOFs) for the base link
        # Joint 0: root_joint (6 DOFs) - free floating base
        # Joint 1: slider_to_cart (1 DOF) - cart slides on x-axis
        # Joint 2: cart_to_pole (1 DOF) - pole rotates around y-axis
        # Total: 8 DOFs (6 + 1 + 1)
        self.cart_dof_idx = 6   # prismatic joint (sliding) - after root_joint
        self.pole_dof_idx = 7   # revolute joint (rotation)
        self.motor_dof_idx = [self.cart_dof_idx]  # we only control the cart
        
        # Set PD control gains for the cart
        self.cartpole.set_dofs_kp([self.env_cfg["kp"]], [self.cart_dof_idx])
        self.cartpole.set_dofs_kv([self.env_cfg["kd"]], [self.cart_dof_idx])
        
        # ============================================================
        # Initialize Buffers (tensors for GPU-accelerated computation)
        # ============================================================
        
        # State buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        
        # Action buffer
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        
        # State information
        self.cart_pos = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.cart_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_angle = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        
        # Extra info for logging
        self.extras = {"observations": {}}
        
        # Episode reward tracking
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
    
    def step(self, actions):
        """
        Execute one simulation step with the given actions.
        
        Args:
            actions: Tensor of shape (num_envs, num_actions) with control commands
            
        Returns:
            obs: New observations (num_envs, num_obs)
            rewards: Rewards for each environment (num_envs,)
            dones: Boolean tensor indicating which environments are done (num_envs,)
            extras: Additional info for logging
        """
        # Store actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Apply action: target position for the cart
        # Action is a velocity command that gets integrated to position
        target_cart_vel = self.actions * self.env_cfg["action_scale"]  # (num_envs, 1)
        current_cart_pos = self.cartpole.get_dofs_position([self.cart_dof_idx])  # (num_envs, 1)
        target_cart_pos = current_cart_pos + target_cart_vel * self.dt  # (num_envs, 1)
        
        # Clamp target position to prevent extreme control commands
        target_cart_pos = torch.clamp(target_cart_pos, -self.env_cfg["cart_limit"], self.env_cfg["cart_limit"])
        
        self.cartpole.control_dofs_position(target_cart_pos, [self.cart_dof_idx])
        
        # Step the simulation
        self.scene.step()
        
        # Update episode length
        self.episode_length_buf += 1
        
        # Get observations from simulation state
        self._update_state()
        
        # Compute rewards
        self.rew_buf[:] = 0.0
        for name, scale in self.reward_scales.items():
            reward_fn = getattr(self, f"_reward_{name}")
            rew = reward_fn() * scale * self.dt  # scale by dt for stability
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # Check termination conditions
        # 1. Pole angle too large (fallen over)
        pole_fallen = torch.abs(self.pole_angle) > self.env_cfg["termination_angle"]
        # 2. Cart out of bounds
        cart_out_of_bounds = torch.abs(self.cart_pos) > self.env_cfg["cart_limit"]
        # 3. Episode timeout
        time_out = self.episode_length_buf > self.max_episode_length
        
        self.reset_buf = pole_fallen | cart_out_of_bounds | time_out
        
        # Track timeouts for proper value function bootstrapping
        self.extras["time_outs"] = time_out.to(dtype=gs.tc_float)
        
        # Reset environments that are done
        envs_to_reset = self.reset_buf.nonzero(as_tuple=False).reshape(-1)
        if len(envs_to_reset) > 0:
            self._reset_idx(envs_to_reset)
        
        # Build observation tensor
        self._update_observation()
        
        self.extras["observations"]["critic"] = self.obs_buf
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _update_state(self):
        """Read current state from simulation."""
        # Get joint positions and velocities
        dof_pos = self.cartpole.get_dofs_position()
        dof_vel = self.cartpole.get_dofs_velocity()
        
        # Cart state (prismatic joint)
        self.cart_pos = dof_pos[:, self.cart_dof_idx]
        self.cart_vel = dof_vel[:, self.cart_dof_idx]
        
        # Pole state (revolute joint) - angle is the position
        self.pole_angle = dof_pos[:, self.pole_dof_idx]
        self.pole_vel = dof_vel[:, self.pole_dof_idx]
    
    def _update_observation(self):
        """Build observation tensor from state."""
        # Standard cartpole observations: [cart_pos, cart_vel, pole_angle, pole_vel]
        self.obs_buf = torch.stack([
            self.cart_pos * self.obs_cfg["obs_scales"]["cart_pos"],
            self.cart_vel * self.obs_cfg["obs_scales"]["cart_vel"],
            self.pole_angle * self.obs_cfg["obs_scales"]["pole_angle"],
            self.pole_vel * self.obs_cfg["obs_scales"]["pole_vel"],
        ], dim=-1)
    
    def _reset_idx(self, envs_idx):
        """Reset specific environments by index."""
        if len(envs_idx) == 0:
            return
        
        # Reset cartpole to random initial state near upright
        # Cart position: random small offset
        random_cart_pos = torch.rand((len(envs_idx),), device=gs.device) * 0.02 - 0.01  # [-0.01, 0.01]
        
        # Pole angle: random small angle near upright
        random_pole_angle = torch.rand((len(envs_idx),), device=gs.device) * 0.02 - 0.01  # [-0.01, 0.01] rad (~[-0.57°, 0.57°])
        
        # Set positions using qpos indices (different from DOF indices!)
        # Note: qpos uses different indexing due to root joint quaternion representation
        # Joint 0 (root): q_start=0, n_qs=7 (3 pos + 4 quat)
        # Joint 1 (slider): q_start=7, n_qs=1 (cart position)
        # Joint 2 (revolute): q_start=8, n_qs=1 (pole angle)
        qpos = torch.zeros((len(envs_idx), self.cartpole.n_qs), device=gs.device)
        qpos[:, 7] = random_cart_pos   # cart position at qpos index 7
        qpos[:, 8] = random_pole_angle  # pole angle at qpos index 8
        
        self.cartpole.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True)
        
        # Reset buffers
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        
        # Log episode rewards
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            self.extras["episode"][f"rew_{key}"] = torch.mean(value[envs_idx]).item() / self.env_cfg["episode_length_s"]
            value[envs_idx] = 0.0
    
    def reset(self):
        """Reset all environments."""
        all_envs = torch.arange(self.num_envs, device=gs.device)
        self._reset_idx(all_envs)
        self._update_state()
        self._update_observation()
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_observations(self):
        """Get current observations without stepping."""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras
    
    def get_privileged_observations(self):
        """For asymmetric actor-critic (optional)."""
        return None
    
    # ============================================================
    # Reward Functions
    # ============================================================
    
    def _reward_pole_upright(self):
        """Reward for keeping the pole upright (cosine of angle)."""
        return torch.cos(self.pole_angle)
    
    def _reward_pole_balance(self):
        """Penalty for pole angular velocity."""
        return -torch.abs(self.pole_vel)
    
    def _reward_cart_center(self):
        """Penalty for cart position far from center."""
        return -torch.abs(self.cart_pos)
    
    def _reward_survival(self):
        """Reward for staying alive (encourages longer episodes)."""
        return torch.ones_like(self.rew_buf)

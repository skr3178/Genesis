"""
CartpoleEnv V2 - Genesis RL Environment with Large Initial Perturbations

Key differences from v1:
- Initial pole angle: ±0.3 rad (~17°) - visually obvious at video start
- Initial cart position: ±0.5 m - clearly off-center
- Wider termination angle (0.8 rad ≈ 46°) so the policy has room to recover
- Simpler build (no visualizer patch needed - let Genesis use EGL normally)
"""

import torch
import math
import genesis as gs


class CartpoleEnvV2:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_scales = reward_cfg["reward_scales"]

        # ── Scene ────────────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=16),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_joint_limit=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        # ── Entities ─────────────────────────────────────────────────
        self.scene.add_entity(gs.morphs.Plane())
        self.cartpole = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/cartpole/urdf/cartpole.urdf",
                pos=(0.0, 0.0, 0.0),
            ),
        )

        # ── Build ─────────────────────────────────────────────────────
        self.scene.build(n_envs=num_envs)

        # ── DOF indices ───────────────────────────────────────────────
        # root_joint (6 DOFs), slider (1 DOF idx=6), revolute (1 DOF idx=7)
        self.cart_dof_idx = 6
        self.pole_dof_idx = 7
        self.motor_dof_idx = [self.cart_dof_idx]

        self.cartpole.set_dofs_kp([env_cfg["kp"]], [self.cart_dof_idx])
        self.cartpole.set_dofs_kv([env_cfg["kd"]], [self.cart_dof_idx])

        # ── Buffers ───────────────────────────────────────────────────
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.actions = torch.zeros((num_envs, env_cfg["num_actions"]), dtype=gs.tc_float, device=gs.device)

        self.cart_pos = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.cart_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_angle = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

        self.extras = {"observations": {}}
        self.episode_sums = {
            name: torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
            for name in self.reward_scales
        }

    # ─────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        target_cart_vel = self.actions * self.env_cfg["action_scale"]
        current_cart_pos = self.cartpole.get_dofs_position([self.cart_dof_idx])
        target_cart_pos = current_cart_pos + target_cart_vel * self.dt
        target_cart_pos = torch.clamp(target_cart_pos, -self.env_cfg["cart_limit"], self.env_cfg["cart_limit"])

        self.cartpole.control_dofs_position(target_cart_pos, [self.cart_dof_idx])
        self.scene.step()
        self.episode_length_buf += 1

        self._update_state()

        # Rewards
        self.rew_buf[:] = 0.0
        for name, scale in self.reward_scales.items():
            rew = getattr(self, f"_reward_{name}")() * scale * self.dt
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Termination
        pole_fallen = torch.abs(self.pole_angle) > self.env_cfg["termination_angle"]
        cart_oob = torch.abs(self.cart_pos) > self.env_cfg["cart_limit"]
        time_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = pole_fallen | cart_oob | time_out
        self.extras["time_outs"] = time_out.to(dtype=gs.tc_float)

        envs_done = self.reset_buf.nonzero(as_tuple=False).reshape(-1)
        if len(envs_done) > 0:
            self._reset_idx(envs_done)

        self._update_observation()
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ─────────────────────────────────────────────────────────────────
    # State helpers
    # ─────────────────────────────────────────────────────────────────
    def _update_state(self):
        dof_pos = self.cartpole.get_dofs_position()
        dof_vel = self.cartpole.get_dofs_velocity()
        self.cart_pos = dof_pos[:, self.cart_dof_idx]
        self.cart_vel = dof_vel[:, self.cart_dof_idx]
        self.pole_angle = dof_pos[:, self.pole_dof_idx]
        self.pole_vel = dof_vel[:, self.pole_dof_idx]

    def _update_observation(self):
        scales = self.obs_cfg["obs_scales"]
        self.obs_buf = torch.stack([
            self.cart_pos   * scales["cart_pos"],
            self.cart_vel   * scales["cart_vel"],
            self.pole_angle * scales["pole_angle"],
            self.pole_vel   * scales["pole_vel"],
        ], dim=-1)

    # ─────────────────────────────────────────────────────────────────
    # Reset - LARGE PERTURBATIONS so the video shows clear recovery
    # ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)
        perturb = self.env_cfg["init_perturbation"]

        # Uniformly random in [-perturb, +perturb]
        random_cart_pos  = (torch.rand(n, device=gs.device) * 2 - 1) * perturb["cart_pos"]
        random_pole_angle = (torch.rand(n, device=gs.device) * 2 - 1) * perturb["pole_angle"]
        random_cart_vel  = (torch.rand(n, device=gs.device) * 2 - 1) * perturb["cart_vel"]
        random_pole_vel  = (torch.rand(n, device=gs.device) * 2 - 1) * perturb["pole_vel"]

        # qpos: root (7 vals) + cart (1 val) + pole (1 val)
        qpos = torch.zeros((n, self.cartpole.n_qs), device=gs.device)
        qpos[:, 7] = random_cart_pos
        qpos[:, 8] = random_pole_angle
        self.cartpole.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True)

        # Also set velocities
        dof_vel = torch.zeros((n, self.cartpole.n_dofs), device=gs.device)
        dof_vel[:, self.cart_dof_idx] = random_cart_vel
        dof_vel[:, self.pole_dof_idx] = random_pole_vel
        self.cartpole.set_dofs_velocity(dof_vel, envs_idx=envs_idx)

        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key, val in self.episode_sums.items():
            self.extras["episode"][f"rew_{key}"] = torch.mean(val[envs_idx]).item() / self.env_cfg["episode_length_s"]
            val[envs_idx] = 0.0

    def reset(self):
        all_envs = torch.arange(self.num_envs, device=gs.device)
        self._reset_idx(all_envs)
        self._update_state()
        self._update_observation()
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ─────────────────────────────────────────────────────────────────
    # Reward functions
    # ─────────────────────────────────────────────────────────────────
    def _reward_pole_upright(self):
        return torch.cos(self.pole_angle)

    def _reward_pole_balance(self):
        return -torch.abs(self.pole_vel)

    def _reward_cart_center(self):
        return -torch.abs(self.cart_pos)

    def _reward_survival(self):
        return torch.ones_like(self.rew_buf)

import os
# Use EGL for headless/offscreen rendering with GPU acceleration
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Fix for PyOpenGL import order issue - monkey-patch _errors before GL is imported
import sys
class _FakeErrorsModule:
    _error_checker = None
sys.modules["OpenGL.raw.GL._errors"] = _FakeErrorsModule()

# Suppress MuJoCo's GL platform check (it conflicts with EGL)
os.environ["MUJOCO_GL"] = "egl"

import argparse
import pickle
from importlib import metadata

import torch

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

from hover_env import HoverEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.record,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            video_path = os.path.join(script_dir, "drone_video.mp4")
            env.cam.stop_recording(save_to_filename=video_path, fps=env_cfg["max_visualize_FPS"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/drone/hover_eval.py

# Note
If you experience slow performance or encounter other issues
during evaluation, try removing the --record option.
"""

1. MPM Options Meaning

mpm_options=gs.options.MPMOptions(
    dt=5e-4,                    # Time step for MPM simulation (smaller = more stable)
    lower_bound=(-1.0, -1.0, -0.2),  # Simulation domain min bounds (x, y, z)
    upper_bound=(1.0, 1.0, 1.0),     # Simulation domain max bounds (x, y, z)
)

• MPM = Material Point Method (a physics simulation technique for soft bodies/fluids)
• dt = Time step specifically for MPM (separate from global sim dt)
• lower_bound/upper_bound = The bounding box where MPM particles can exist (optimization)

2. No gs.cpu or gs.gpu?

Genesis auto-detects the backend. From the logs:
[Genesis] [INFO] Running on [NVIDIA GeForce RTX 3060] with backend gs.cuda
It automatically uses gs.cuda (GPU) if available. You can explicitly set it in gs.init():
gs.init(backend=gs.gpu)  # or gs.cpu, gs.cuda, gs.metal, gs.vulkan

3. show_viewer Not Set
In the working script, show_viewer defaults to False (no GUI window). The video is recorded headlessly using:

cam = scene.add_camera(...)  # Off-screen camera
cam.start_recording()
cam.render()  # Capture frames
cam.stop_recording(save_to_filename='...')  # Save video

If you want to see a live window AND record:

scene = gs.Scene(
    ...,
    show_viewer=True,  # Opens live GUI window
)

4. scene.build() Function

scene.build()           # Single environment (default)
scene.build(n_envs=3)   # Parallel environments for batch simulation

build() compiles all the physics kernels and initializes the simulation. Must be called after adding all entities an
d before running the simulation.


  If you need low friction (0.1) + muscle actuation, Genesis currently doesn't have a working solver comb
  ination. The options are:

  1. Use MPM + high friction (5.0) - Works for inchworming
  2. Use PBD - Stable at low friction but no muscle support
  3. Use FEM - Has muscle support but no effective ground friction

  There's no solver in Genesis that gives you both low friction stability AND muscle actuation for ground
  locomotion.
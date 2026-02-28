import numpy as np
import genesis as gs

########################## init ##########################
gs.init(seed=0, precision='32', logging_level='info')

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
        gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, 3.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    pbd_options=gs.options.PBDOptions(
        dt=4e-3,
        gravity=(0, 0, -9.8),
        particle_size=0.02,
        max_stretch_solver_iterations=10,
        max_bending_solver_iterations=5,
        lower_bound=(-3.0, -3.0, -0.5),
        upper_bound=(3.0, 3.0, 3.0),
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
    ),
)

########################## entities ##########################
# Ground plane
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(coup_friction=0.5),
)

# PBD Cloth 1: Pinned at corners (like a flag)
cloth_pinned = scene.add_entity(
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=2.0,
        pos=(0.0, 0.0, 1.5),
        euler=(0.0, 0, 0.0),
    ),
    material=gs.materials.PBD.Cloth(
        rho=500.0,
        static_friction=0.3,
        kinetic_friction=0.2,
        stretch_compliance=1e-4,
        bending_compliance=1e-3,
        stretch_relaxation=0.5,
        bending_relaxation=0.5,
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.5, 0.8, 1.0),  # Blue cloth
        vis_mode='visual',
    ),
)

# PBD Cloth 2: Free falling (visualized as particles)
cloth_free = scene.add_entity(
    morph=gs.morphs.Mesh(
        file='meshes/cloth.obj',
        scale=1.5,
        pos=(1.5, 0.0, 2.0),
        euler=(10.0, 20, 0.0),  # Rotated slightly
    ),
    material=gs.materials.PBD.Cloth(
        rho=300.0,
        static_friction=0.5,
        kinetic_friction=0.3,
        stretch_compliance=5e-4,
        bending_compliance=5e-3,
        stretch_relaxation=0.3,
        bending_relaxation=0.3,
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.3, 0.2, 1.0),  # Red cloth
        vis_mode='particle',  # Show as particles
    ),
)

# Obstacle: Box in the center
obstacle = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.25),
        size=(0.4, 0.4, 0.5),
    ),
    material=gs.materials.Rigid(coup_friction=0.4),
)

########################## add camera for recording ##########################
cam = scene.add_camera(
    pos=(3.0, 3.0, 2.0),
    lookat=(0.0, 0.0, 0.5),
    fov=40,
)

########################## build ##########################
scene.build()

# Pin the corners of cloth_pinned (like a hanging banner)
# Find particles closest to the corners
cloth_pinned.fix_particles(cloth_pinned.find_closest_particle((-1, -1, 1.5)))
cloth_pinned.fix_particles(cloth_pinned.find_closest_particle((1, -1, 1.5)))

########################## run and record video ##########################
video_interval = 5
cam.start_recording()

scene.reset()

# Let cloth settle and interact
for i in range(1000):
    scene.step()
    
    if i % video_interval == 0:
        cam.render()

########################## save video ##########################
cam.stop_recording(save_to_filename='cloth_simulation.mp4', fps=30)
print("Video saved to: cloth_simulation.mp4")
print("\nCloth 1 (Blue): Pinned at two corners, drapes over box")
print("Cloth 2 (Red): Free falling with particle visualization")

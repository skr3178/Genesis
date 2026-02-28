import numpy as np
import genesis as gs


########################## init ##########################
gs.init(seed=0, precision='32', logging_level='info')

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt = 5e-4,
        substeps=10,
        gravity=(0, 0, 0),
    ),
    viewer_options= gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=( 1.0,  1.0,  1.0),
    ),
    fem_options=gs.options.FEMOptions(
        damping=45.0,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
    ),
)

########################## entities ##########################
scene.add_entity(morph=gs.morphs.Plane())

E, nu = 3.e4, 0.45
rho = 1000.

robot_mpm = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.5, 0.2, 0.3),
        radius=0.1,
    ),
    material=gs.materials.MPM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model='neohooken',
    ),
)

robot_fem = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.5, -0.2, 0.3),
        radius=0.1,
    ),
    material=gs.materials.FEM.Muscle(
        E=E,
        nu=nu,
        rho=rho,
        model='stable_neohooken',
    ),
)

########################## add camera for recording ##########################
cam = scene.add_camera(
    pos=(1.5, 0, 0.8),
    lookat=(0.5, 0.0, 0.3),
    fov=40,
)

########################## build ##########################
scene.build()

########################## run and record video ##########################
# Set video recording interval (render every N simulation steps)
video_interval = 5  # Record every 5 steps for smoother video
cam.start_recording()

scene.reset()
for i in range(1000):
    actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])

    robot_mpm.set_actuation(actu)
    robot_fem.set_actuation(actu)
    scene.step()
    
    # Render frame for video
    if i % video_interval == 0:
        cam.render()

########################## save video ##########################
cam.stop_recording(save_to_filename='muscle_simulation.mp4', fps=30)
print("Video saved to: muscle_simulation.mp4")

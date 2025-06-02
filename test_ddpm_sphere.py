import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_cond\
    import ConditionalUnet1D, GaussianDiffusion1DConditional, \
        Trainer1DCond, Dataset1DCond, TransformerForDiffusion
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
import scipy.special as sp
import json

def grad_vec(x, y, z):
    '''
    The function to return the direction of the gradient vector
    '''
    r = np.sqrt(x*x+y*y+z*z)

    return np.array([
        x*z/(r*np.sqrt(x*x+y*y)),
        y*z/(r*np.sqrt(x*x+y*y)),
        -np.sqrt(x*x+y*y)/r
    ])
seq_length = 4
sample_num = 5400
backbone = "transformer"
if backbone == "unet":
    model = ConditionalUnet1D(
        input_dim = 3,
        local_cond_dim = 1,
        global_cond_dim = 3,
    )
else:
    model = TransformerForDiffusion(
        input_dim = 3,
        output_dim = 3,
        horizon = seq_length,
        local_cond_dim = 1,
        global_cond_dim = 3
    )

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = seq_length,
    timesteps = 1000,
    objective = 'pred_noise'
)

r = 5
angle_theta = 2 * np.pi * np.random.rand(sample_num)
angle_phi = (np.pi/2) * np.random.rand(sample_num)
x0set = r * np.cos(angle_theta) * np.sin(angle_phi)
y0set = r * np.sin(angle_theta) * np.sin(angle_phi)
z0set = -r * np.cos(angle_phi)

# The dataset for training
velocity = []
local_cond = []
global_cond = [] #Should be the current point location

# Create the window to display everything
vis1= o3d.visualization.Visualizer()
vis1.create_window()
vis1.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20))
fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
fundamental_frame.scale(8, [0, 0, 0])
vis1.add_geometry(fundamental_frame)
for i in range(sample_num):
    x0, y0, z0 = x0set[i], y0set[i], z0set[i]
    dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.01*r)
    dot.translate(np.array([x0, y0, z0]))
    dot.paint_uniform_color((1, 0, 0))

    arrow = o3d.geometry.TriangleMesh.create_arrow(\
        cylinder_radius=0.02, cone_radius=0.03, cylinder_height=0.2, cone_height=0.1)
    # Do the rotation
    v1 = np.array([\
        x0*z0/(r*np.sqrt(x0*x0+y0*y0)),\
        y0*z0/(r*np.sqrt(x0*x0+y0*y0)),\
        -np.sqrt(x0*x0+y0*y0)/r])
    v2 = np.array([0, 0, 1])
    rot_axis = np.cross(v2, v1)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(v2, v1))
    rot = R.from_rotvec(rot_angle * rot_axis).as_matrix()
    arrow.rotate(rot, [0, 0, 0])
    arrow.translate(np.array([x0, y0, z0]))
    arrow.paint_uniform_color((1, 0, 0))

    if i%100 == 0:
        vis1.add_geometry(dot)
        vis1.add_geometry(arrow)
    

    # Create the dataset
    velocity.append(np.vstack((v1, v1, v1, v1)).T)
    global_cond.append([x0, y0, z0])

velocity = torch.from_numpy(np.float32(np.array(velocity)))
assert velocity.shape == (sample_num, 3, 4)
global_cond = torch.from_numpy(np.float32(np.array(global_cond)))
assert global_cond.shape == (sample_num, 3)
local_cond = torch.zeros((sample_num, 1, 4))

vis1.run()
# Close all windows
vis1.destroy_window()

# Normalize the statistics

training_sq = (velocity + 1)/2
print(training_sq.shape)
# training_sq = 0.5*torch.rand(64, 2, 8)
# loss = diffusion(training_sq)
# loss.backward()
# Or using trainer
# 
local_label = local_cond
global_label = global_cond
dataset = Dataset1DCond(training_sq, local_label, global_label)  

trainer = Trainer1DCond(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 20000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=100000      # Force not to save the sample result
)
load_prev = False
if not load_prev:
    trainer.train()
else:
    trainer.load(1)

# after a lot of training
sample_num_check = 100
angle_theta = 2 * np.pi * np.random.rand(sample_num_check)
angle_phi = (np.pi/2) * np.random.rand(sample_num_check)
x0sample = r * np.cos(angle_theta) * np.sin(angle_phi)
y0sample = r * np.sin(angle_theta) * np.sin(angle_phi)
z0sample = -r * np.cos(angle_phi)
global_label_sample = torch.from_numpy(np.float32(np.vstack((x0sample, y0sample, z0sample)).T))
local_label_sample = local_cond[0:sample_num_check]
print(global_label_sample.shape)
print(local_label_sample.shape)
sampled_seq = diffusion.sample(batch_size = sample_num_check, local_cond = local_label_sample, global_cond = global_label_sample)
print("==Initial Check===")
print(sampled_seq.shape)
sampled_seq = sampled_seq * 2 - 1
sampled_seq = sampled_seq.cpu().numpy()
print("==Inverse the mapping and check the plot==")
# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20))
fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
fundamental_frame.scale(8, [0, 0, 0])
vis.add_geometry(fundamental_frame)
for i in range(sample_num_check):
    x0, y0, z0 = x0sample[i], y0sample[i], z0sample[i]
    dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.01*r)
    dot.translate(np.array([x0, y0, z0]))
    dot.paint_uniform_color((0, 1, 0))

    arrow = o3d.geometry.TriangleMesh.create_arrow(\
        cylinder_radius=0.02, cone_radius=0.03, cylinder_height=0.2, cone_height=0.1)
    # Do the rotation
    v1 = np.mean(sampled_seq[i], axis=-1)
    assert v1.shape[0] == 3
    v2 = np.array([0, 0, 1])
    rot_axis = np.cross(v2, v1)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(v2, v1))
    rot = R.from_rotvec(rot_angle * rot_axis).as_matrix()
    arrow.rotate(rot, [0, 0, 0])
    arrow.translate(np.array([x0, y0, z0]))
    arrow.paint_uniform_color((0, 1, 0))
    vis.add_geometry(dot)
    vis.add_geometry(arrow)
vis.run()
# Close all windows
vis.destroy_window()

# Save the statistics
trainer.save(1)

# Use the statistics to guide the point to roll to the pole
angle_theta = 2 * np.pi * np.random.rand()
angle_phi = (np.pi/2) * np.random.rand()
x0sample = r * np.cos(angle_theta) * np.sin(angle_phi)
y0sample = r * np.sin(angle_theta) * np.sin(angle_phi)
z0sample = -r * np.cos(angle_phi)

eps = 0.1
stepsize = 0.3
traj = [[x0sample, y0sample, z0sample]]
traj_vec = []
while (x0sample**2 + y0sample*y0sample + (z0sample+r)*(z0sample+r) > eps):
    global_label_sample = torch.from_numpy(\
        np.float32(np.vstack((x0sample, y0sample, z0sample)).T))
    global_label_sample = global_label_sample.view(1, -1)
    local_label_sample = local_cond[0:1]

    # Find the predicted gradient from diffusion model
    sampled_seq = diffusion.sample(batch_size = 1, \
                local_cond = local_label_sample, global_cond = global_label_sample)
    sampled_seq = sampled_seq * 2 - 1
    sampled_seq = sampled_seq.cpu().numpy()
    v1 = np.mean(sampled_seq[0], axis=-1)

    # Record the direction and update the location
    traj_vec.append([v1[0], v1[1], v1[2]])
    x0sample = x0sample + stepsize * v1[0]
    y0sample = y0sample + stepsize * v1[1]
    z0sample = z0sample + stepsize * v1[2]

    # Make sure the point is on the sphere
    norm = np.linalg.norm(np.array([x0sample, y0sample, z0sample]))
    x0sample = x0sample * r/norm
    y0sample = y0sample * r/norm
    z0sample = z0sample * r/norm

    print("==Test==")
    print([x0sample, y0sample, z0sample])
    traj.append([x0sample, y0sample, z0sample])
    
traj_vec.append([0, 0, -1])
# Create the window to display the trajectory
vis= o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20))
fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
fundamental_frame.scale(8, [0, 0, 0])
vis.add_geometry(fundamental_frame)
for i in range(len(traj)):
    x0, y0, z0 = traj[i][0], traj[i][1], traj[i][2]
    dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.01*r)
    dot.translate(np.array([x0, y0, z0]))
    dot.paint_uniform_color((0, 0, 1))

    arrow = o3d.geometry.TriangleMesh.create_arrow(\
        cylinder_radius=0.02, cone_radius=0.03, cylinder_height=0.2, cone_height=0.1)
    # Do the rotation
    v1 = np.array(traj_vec[i])
    assert v1.shape[0] == 3
    v2 = np.array([0, 0, 1])
    rot_axis = np.cross(v2, v1)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(v2, v1))
    rot = R.from_rotvec(rot_angle * rot_axis).as_matrix()
    arrow.rotate(rot, [0, 0, 0])
    arrow.translate(np.array([x0, y0, z0]))
    arrow.paint_uniform_color((0, 1, 0))
    vis.add_geometry(dot)
    vis.add_geometry(arrow)
vis.run()
# Close all windows
vis.destroy_window()
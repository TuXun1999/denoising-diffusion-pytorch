import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_cond\
    import ConditionalUnet1D, GaussianDiffusion1DConditional, \
        Trainer1DCond, Dataset1DCond
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
import scipy.special as sp
import json
def lie_algebra(gripper_pose):
    '''
    Convert a SE(3) pose to se(3)
    '''
    translation = gripper_pose[0:3, 3]
    omega = R.from_matrix(gripper_pose[:3, :3]).as_rotvec()
    x, y, z = omega
    theta = np.linalg.norm(omega)

    omega_hat = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    if theta < 1e-5: # Deal with the special case, where nearly no rotation occurs
        vw = np.hstack((translation, np.array([0, 0, 0])))
    else:
        coeff = 1 - (theta * np.cos(theta/2))/(2*np.sin(theta/2))
        V_inv = np.eye(3) - (1/2) * omega_hat + (coeff / theta ** 2) * (omega_hat@omega_hat)
        tp = V_inv@translation.flatten()
        vw = np.hstack((tp, omega))
    assert vw.shape[0] == 6
    return vw

def lie_group(vw):
    '''
    Convert a 6D vector in se(3) to a SE(3) pose
    '''
    t = vw[0:3]
    w = vw[3:]
    x, y, z = w
    omega_hat = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    A = np.vstack((np.hstack((omega_hat, t.reshape(-1, 1))), np.array([0, 0, 0, 0])))
    return scipy.linalg.expm(A)

seq_length = 32
sample_num = 6
model = ConditionalUnet1D(
    input_dim = 6,
    local_cond_dim = 1,
    global_cond_dim = 1,
)

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = seq_length,
    timesteps = 1000,
    objective = 'pred_noise'
)
option = 'circle'
# Create a circle
if option == 'circle':
    circle_whole = []
    circle_theta = []
    p = 1
    for theta in np.linspace(0, 2*np.pi, sample_num, endpoint=False):
        # Global rotation
        pose = R.from_euler('xyz', [theta, 0, 0], degrees=False).as_matrix()
        pose = np.vstack((np.hstack((pose, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
        circle_theta.append(theta)
        circle_one = []
        for l in np.linspace(0, 3, seq_length):
            # Translational part

            trans = np.array([
                [1, 0, 0, l],
                [0, 1, 0, 0],
                [0, 0, 1, np.sqrt(l) * p * (1/2)],
                [0, 0, 0, 1]
            ])
            
            pose = pose@trans
            pose_lie = lie_algebra(pose)
            circle_one.append(pose_lie)
        circle_whole.append(circle_one)
        p = p + 1

    traj_noisy = []
    for j in range(sample_num):
        for i in range(sample_num):
            traj_noisy_it = []
            circle = circle_whole[i]
            for item in circle:
                # Add some noises
                v_noisy = item[0:3] + 0 * np.random.rand(3)
                w_noisy = item[3:] + (np.pi/90) * np.random.rand(3)
                traj_noisy_it.append(np.hstack((v_noisy, w_noisy)))
            # Build one group of noisy data
            traj_noisy_it = np.array(traj_noisy_it).T

            traj_noisy.append(traj_noisy_it)
    # Combine them together
    traj_noisy_np = np.array(traj_noisy)
    assert traj_noisy_np.shape == (sample_num * sample_num, 6, seq_length)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy))

    global_label = torch.from_numpy(np.float32(np.array(circle_theta))).unsqueeze(1)
    
    global_label = torch.tile(global_label, (6, 1))

elif option == "spot":
    gripper_pose_file = open("./gripper_poses.json", 'r')
    traj_noisy_matrix = json.load(gripper_pose_file)
    traj_noisy = []
    for item in traj_noisy_matrix[5:]:
        traj_noisy_it_mat = item
        # Build one group of noisy data by sampling the waypoints on the trajectory
        traj_noisy_it_mat = np.array(traj_noisy_it_mat)
        traj_noisy_it_mat = traj_noisy_it_mat[\
            np.linspace(0, traj_noisy_it_mat.shape[0] - 1, num=seq_length).astype(int)]
        traj_noisy_it=[]
        # Convert them into lie algebra
        for i in range(len(traj_noisy_it_mat)):
            traj_noisy_it.append(lie_algebra(traj_noisy_it_mat[i]))
        traj_noisy_it = np.array(traj_noisy_it).T
        traj_noisy.append(traj_noisy_it)
    traj_noisy_np = np.array(traj_noisy)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy_np))
elif option == "spot_rel":
    # Calculate the relative transformation instead
    gripper_pose_file = open("./gripper_poses.json", 'r')
    traj_noisy_matrix = json.load(gripper_pose_file)
    traj_noisy = []
    for item in traj_noisy_matrix[5:]:
        traj_noisy_it_mat = item
        # Build one group of noisy data by sampling the waypoints on the trajectory
        traj_noisy_it_mat = np.array(traj_noisy_it_mat)
        traj_noisy_it_mat = traj_noisy_it_mat[\
            np.linspace(0, traj_noisy_it_mat.shape[0] - 1, num=seq_length).astype(int)]
        traj_noisy_it=[]
        # Convert them into lie algebra
        for i in range(len(traj_noisy_it_mat)-1):
            traj_noisy_it.append(lie_algebra(\
                np.linalg.inv(traj_noisy_it_mat[i+1])@traj_noisy_it_mat[i]))
        traj_noisy_it.append(np.array([0, 0, 0, 0, 0, 0])) #lie algebra of eye(4)
        traj_noisy_it = np.array(traj_noisy_it).T
        traj_noisy.append(traj_noisy_it)
    traj_noisy_np = np.array(traj_noisy)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy_np))
# Create the window to display everything
vis1= o3d.visualization.Visualizer()
vis1.create_window()
traj_noisy_np = np.transpose(traj_noisy_np, (0, 2, 1))
for i in range(sample_num):
    for item in traj_noisy_np[i]:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(1.5, [0, 0, 0])
        gripper_pose = lie_group(item)
        camera_frame.transform(gripper_pose)
        vis1.add_geometry(camera_frame)

vis1.run()
# Close all windows
vis1.destroy_window()


# Normalize the statistics
traj_noisy_min = torch.min(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)
traj_noisy_max = torch.max(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)

traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
training_sq = torch.nan_to_num(traj_noisy_normalize)
print(training_sq[4])
# training_sq = 0.5*torch.rand(64, 2, 8)
# loss = diffusion(training_sq)
# loss.backward()
# Or using trainer
# 
local_label = torch.from_numpy(np.float32(np.zeros((sample_num * 6, 1, seq_length))))
dataset = Dataset1DCond(training_sq, local_label, global_label)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1DCond(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 5000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=100000      # Force not to save the sample result
)
load_prev = True
if not load_prev:
    trainer.train()
else:
    trainer.load(1)

# after a lot of training
print(global_label[0].shape)
global_label_sample = torch.tile(global_label[3], (6, 1))
local_label_sample = torch.tile(local_label[3], (6, 1, 1))
print(local_label_sample.shape)
sampled_seq = diffusion.sample(batch_size = 6, local_cond = local_label_sample, global_cond = global_label_sample)
print("==Initial Check===")
print(sampled_seq[0])
chosen_seq = np.linspace(3, 3 + 6*6, num=6, endpoint=False)
print("==Inverse the mapping and check the plot==")
traj_recon = torch.mean(sampled_seq, dim = 0).cpu()
traj_noisy_max_batch_mean = torch.mean(traj_noisy_max[chosen_seq, :], dim=0)
traj_noisy_min_batch_mean = torch.mean(traj_noisy_min[chosen_seq, :], dim=0)
traj_recon = traj_recon * (\
    traj_noisy_max_batch_mean - traj_noisy_min_batch_mean) \
    + traj_noisy_min_batch_mean
traj_recon = traj_recon.numpy()
# Visualize the reconstructed grasp poses
traj_recon = traj_recon.T


# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()

gripper_pose2save = []
for item in traj_recon:
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(1.5, [0, 0, 0])
    gripper_pose = lie_group(item)
    camera_frame.transform(gripper_pose)
    camera_frame.paint_uniform_color((1, 0, 0))
    vis.add_geometry(camera_frame)

    gripper_pose2save.append(gripper_pose)
for i in range(sample_num):
    for item in traj_noisy_np[i]:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(1.5, [0, 0, 0])
        gripper_pose = lie_group(item)
        camera_frame.transform(gripper_pose)
        vis.add_geometry(camera_frame)
vis.run()
# Close all windows
vis.destroy_window()

# Save the statistics
trainer.save(1)


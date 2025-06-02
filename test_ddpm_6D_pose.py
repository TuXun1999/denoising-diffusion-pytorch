import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
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
sample_num = 1
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 6
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = seq_length,
    timesteps = 1000,
    objective = 'pred_v'
)
option = 'circle'
# Create a circle
if option == 'circle':
    trans = np.array([
        [1, 0, 0, -3],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    circle= []
    for theta in np.linspace(0, np.pi*2, seq_length):
        pose = R.from_euler('zyz', [theta, 0, 0], degrees=False).as_matrix()
        pose = np.vstack((np.hstack((pose, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
        pose = pose@trans
        pose_lie = lie_algebra(pose)
        circle.append(pose_lie)

    traj_noisy = []
    for i in range(sample_num):
        traj_noisy_it = []
        for item in circle:
            # Add some noises
            v_noisy = item[0:3] + 0.1 * np.random.rand(3)
            w_noisy = item[3:] + (np.pi/180) * np.random.rand(3)
            traj_noisy_it.append(np.hstack((v_noisy, w_noisy)))
        # Build one group of noisy data
        traj_noisy_it = np.array(traj_noisy_it).T

        traj_noisy.append(traj_noisy_it)
    # Combine them together
    traj_noisy = np.array(traj_noisy)
    traj_noisy_np = traj_noisy
    assert traj_noisy.shape == (sample_num, 6, seq_length)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy))
elif option == "straight_line":
    line = []
    for dist in np.linspace(0, 4, seq_length):
        pose = np.vstack((np.hstack((np.eye(3), np.array([[dist], [0], [0]]))), np.array([0, 0, 0, 1])))
        pose_lie = lie_algebra(pose)
        line.append(pose_lie)

    traj_noisy = []
    for i in range(sample_num):
        traj_noisy_it = []
        for item in line:
            # Add some noises
            v_noisy = item[0:3] + 0.3 * np.random.rand(3)
            w_noisy = item[3:] + (np.pi/36) * np.random.rand(3)
            traj_noisy_it.append(np.hstack((v_noisy, w_noisy)))
        # Build one group of noisy data
        traj_noisy_it = np.array(traj_noisy_it).T

        traj_noisy.append(traj_noisy_it)
    # Combine them together
    traj_noisy = np.array(traj_noisy)
    assert traj_noisy.shape == (sample_num, 6, seq_length)
    traj_noisy_np = traj_noisy
    traj_noisy = torch.from_numpy(np.float32(traj_noisy))

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
for item in traj_noisy_np[0]:
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(1, [0, 0, 0])
    gripper_pose = lie_group(item)
    camera_frame.transform(gripper_pose)
    vis1.add_geometry(camera_frame)

vis1.run()
# Close all windows
vis1.destroy_window()

# Normalize the statistics
traj_noisy_min = torch.min(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)
traj_noisy_max = torch.max(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)
print(traj_noisy[0])
print(traj_noisy_min[0])
print(traj_noisy_max[0])
traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
training_sq = torch.nan_to_num(traj_noisy_normalize)
print(training_sq[0])

# training_sq = 0.5*torch.rand(64, 2, 8)
# loss = diffusion(training_sq)
# loss.backward()
# Or using trainer
# 
dataset = Dataset1D(training_sq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 10)
sampled_seq.shape # (10, 2, 30)
print("==Initial Check===")
print(sampled_seq.shape)
print(sampled_seq[0])
print("==Inverse the mapping and check the plot==")
traj_recon = torch.mean(sampled_seq, dim = 0).cpu()
traj_noisy_max_batch_mean = torch.mean(traj_noisy_max, dim=0)
traj_noisy_min_batch_mean = torch.mean(traj_noisy_min, dim=0)
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
    camera_frame.scale(0.01, [0, 0, 0])
    gripper_pose = lie_group(item)
    camera_frame.transform(gripper_pose)
    camera_frame.paint_uniform_color((1, 0, 0))
    vis.add_geometry(camera_frame)

    gripper_pose2save.append(gripper_pose)

vis.run()
# Close all windows
vis.destroy_window()
gripper_pose2save = np.array(gripper_pose2save)
spot_pose2exec = open("./gripper_pose_sample.npy", "wb")
np.save(spot_pose2exec, gripper_pose2save)

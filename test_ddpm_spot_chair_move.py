import torch

from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditional, Trainer1DCond, Dataset1DCond
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

def SE3ToSE2(matrix):
    '''
    The function to convert an SE3 matrix into SE2 vector (x,y,theta)
    '''
    theta = R.from_matrix(matrix[:3, :3]).as_euler('zyx')[0]
    x, y = matrix[0:2, 3]
    return np.array([x, y, theta])
def SE2ToSE3(vector):
    '''
    The function to convert an SE2 vector to SE3 matrix ([x,y,theta] -> 4x4 matrix)
    '''
    theta = vector[2]
    rot = R.from_euler('z', theta, degrees=False).as_matrix()
    res = np.eye(4)
    res[:3, :3] = rot
    res[:3, 3] = np.array([vector[0], vector[1], 0])
    return res

seq_length = 16
sample_num = 1

option = 'spot_se2_rel_traj'  # Options: circle, straight_line, spot, spot_rel, spot_se2_rel_traj
if option == "spot_se2_rel_traj":
    model = ConditionalUnet1D(
        input_dim = 9,
        local_cond_dim = 1,
        global_cond_dim = 1,
    )
else:
    model = ConditionalUnet1D(
        input_dim = 6,
        local_cond_dim = 1,
        global_cond_dim = 6,
    )

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = seq_length,
    timesteps = 5000,
    objective = 'pred_noise'
)

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
    gripper_pose_file = open("./gripper_poses_move_chair_old.json", 'r')
    traj_noisy_matrix = json.load(gripper_pose_file)
    traj_noisy = []
    global_label = []

    # NOTE:
    # The body pose read from SPOT varies significantly across different
    # trials, so we need to force the height to be fixed. 
    body_height =  traj_noisy_matrix[0][0][2][3]
    for item in traj_noisy_matrix:
        traj_noisy_it_mat = item
        # Build one group of noisy data by sampling the waypoints on the trajectory
        traj_noisy_it_mat = np.array(traj_noisy_it_mat)
        traj_noisy_it_mat = traj_noisy_it_mat[\
            np.linspace(0, traj_noisy_it_mat.shape[0] - 1, num=seq_length).astype(int)]
        traj_noisy_it=[]

        # The global conditioning for this trajectory (the initial grasp pose)
        grasp_pose = traj_noisy_it_mat[0][:, 8:12]
        global_label.append(lie_algebra(grasp_pose))
        # Convert them into lie algebra
        for i in range(len(traj_noisy_it_mat)):
            gripper_pose = traj_noisy_it_mat[i][:, 0:4]
            body_pose = traj_noisy_it_mat[i][:, 4:8]
            body_pose[2, 3] = body_height  # Fix the height of the body pose
            gripper_pose = body_pose@gripper_pose

            traj_noisy_it.append(lie_algebra(gripper_pose))
        traj_noisy_it = np.array(traj_noisy_it).T
        traj_noisy.append(traj_noisy_it)
    traj_noisy_np = np.array(traj_noisy)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy_np))

elif option == "spot_rel":
    # Calculate the relative transformation instead
    gripper_pose_file = open("./gripper_poses_move_chair.json", 'r')
    traj_noisy_matrix = json.load(gripper_pose_file)
    print(len(traj_noisy_matrix))
    traj_noisy = []
    for item in traj_noisy_matrix:
        traj_noisy_it_mat = item
        # Build one group of noisy data by sampling the waypoints on the trajectory
        traj_noisy_it_mat = np.array(traj_noisy_it_mat)
        traj_noisy_it_mat = traj_noisy_it_mat[\
            np.linspace(0, traj_noisy_it_mat.shape[0] - 1, num=seq_length).astype(int)]
        traj_noisy_it=[]
        # Convert them into lie algebra
        for i in range(len(traj_noisy_it_mat)-1):
            # TODO: extract both the body and the gripper pose & have a policy learning
            gripper_pose = traj_noisy_it_mat[i][:, 0:4]
            gripper_pose_next = traj_noisy_it_mat[i+1][:, 0:4]
            traj_noisy_it.append(lie_algebra(\
                np.linalg.inv(gripper_pose_next)@gripper_pose))
        traj_noisy_it.append(np.array([0, 0, 0, 0, 0, 0])) #lie algebra of eye(4)
        traj_noisy_it = np.array(traj_noisy_it).T
        traj_noisy.append(traj_noisy_it)
    traj_noisy_np = np.array(traj_noisy)
    print(traj_noisy_np.shape)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy_np))


# Calculate the relative transformation in SE2
# Observation: SE2 pose of robot's base, SE3 pose of robot gripper in body frame
# Output: SE2 velocity of robot's base, next SE3 pose of robot gripper in body frame
# NOTE: there are two options => we need to try both
# 1. Trajectory generation
# 2. Policy 
elif option == "spot_se2_rel_traj":
    # Option 1: generate the whole trajectory by looking at the relative transformations
    gripper_pose_file = open("./gripper_poses_move_chair.json", 'r')
    traj_noisy_matrix = json.load(gripper_pose_file)
    print(len(traj_noisy_matrix))
    traj_noisy = []
    traj_noisy_vis = [] # The matrix for visualization purpose
    global_label = []
    for item in traj_noisy_matrix:
        traj_noisy_it_mat = item
        # Build one group of noisy data by sampling the waypoints on the trajectory
        traj_noisy_it_mat = np.array(traj_noisy_it_mat)
        traj_noisy_it_mat = traj_noisy_it_mat[\
            np.linspace(0, traj_noisy_it_mat.shape[0] - 1, num=seq_length).astype(int)]
        traj_noisy_it=[]
        traj_noisy_vis_it = [] # One row of the matrix for visualization purpose
        # No specific global labelling for a trajectory generation
        global_label.append([0])
        # Convert them into lie algebra
        for i in range(len(traj_noisy_it_mat)-1):
            # Extract both the body and the gripper pose & have a trajectory
            gripper_pose_body = traj_noisy_it_mat[i][:, 0:4]
            gripper_pose_body_next = traj_noisy_it_mat[i+1][:, 0:4]
            body_pose = traj_noisy_it_mat[i][:, 4:8]
            body_pose_se2 = SE3ToSE2(body_pose)
            body_pose_next = traj_noisy_it_mat[i+1][:, 4:8]
            body_pose_next_se2 = SE3ToSE2(body_pose_next)
            gripper_pose_body_rel = lie_algebra(\
                np.linalg.inv(gripper_pose_body)@gripper_pose_body_next)
            body_pose_se2_rel = body_pose_next_se2 - body_pose_se2
            traj_noisy_it.append(np.concatenate([gripper_pose_body_rel, body_pose_se2_rel]))

        # Add the visualization data
        for i in range(len(traj_noisy_it_mat)):
            gripper_pose_body = traj_noisy_it_mat[i][:, 0:4]
            body_pose = traj_noisy_it_mat[i][:, 4:8]

            traj_noisy_vis_it.append(np.concatenate(\
                [lie_algebra(gripper_pose_body), lie_algebra(body_pose)]))


            
        traj_noisy_it.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) #lie algebra of eye(4)
        traj_noisy_it = np.array(traj_noisy_it).T
        traj_noisy.append(traj_noisy_it)

        traj_noisy_vis.append(traj_noisy_vis_it)
    traj_noisy_np = np.array(traj_noisy)
    print(traj_noisy_np.shape)
    traj_noisy = torch.from_numpy(np.float32(traj_noisy_np))

    traj_noisy_vis = np.array(traj_noisy_vis)
# Create the window to display everything
vis1= o3d.visualization.Visualizer()
vis1.create_window()

traj_noisy_np = np.transpose(traj_noisy_np, (0, 2, 1))
if option == "spot":
    select_idx = 2
    for item in traj_noisy_np[select_idx]:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.1, [0, 0, 0])
        gripper_pose = lie_group(item)
        camera_frame.transform(gripper_pose)
        vis1.add_geometry(camera_frame)
elif option == "spot_se2_rel_traj":
    select_idx = 0
    for item in traj_noisy_vis[select_idx]:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.1, [0, 0, 0])
        body_frame.scale(0.1, [0, 0, 0])
        gripper_pose_body = lie_group(item[0:6])
        body_pose = lie_group(item[6:])
        gripper_pose = body_pose@gripper_pose_body
        camera_frame.transform(gripper_pose)
        body_frame.transform(body_pose)
        vis1.add_geometry(camera_frame)
        vis1.add_geometry(body_frame)
vis1.run()
# Close all windows
vis1.destroy_window()

# Normalize the statistics
traj_noisy_min = torch.min(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)
traj_noisy_max = torch.max(traj_noisy, dim=-1)[0].unsqueeze(dim=-1)
traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
training_sq = torch.nan_to_num(traj_noisy_normalize)


# training_sq = 0.5*torch.rand(64, 2, 8)
# loss = diffusion(training_sq)
# loss.backward()
# Or using trainer
# 
global_label = torch.from_numpy(np.float32(np.array(global_label)))
local_label = torch.from_numpy(np.float32(np.zeros((global_label.shape[0], 1, seq_length))))
print("==Check the shape of the labels==")
print(global_label.shape)
print(local_label.shape)
dataset = Dataset1DCond(training_sq, local_label, global_label)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1DCond(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=100000      # Force not to save the sample result
)
load_prev = False  # Set to True if you want to load the previous model
# Not Load the previous model
if not load_prev:
    trainer.train()
    # Save the statistics
    trainer.save(1)
else:
    trainer.load(1)

# after a lot of training


batch_size = 10
global_label_sample = torch.tile(global_label[select_idx], (batch_size, 1))
print("==Sample the trajectory==")

local_label_sample = torch.tile(local_label[select_idx], (batch_size, 1, 1))
print(local_label_sample.shape)
print(global_label_sample.shape)
print(global_label_sample[0])
sampled_seq = diffusion.sample(batch_size = batch_size, \
            local_cond = local_label_sample, global_cond = global_label_sample)


print("==Initial Check===")
print(sampled_seq.shape)
print(sampled_seq[0])
print("==Inverse the mapping and check the plot==")
traj_recon = torch.mean(sampled_seq, dim = 0).cpu()
traj_noisy_max_sel = traj_noisy_max[select_idx]
traj_noisy_min_sel = traj_noisy_min[select_idx]
traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel
traj_recon = traj_recon.numpy()
# Visualize the reconstructed grasp poses
traj_recon = traj_recon.T


# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()


if option == "spot":
    gripper_pose2save = []
    for item in traj_recon:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.1, [0, 0, 0])
        gripper_pose = lie_group(item)
        camera_frame.transform(gripper_pose)
        camera_frame.paint_uniform_color((1, 0, 0))
        vis.add_geometry(camera_frame)

        gripper_pose2save.append(gripper_pose)
elif option == "spot_se2_rel_traj":
    gripper_pose_body_current = lie_group(traj_noisy_vis[select_idx][0][0:6])
    body_pose_current = SE3ToSE2(lie_group(traj_noisy_vis[select_idx][0][6:]))
    pose2save = []
    for item in traj_recon:
        gripper_pose_body_tran = lie_group(item[0:6])
        gripper_pose_body_current = gripper_pose_body_current@gripper_pose_body_tran

        # The relative transformation of body pose is SE2
        body_pose_tran = item[6:]
        # Transform the body transformation from odom frame to current body frame
        body_pose_tran_body = SE2ToSE3(body_pose_tran)
        body_pose_current = body_pose_current + body_pose_tran

        # Add the visualization frames
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.1, [0, 0, 0])
        gripper_pose = SE2ToSE3(body_pose_current)@gripper_pose_body_current
        camera_frame.transform(gripper_pose)
        body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        body_frame.scale(0.1, [0, 0, 0])
        body_frame.transform(SE2ToSE3(body_pose_current))

        vis.add_geometry(camera_frame)
        vis.add_geometry(body_frame)

        
        print(body_pose_tran_body)
        pose2save.append(np.hstack((gripper_pose_body_current, body_pose_tran_body)))


vis.run()
# Close all windows
vis.destroy_window()

if option == "spot":
    gripper_pose2save = np.array(gripper_pose2save)
    spot_pose2exec = open("./gripper_pose_sample_" + option + ".npy", "wb")
    np.save(spot_pose2exec, gripper_pose2save)
elif option == "spot_se2_rel_traj":
    pose2save = np.array(pose2save)
    spot_pose2exec = open("./gripper_pose_sample_" + option + ".npy", "wb")
    np.save(spot_pose2exec, pose2save)

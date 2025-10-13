import torch

from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditional, Trainer1DCond, Dataset1DCond, TransformerForDiffusion
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
import json
import pickle

def select_closest_sample(matrix, array):
    # The function to choose the index of row in the matrix closest to array
    assert matrix.shape[1] == array.shape[0] 
    matrix_reduce = matrix - array 
    matrix_norm = torch.norm(matrix_reduce, dim=-1)
    return torch.argmin(matrix_norm)

def se2norm(array):
    # Find the norm of a se2 vector; use l2 norm temporarily
    return torch.norm(array[0:2]) + 0.36 * torch.norm(array[2])

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


obs_length = 3
obs_dim = 3
seq_length = 4
action_dim = 3

option = 'unet1d'  # Model hierarchy
'''
NOTE: Look at only object poses
'''
if option == "unet1d":
    model = ConditionalUnet1D(
        input_dim = action_dim,
        local_cond_dim = 1,
        global_cond_dim = obs_length * obs_dim,
    )
else:
   model = TransformerForDiffusion(
        input_dim = action_dim,
        output_dim = action_dim,
        horizon = seq_length,
        local_cond_dim = 1,
        global_cond_dim = 2 * obs_length * obs_dim
    )

diffusion = GaussianDiffusion1DConditional(
    model,
    seq_length = seq_length,
    timesteps = 1000,
    objective = 'pred_noise'
)

# Preprocess the data & Create the training dataset
obj_type = "banana"
filename = f"./trained_models/{obj_type}/grasp_pose2/{obj_type}.pkl"

with open(filename, 'rb') as f:
    data  = pickle.load(f)

gripper_poses = data["gripper_poses"]
object_poses = data["object_poses"]

grasp_pose = data["grasp_pose"]


traj_noisy = []
global_label = []
local_label = []
rewards = []
next_states = []
done_signals = []

for i in range(len(gripper_poses)):
    gripper_poses_one_demo = gripper_poses[i][::2] # time_steps x 3
    object_poses_one_demo = object_poses[i][::2] # time_steps x 3
    poses_one_demo = np.hstack((gripper_poses_one_demo, object_poses_one_demo)) # time_steps x 6

    # _, poses_unique_idx = np.unique(poses_one_demo, axis=0, return_index=True) # Remove the duplicate elements
    # poses_one_demo = poses_one_demo[poses_unique_idx]
    demo_length = poses_one_demo.shape[0]
    
    # for j in range(obs_length-1, demo_length-seq_length-1):
    for j in range(obs_length - 1, demo_length - 2):
        # Extract out the observations
        obs_gripper = poses_one_demo[j-obs_length+1:j+1, 0:action_dim].flatten()
        assert obs_gripper.shape[0] == obs_length * action_dim, "incorrect shape: " + str(obs_gripper.shape[0])
        obs_obj = poses_one_demo[j-obs_length+1:j+1, action_dim:].flatten()
        assert obs_obj.shape[0] == obs_length * obs_dim, "incorrect shape: " + str(obs_obj.shape[0])
        
        # Find the sequence of actions
        # action = poses_one_demo[j+1:j+seq_length+1, 0:action_dim] - poses_one_demo[j:j+seq_length, 0:action_dim]
        action = poses_one_demo[j+1, 0:action_dim] - poses_one_demo[j, 0:action_dim]
        action = np.tile(action, (seq_length, 1))
        assert action.shape[0] == seq_length and action.shape[1] == action_dim, "incorrect shape: " + str(action.shape[0])
        traj_noisy.append(action)

        '''
        NOTE: Look at both gripper pose & object pose => strange RL behaviors
        Current approach: only look at object pose
        '''
        # obs = np.concatenate([obs_gripper, obs_obj], axis=-1)
        obs = obs_obj
        global_label.append(obs)

        # Extra attributes for RL: reward & next state & done
        next_obs_gripper = poses_one_demo[j-obs_length+2:j+2, 0:action_dim].flatten()# next state
        next_obs_obj = poses_one_demo[j-obs_length+2:j+2, action_dim:].flatten()
        
        '''
        NOTE: look at both object pose & gripper pose => strange RL behaviors
        Current approach: only look at object poses
        '''
        next_obs = next_obs_obj
        # next_obs = np.concatenate([next_obs_gripper, next_obs_obj], axis=-1)
        
        next_states.append(next_obs)

        reward = -se2norm(torch.from_numpy(poses_one_demo[j, action_dim:]))# reward
        rewards.append(reward)

        done = reward > -0.05
        done_signals.append(done)

        

# Shape convention
# traj_noisy: N (total segment number) x D (action_dim, 3) x T (seq_length, number of actions to predict)
# global_label: N (total sample number) x G (global observation, which is the concatenation of the previous obs_length steps of observations)

traj_noisy = np.array(traj_noisy)
traj_noisy = np.transpose(traj_noisy, [0, 2, 1])
global_label = np.array(global_label)
next_states = np.array(next_states)
rewards = np.array(rewards)
done_signals = np.array(done_signals)
print(traj_noisy.shape)
print(global_label.shape)
# It seems that the local label is not used
local_label = np.zeros((global_label.shape[0], 1, seq_length))


traj_noisy = torch.from_numpy(np.float32(traj_noisy))
global_label = torch.from_numpy(np.float32(global_label))
local_label = torch.from_numpy(np.float32(local_label))
next_states = torch.from_numpy(np.float32(next_states))
rewards = torch.from_numpy(np.float32(rewards))
done_signals = torch.from_numpy(done_signals)

# Visualize some of the data
vis_select_idx = np.random.randint(0, len(gripper_poses)) # The index of the demo to evaluate

'''
NOTE: now, look at both object poses and gripper poses => RL behaves strangely
Current approach: only looks at object poses
'''
vis_demo_start = object_poses[vis_select_idx][0]
# vis_demo_start = np.concatenate(
#     [gripper_poses[vis_select_idx][0], object_poses[vis_select_idx][0]])
vis_demo_start = np.tile(vis_demo_start, (obs_length, ))

select_idx = select_closest_sample(global_label, vis_demo_start)# The index of the starting location in global_label


object_pose_test = object_poses[vis_select_idx]

# Create the object frames
vis1= o3d.visualization.Visualizer()
vis1.create_window()

frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
frame.scale(0.2, [0, 0, 0])



vis1.add_geometry(frame)

for object_pose in object_pose_test[::4]:
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler('xyz', [0, 0, object_pose[2]], degrees=False).as_matrix()
    pose[:3, 3] = np.array([object_pose[0], object_pose[1], 0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(0.1, [0, 0, 0])
    frame.transform(pose)
    vis1.add_geometry(frame)

vis1.run()
# Close all windows
vis1.destroy_window()

# Normalize the statistics
# traj_noisy_min = torch.min(traj_noisy, dim=-1)[0].unsqueeze(dim=-1) # NxDx1
# traj_noisy_max = torch.max(traj_noisy, dim=-1)[0].unsqueeze(dim=-1) # NxDx1

v_min = torch.min(traj_noisy[:, 0:2, :])
v_max = torch.max(traj_noisy[:, 0:2, :])
angular_v_min = torch.min(traj_noisy[:, 2, :])
angular_v_max = torch.max(traj_noisy[:, 2, :])

actions = traj_noisy.clone()

traj_noisy_min = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(dim=-1).unsqueeze(dim=0)
traj_noisy_max = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(dim=-1).unsqueeze(dim=0)
traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
training_sq = torch.nan_to_num(traj_noisy_normalize)

dataset = Dataset1DCond(training_sq, local_label, global_label)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

# Save the normalization statistics
normalization_stats = {
    "v_min": v_min,
    "v_max": v_max,
    "angular_v_min": angular_v_min,
    "angular_v_max": angular_v_max,
    "training_sq": training_sq,
    "actions": actions,
    "local_label": local_label,
    "global_label": global_label,
    "next_states": next_states,
    "rewards": rewards,
    "done_signals": done_signals
}
torch.save(normalization_stats, "normalization_stats.pth")

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
load_prev = False # Set to True if you want to load the previous model
# Not Load the previous model
if not load_prev:
    trainer.train()
    # Save the statistics
    trainer.save(1)
else:
    trainer.load(1)

# after a lot of training

frame_poses = []
batch_size_sample = 10
global_label_sample = torch.tile(global_label[select_idx], (batch_size_sample, 1)) # (2 x obs_length x obs_dim)
local_label_sample = torch.tile(local_label[select_idx], (batch_size_sample, 1)).unsqueeze(1) # This is constant

'''
NOTE: now, only consider the object pose
'''
obs_pose = global_label[select_idx][-obs_dim:]
# obs_pose = torch.concatenate(
#     [global_label[select_idx][(obs_length - 1) * action_dim : obs_length * action_dim],
#      global_label[select_idx][-obs_dim:]])
steps = 0
print("**** Visualization Check ****")
print(obs_pose)
print(global_label[select_idx])
print(traj_noisy[0])
while True:
    steps += 1
    frame_poses.append(obs_pose.numpy()) # obs_length x obs_dim x 1
    # print(local_label_sample.shape)
    # print(global_label_sample.shape)
    sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
            local_cond = local_label_sample, global_cond = global_label_sample)
    # print("==Initial Check===")
    # print(sampled_seq.shape)
    # print(sampled_seq[0])
    # print("==Inverse the mapping and check the plot==")
    traj_recon = torch.mean(sampled_seq, dim = 0)
    traj_recon = traj_recon.to(device='cpu') # DxT
    torch.cuda.synchronize()

    # traj_noisy_max_sel = traj_noisy_max[select_idx]
    # traj_noisy_min_sel = traj_noisy_min[select_idx]
    traj_noisy_max_sel = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1)
    traj_noisy_min_sel = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1)
    traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel

    # TODO: Remove the hard coding
    # last_gripper_pose = obs_pose[0:action_dim]
    # last_object_pose = obs_pose[action_dim:]

    # obs_gripper_pose = torch.zeros(3)
    # obs_object_pose = torch.zeros(3)
    # for p in range(seq_length):
    #     last_gripper_pose += traj_recon[:, p]
    #     last_object_pose += traj_recon[:, p]

    #     if p >= seq_length - obs_length:
    #         # Concatenate the observed poses
    #         obs_gripper_pose = torch.concatenate((obs_gripper_pose, last_gripper_pose))
    #         obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))

    # Ensure that the shape is correct
    # global_label_sample = torch.concatenate([obs_gripper_pose[3:], obs_object_pose[3:]])
    '''
    NOTE: The following code block looks at both gripper pose & object pose
    '''
    # last_gripper_pose = obs_pose[0:action_dim]
    # last_object_pose = obs_pose[action_dim:]
    # action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
    # last_gripper_pose += action
    # # The action is applied on the gripper, but the object needs to updated differently
    # last_object_pose[0:2] += action[0:2] # Update the position
    # last_object_pose[2] += action[2] # Update the angle
    # obs_gripper_pose = global_label_sample[0][action_dim : obs_length * action_dim]
    # obs_gripper_pose = torch.concatenate((obs_gripper_pose, last_gripper_pose))
    # obs_object_pose = global_label_sample[0][obs_length * action_dim + obs_dim :]
    # obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
    # global_label_sample = torch.concatenate([obs_gripper_pose, obs_object_pose])
    # obs_pose = torch.concatenate([last_gripper_pose, last_object_pose])
    # assert obs_pose.shape[0] == 2 * obs_dim
    # assert global_label_sample.shape[0] == 2 * obs_dim * obs_length
    
    
    '''
    NOTE: The following block only looks at the object pose
    '''
    last_object_pose = obs_pose.clone()
    action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
    
    # The action is applied on the gripper, but the object needs to updated differently
    # last_object_pose += action
    last_object_pose[0:2] += action[0:2] # Update the position
    last_object_pose[2] -= action[2] # Update the angle
    
    obs_object_pose = global_label_sample[0][obs_dim :]
    obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
    global_label_sample = obs_object_pose
    
    obs_pose = last_object_pose.clone()
    
    

    # Determine whether to exit
    if se2norm(last_object_pose) < 0.03 or steps >= 40:
        print(last_object_pose)
        break

    # Stack global_label_sample
    global_label_sample = torch.tile(global_label_sample, (batch_size_sample, 1))
    

# Visualize the reconstructed gripper & object poses


# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()


frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
frame.scale(0.2, [0, 0, 0])
vis.add_geometry(frame)

for item in frame_poses:
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(0.1, [0, 0, 0])
    object_pose = SE2ToSE3(item[-3:])
    camera_frame.transform(object_pose)
    
    vis.add_geometry(camera_frame)




vis.run()
# Close all windows
vis.destroy_window()


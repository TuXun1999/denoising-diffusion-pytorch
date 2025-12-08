from optax import add_noise
import torch
import os
from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditionalRL, Trainer1DCondRL, Dataset1DCond,\
        TransformerForDiffusion, RewardModel, NaiveCriticModel
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
import json
import pickle
import argparse

"""
Global hyperparameters
"""
obs_length = 2 # Length of history of observations
obs_dim = 3 # Dimension of the observed states
pred_length = 16 # Length of sequence of action to predict
action_length = 8
action_dim = 3 # Dimension of the predicted action
action_repeat = 5 # Number actions to predict for each state

option = 'unet1d'  # Model hierarchy

"""
Main helper functions
"""
def select_closest_sample(matrix, array):
    # The function to choose the index of row in the matrix closest to array
    assert matrix.shape[1] == array.shape[0] 
    matrix_reduce = matrix - array 
    matrix_norm = torch.norm(matrix_reduce, dim=-1)
    return torch.argmin(matrix_norm)

def se2norm(array):
    # Find the norm of a se2 vector; use l2 norm temporarily
    return torch.sqrt(array[0]**2 + array[1]**2) + 0.6 * torch.norm(array[2])

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


def create_dataset(args):
    """
    The function to preprocess the demos & Create the training dataset
    """
    ## Step 1: Read the demos
    # Hyperparameters
    obj_type = args.object
    dataset_dir = args.dataset
    grasp_pose = "grasp_pose" + str(args.grasp_pose)
    filename = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + obj_type + ".pkl"
    normalization_file = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + "training_stats.pth"
    
    with open(filename, 'rb') as f:
        data  = pickle.load(f)
        
    # Create a new dataset only if we want to train the baseline diffusion policy from scratch again
    create_new_dataset = (args.train_mode == "train_ddpm")
    if not create_new_dataset:
        # If no need to create a new training dataset for diffusion policy, 
        # read the stats and return directly
        training_stats = torch.load(normalization_file)
        training_sq = training_stats["training_sq"]
        
        # Build up the training dataset
        local_label = training_stats["local_label"]
        global_label = training_stats["global_label"]
        
        # The raw dataset without any optimality label
        training_dataset = Dataset1DCond(training_sq, local_label, global_label)
        
        return data, training_dataset, training_stats
        
    

    # Extract the gripper & object poses
    gripper_poses = data["gripper_poses"]
    object_poses = data["object_poses"]

    grasp_pose = data["grasp_pose"]


    traj_noisy = []
    global_label = []
    local_label = []
    rewards = []
    next_states = []
    
    ## Step 2: Build up the training samples
    for i in range(len(gripper_poses)):
        # Select samples with a gap between them => ensure a sufficient displacement
        gripper_poses_one_demo = gripper_poses[i] # time_steps x 3
        object_poses_one_demo = object_poses[i] # time_steps x 3
        poses_one_demo = np.hstack((gripper_poses_one_demo, object_poses_one_demo)) # time_steps x 6

        # _, poses_unique_idx = np.unique(poses_one_demo, axis=0, return_index=True) # Remove the duplicate elements
        # poses_one_demo = poses_one_demo[poses_unique_idx]
        demo_length = poses_one_demo.shape[0]
        
        for j in range(obs_length-1, demo_length-pred_length-1):
        # for j in range(obs_length - 1, demo_length - 2):
            # Extract out the observations
            obs_gripper = poses_one_demo[j-obs_length+1:j+1, 0:action_dim].flatten()
            assert obs_gripper.shape[0] == obs_length * action_dim, "incorrect shape: " + str(obs_gripper.shape[0])
            obs_obj = poses_one_demo[j-obs_length+1:j+1, action_dim:].flatten()
            assert obs_obj.shape[0] == obs_length * obs_dim, "incorrect shape: " + str(obs_obj.shape[0])
            
            # Find the sequence of actions
            action = poses_one_demo[j+1:j+pred_length+1, 0:action_dim] - poses_one_demo[j:j+pred_length, 0:action_dim]
            # action = poses_one_demo[j+1, 0:action_dim] - poses_one_demo[j, 0:action_dim]
            # action = np.tile(action, (seq_length, 1))
            # assert action.shape[0] == seq_length and action.shape[1] == action_dim, "incorrect shape: " + str(action.shape[0])
            
            # Ensure the angle is within [-pi, pi] => for the angles that pass the singularity
            action[action < -np.pi] += 2 * np.pi
            action[action > np.pi] -= 2 * np.pi
            # Append to the dataset
            traj_noisy.append(action)

            '''
            NOTE: Look at both gripper pose & object pose
            Current approach: Look at object pose
            '''
            obs = poses_one_demo[j-obs_length+1:j+1, :].flatten()
            # obs = obs_obj
            global_label.append(obs)
            
            
            # Calculate the rewards
            reward = se2norm(torch.from_numpy(poses_one_demo[j+1, action_dim:]))# reward
            state_score = se2norm(torch.from_numpy(obs_obj[-3:]))# state
            rewards.append(10 * (-reward + state_score))

            # Find the next-states
            next_obs_gripper = poses_one_demo[j-obs_length+2:j+2, 0:action_dim].flatten()# next state
            next_obs_obj = poses_one_demo[j-obs_length+2:j+2, action_dim:].flatten()
            
            '''
            NOTE: look at both object pose & gripper pose
            '''
            # next_obs = next_obs_obj
            next_obs = poses_one_demo[j-obs_length+2:j+2, :].flatten()
            next_states.append(next_obs)

    ## Step 3: Post-processing
    
    # Shape convention
    # traj_noisy: N (total segment number) x D (action_dim, 3) x T (seq_length, number of actions to predict)
    # global_label: N (total sample number) x G (global observation, which is the concatenation of the previous obs_length steps of observations)

    traj_noisy = np.array(traj_noisy)
    traj_noisy = np.transpose(traj_noisy, [0, 2, 1])
    global_label = np.array(global_label)
    print("Shape of global_label: ", global_label.shape)
    rewards = np.array(rewards)

    # The local label (optimality) is not used for traditional training
    local_label = np.zeros((global_label.shape[0], 1, pred_length))


    # To torch tensor
    traj_noisy = torch.from_numpy(np.float32(traj_noisy))
    global_label = torch.from_numpy(np.float32(global_label))
    local_label = torch.from_numpy(np.float32(local_label))
    rewards = torch.from_numpy(np.float32(rewards))
    
    v_min = torch.min(traj_noisy[:, 0:2, :])
    v_max = torch.max(traj_noisy[:, 0:2, :])
    angular_v_min = torch.min(traj_noisy[:, 2, :])
    angular_v_max = torch.max(traj_noisy[:, 2, :])
    actions = traj_noisy.clone()

    # For the diffusion model, we have to normalize the samples at first
    traj_noisy_min = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(dim=-1).unsqueeze(dim=0)
    traj_noisy_max = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(dim=-1).unsqueeze(dim=0)
    traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
    training_sq = torch.nan_to_num(traj_noisy_normalize)

    ## Step 4: Build up the training dataset & Store the statistics
    training_dataset = Dataset1DCond(training_sq, local_label, global_label)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below
    # Save the normalization statistics
    training_stats = {
        "v_min": v_min,
        "v_max": v_max,
        "angular_v_min": angular_v_min,
        "angular_v_max": angular_v_max,
        "training_sq": training_sq,
        "actions": actions,
        "local_label": local_label,
        "global_label": global_label,
    }
    
    torch.save(training_stats, normalization_file)
    
    return data, training_dataset, training_stats




# rewarding_model = RewardModel(
#     state_dim = obs_length * obs_dim + action_dim,
#     v_min = v_min, v_max = v_max, \
#     angular_v_max = angular_v_max, angular_v_min = angular_v_min,\
#     device="cuda:0")
# rewarding_model.load_dataset(states = global_label, actions = actions[:, :, 0], rewards = rewards)
# rewarding_model.train()
# rewarding_model.save_model(path="./results/rewarding_model.pth")
# rewarding_model.model.eval()




def train_ddpm(training_dataset, training_stats, args):
    """
    The main function to train a diffusion model from the existing dataset
    """
    is_wandb = args.wandb
    if is_wandb:
        import wandb
        project="object-moving-se2"
        # NOTE: the stats used when finetuning PPO
        config = {
            "RL algorithm": "PPO",
            "dataset": "banana",
            "batch_size": 32,
            "old-policy": "new",
            "epochs": 500,
        }
        wandb_logger = wandb.init(project=project, config=config)
    else:
        wandb_logger = None
    
    # Create the networks
    '''
    NOTE: Look at only object poses
    '''
    if option == "unet1d":
        model_baseline = ConditionalUnet1D(
            input_dim = action_dim,
            local_cond_dim = 1,
            global_cond_dim = 2 * obs_length * obs_dim,
        )
        model = ConditionalUnet1D(
            input_dim = action_dim,
            local_cond_dim = 1,
            global_cond_dim = 2 * obs_length * obs_dim,
        )
    else:
        model = TransformerForDiffusion(
                input_dim = action_dim,
                output_dim = action_dim,
                horizon = pred_length,
                local_cond_dim = 1,
                global_cond_dim = 2 * obs_length * obs_dim
            )

    diffusion_baseline = GaussianDiffusion1DConditionalRL(
        model_baseline,
        seq_length = pred_length,
        timesteps = 10,
        sampling_timesteps = 8, 
        ddim_sampling_eta = 1.0, # More deterministic sampling
        objective = 'pred_noise',
        combine_DDPO_MSE=True
    )
    diffusion = GaussianDiffusion1DConditionalRL(
        model,
        seq_length = pred_length,
        timesteps = 10,
        sampling_timesteps = 8,
        ddim_sampling_eta = 1.0, # More deterministic sampling
        objective = 'pred_noise',
        combine_DDPO_MSE=True
    )
    # Create the rewarding model (TODO: only used for PPO finetuning, which is under progress)
    v_min = training_stats["v_min"]
    v_max = training_stats["v_max"]
    angular_v_min = training_stats["angular_v_min"]
    angular_v_max = training_stats["angular_v_max"]
    rewarding_model = NaiveCriticModel(scale = 0.6, v_min = v_min, v_max = v_max, \
        angular_v_max = angular_v_max, angular_v_min = angular_v_min,\
        device="cuda:0")
    
    # Construct the trainer
    obj_type = args.object
    dataset_dir = args.dataset
    grasp_pose = "grasp_pose" + str(args.grasp_pose)
    results_folder = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/results"
    
    # If cfg is needed, we need to relabel the local label, and generae a new training dataset
    if args.train_mode == "train_ddpm_cfg":
        training_dataset.label_optimality(rewarding_model)
        print("Dataset length: ", training_dataset.__len__())
    trainer = Trainer1DCondRL(
        diffusion_baseline, 
        diffusion,
        dataset = training_dataset,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 5000,         # total training steps
        PPO_train_num_steps= 1000,
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        save_and_sample_every = 100000,      # Force not to save the sample result
        results_folder = results_folder,     # Read the pretrained weights
        
        # RL rewarding model
        rewarding_model = rewarding_model,
        
        # wandb logger
        wandb_logger = wandb_logger
    )

    # Mode (max-reward action selected by default)
    # train_ddpm: train the ddpm (without RL)
    # load_ddpm: load the trained ddpm (without RL)
    # train_ddpm_ppo: load the trained ddpm, and finetune it with PPO
    # load_ddpm_ppo: load the finetuned ddpm
    training_mode = args.train_mode
    if training_mode == "train_ddpm":
        trainer.train()
        trainer.save(1)
    elif training_mode == "load_ddpm":
        trainer.load(1)
    elif training_mode == "train_ddpm_cont":
        trainer.load(1)
        trainer.train()
        trainer.save(4)
    elif training_mode == "load_ddpm_cont":
        trainer.load(4)
    elif training_mode == "train_ddpm_ppo":
        trainer.load(1)
        trainer.finetune_PPO()
        trainer.save(3)
    elif training_mode == "load_ddpm_ppo":
        trainer.load(3)
    elif training_mode == "train_ddpm_cfg":
        trainer.train_cfg()
        trainer.save(2)
    elif training_mode == "load_ddpm_cfg":
        trainer.load(2)
    elif training_mode == "train_ddpm_dsrl":
        trainer.load(1)
        trainer.train_dsrl()
        obj_type = args.object
        dataset_dir = args.dataset
        grasp_pose = "grasp_pose" + str(args.grasp_pose)
        filename = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + obj_type + "_dsrl"
        trainer.td3_agent.save(filename = filename)
    elif training_mode == "load_ddpm_dsrl":
        obj_type = args.object
        dataset_dir = args.dataset
        grasp_pose = "grasp_pose" + str(args.grasp_pose)
        filename = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + obj_type + "_dsrl"
        trainer.td3_agent.load(filename = filename)
    else:
        assert False, "Unknown training mode"

    return diffusion, trainer, rewarding_model




if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--object", default = "banana")    # Type of object
    parser.add_argument("-t", "--train-mode", default = "train_ddpm") # Type of training objective
    parser.add_argument("-g", "--grasp-pose", default = 1, type = int) # Type of grasp pose
    parser.add_argument("-v", "--visualization", action="store_true", default = False) # Visualization
    parser.add_argument("-d", "--dataset", default = "./trained_models") # Where to read the demos & where to store the data
    parser.add_argument("--wandb", "-w", action ="store_true", default = False)
    
    """
	Step 0: Specifications
 	"""
    args = parser.parse_args()
    
    
    """
    Step 1: Dataset creation
    """
    
    demo_data, training_dataset, training_stats = create_dataset(args)
    
    # (Optional) Visualize one data
    if args.visualization:
        gripper_poses = demo_data["gripper_poses"]
        object_poses = demo_data["object_poses"]
        global_label = training_stats["global_label"]
        local_label = training_stats["local_label"]
        v_min = training_stats["v_min"]
        v_max = training_stats["v_max"]
        angular_v_min = training_stats["angular_v_min"]
        angular_v_max = training_stats["angular_v_max"]
        action_gt = training_stats["actions"]
        training_sq = training_stats["training_sq"]
        # Visualize some of the data (39, 27: important)
        vis_select_idx = np.random.randint(0, len(gripper_poses)) # The index of the demo to evaluate
        print(vis_select_idx)
        '''
        NOTE: now, look at both object poses and gripper poses
        '''
        # vis_demo_start = object_poses[vis_select_idx][0]
        vis_demo_start = np.concatenate(
            [gripper_poses[vis_select_idx][0], object_poses[vis_select_idx][0]])

        select_idx = select_closest_sample(global_label[:, :2 * obs_dim], vis_demo_start)# The index of the starting location in global_label
        object_pose_test = object_poses[vis_select_idx]

        # Create the object frames
        vis1= o3d.visualization.Visualizer()
        vis1.create_window()

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        frame.scale(0.2, [0, 0, 0])

        vis1.add_geometry(frame)

        for object_pose in object_pose_test[::8]:
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
    
    """
    Step 2: Train the diffusion model
    """
    diffusion, trainer, rewarding_model = train_ddpm(training_dataset, training_stats, args)
    
    """
    Step 3: (Optional) Test the the trained model
    """
    trainer.load(1)
    if args.visualization:
        frame_poses = []
        batch_size_sample = 1
        global_label_sample = torch.tile(global_label[select_idx], (batch_size_sample, 1)) # (2 x obs_length x obs_dim)
        local_label_sample = torch.tile(local_label[select_idx], (batch_size_sample, 1)).unsqueeze(1) # This is constant

        '''
        NOTE: now, consider the object pose & agent pose
        '''
        # obs_pose = global_label[select_idx][-obs_dim:]
        obs_pose = global_label[select_idx][-2 * obs_dim:].clone()
        steps = 0
        rewards_diff = []
        while True:
            steps += 1
            frame_poses.append(obs_pose.numpy()) # obs_length x obs_dim x 1
            
            # Sample from the diffusion model
            if args.train_mode != "train_ddpm_cfg" and args.train_mode != "load_ddpm_cfg":
                sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
                        local_cond = local_label_sample.to(trainer.device), global_cond = global_label_sample.to(trainer.device))
            else:
                sampled_seq = diffusion.sample_cfg(batch_size = batch_size_sample, \
                        local_cond = local_label_sample.to(trainer.device), global_cond = global_label_sample.to(trainer.device))
            # NOTE: method 1: From experiments, involving an estimated rewarding model helps to improve the performance
            # rewards = rewarding_model(global_label_sample.to(trainer.device), sampled_seq).squeeze() # (B, )
            # traj_recon = sampled_seq[torch.argmax(rewards), :]
            
           
            # NOTE: method 2: choose a random action
            traj_recon = sampled_seq[0, :]
            
            traj_recon = traj_recon.to(device='cpu') # DxT
            torch.cuda.synchronize()
            
            # Un-normalize the sampled actions
            traj_noisy_max_sel = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1)
            traj_noisy_min_sel = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1)
            traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel

            '''
            NOTE: The following code block looks at both gripper pose & object pose
            '''
            last_gripper_pose = obs_pose[0:action_dim]
            last_object_pose = obs_pose[action_dim:]
            action = torch.sum(traj_recon[:, :action_length], dim=1)

            ## Compare between the PPO-finetuned model & original
            rewards = rewarding_model(global_label_sample.to(trainer.device), sampled_seq[:, :, :action_length]).squeeze() # (B, )
            
            trainer.load(3)
            sampled_seq_new = diffusion.sample_cfg(batch_size = batch_size_sample, \
                        local_cond = local_label_sample.to(trainer.device), global_cond = global_label_sample.to(trainer.device))
            traj_recon_new = sampled_seq_new[0, :]
            
            traj_recon_new = traj_recon_new.to(device='cpu') # DxT
            torch.cuda.synchronize()
            traj_recon_new = traj_recon_new * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel
            rewards_new = rewarding_model(global_label_sample.to(trainer.device), sampled_seq_new[:, :, :action_length]).squeeze() # (B, )
            rewards_diff.append(rewards.detach().cpu().numpy() - rewards_new.detach().cpu().numpy())
            
            trainer.load(1) # Load back the original model
            
            last_gripper_pose += action
            # The action is applied on the gripper, but the object needs to updated as well
            last_object_pose[0:2] += action[0:2] # Update the position
            last_object_pose[2] += action[2] # Update the angle

            # Ensure the angle is within [-pi, pi] => avoid unbounded angles unseen in training
            if last_gripper_pose[2] > np.pi:
                last_gripper_pose[2] -= np.pi * 2
            elif last_gripper_pose[2] < -np.pi:
                last_gripper_pose[2] += np.pi * 2
                
            if last_object_pose[2] > np.pi:
                last_object_pose[2] -= np.pi * 2
            elif last_object_pose[2] < -np.pi:
                last_object_pose[2] += np.pi * 2
            
            obs_pose = torch.concatenate([last_gripper_pose, last_object_pose])
            
            # Include the latest gripper & obj pose
            global_label_sample = torch.concatenate([\
                global_label_sample[0, 2 * obs_dim :], obs_pose])
            assert obs_pose.shape[0] == 2 * obs_dim
            assert global_label_sample.shape[0] == 2 * obs_dim * obs_length
            
            '''
            NOTE: The following block only looks at the object pose
            '''
            # last_object_pose = obs_pose.clone()
            # action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
            # # action = torch.sum(traj_recon, dim=1)
            # # The action is applied on the gripper, but the object needs to updated differently
            # # last_object_pose += action
            # last_object_pose[0:2] += action[0:2] # Update the position
            # last_object_pose[2] += action[2] # Update the angle
            
            # obs_object_pose = global_label_sample[0][obs_dim :]
            # obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
            # global_label_sample = obs_object_pose
            
            # obs_pose = last_object_pose.clone()
            

            # Determine whether to exit
            if se2norm(last_object_pose) < 0.05 or steps >= 50:
                print("Final object pose: ")
                print(last_object_pose)
                print("Final timesteps: ")
                print(steps)
                print('Final reward: ')
                print(se2norm(last_object_pose))
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
        

    trainer.load(1)
    if args.visualization:
        frame_poses = []
        batch_size_sample = 1
        global_label_sample = torch.tile(global_label[select_idx], (batch_size_sample, 1)) # (2 x obs_length x obs_dim)
        local_label_sample = torch.tile(local_label[select_idx], (batch_size_sample, 1)).unsqueeze(1) # This is constant

        '''
        NOTE: now, consider the object pose & agent pose
        '''
        # obs_pose = global_label[select_idx][-obs_dim:]
        obs_pose = global_label[select_idx][-2 * obs_dim:].clone()
        steps = 0
        load_rl = False
        while True:
            steps += 1
            frame_poses.append(obs_pose.numpy()) # obs_length x obs_dim x 1
            
            # Sample from the diffusion model using dsrl
            if args.train_mode == "train_ddpm_dsrl" or args.train_mode == "load_ddpm_dsrl":
                noise_latent = trainer.td3_agent.actor(global_label_sample.to(trainer.device))
                noise_latent = noise_latent.view(-1, action_dim, pred_length)
                sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
                            local_cond = local_label_sample.to(trainer.device), 
                            global_cond = global_label_sample.to(trainer.device), 
                            w_init = noise_latent)
            else:
                sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
                        local_cond = local_label_sample.to(trainer.device), 
                        global_cond = global_label_sample.to(trainer.device))
            # Choose a random action
            traj_recon = sampled_seq[0, :]
            
            traj_recon = traj_recon.to(device='cpu') # DxT
            torch.cuda.synchronize()
            
            # Un-normalize the sampled actions
            traj_noisy_max_sel = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1)
            traj_noisy_min_sel = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1)
            traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel

            '''
            NOTE: The following code block looks at both gripper pose & object pose
            '''
            last_gripper_pose = obs_pose[0:action_dim]
            last_object_pose = obs_pose[action_dim:]
            action = torch.sum(traj_recon[:, :action_length], dim=1)
            
            last_gripper_pose += action
            # The action is applied on the gripper, but the object needs to updated as well
            last_object_pose[0:2] += action[0:2] # Update the position
            last_object_pose[2] += action[2] # Update the angle

            # Ensure the angle is within [-pi, pi] => avoid unbounded angles unseen in training
            if last_gripper_pose[2] > np.pi:
                last_gripper_pose[2] -= np.pi * 2
            elif last_gripper_pose[2] < -np.pi:
                last_gripper_pose[2] += np.pi * 2
                
            if last_object_pose[2] > np.pi:
                last_object_pose[2] -= np.pi * 2
            elif last_object_pose[2] < -np.pi:
                last_object_pose[2] += np.pi * 2
            
            obs_pose = torch.concatenate([last_gripper_pose, last_object_pose])
            
            # Include the latest gripper & obj pose
            global_label_sample = torch.concatenate([\
                global_label_sample[0, 2 * obs_dim :], obs_pose])
            assert obs_pose.shape[0] == 2 * obs_dim
            assert global_label_sample.shape[0] == 2 * obs_dim * obs_length
            
            '''
            NOTE: The following block only looks at the object pose
            '''
            # last_object_pose = obs_pose.clone()
            # action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
            # # action = torch.sum(traj_recon, dim=1)
            # # The action is applied on the gripper, but the object needs to updated differently
            # # last_object_pose += action
            # last_object_pose[0:2] += action[0:2] # Update the position
            # last_object_pose[2] += action[2] # Update the angle
            
            # obs_object_pose = global_label_sample[0][obs_dim :]
            # obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
            # global_label_sample = obs_object_pose
            
            # obs_pose = last_object_pose.clone()
            
            

            # Determine whether to exit
            if se2norm(last_object_pose) < 0.05 or steps >= 50:
                print("Final object pose: ")
                print(last_object_pose)
                print("Final timesteps: ")
                print(steps)
                print('Final reward: ')
                print(se2norm(last_object_pose))
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
    

    plt.plot(rewards_diff)
    plt.xlabel("Steps")
    plt.ylabel("Reward difference (original - PPO-finetuned)")
    plt.title("Reward difference between original diffusion model & PPO-finetuned model")
    plt.show()
    
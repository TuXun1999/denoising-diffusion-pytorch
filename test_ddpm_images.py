from denoising_diffusion_pytorch.denoising_diffusion_pytorch_image_cond import \
    ConditionalUnet2DImage, GaussianDiffusionImageConditional, TrainerImageCond,\
    DatasetImageCond
import numpy as np
import torch
import einops
sample_input = torch.from_numpy(np.load("./diffusion_demo_data/actions.npy"))
sample_image1 = torch.from_numpy(\
    np.load("./diffusion_demo_data/camera_both_front.npy")).unsqueeze(-1)
sample_image2 = torch.from_numpy(\
    np.load("./diffusion_demo_data/camera_lightning_wrist.npy")).unsqueeze(-1)
# NOTE: the provided data is not complete
# sample_image3 = torch.from_numpy(\
#     np.load("./diffusion_demo_data/camera_thunder_wrist.npy")).unsqueeze(-1)
sample_image3 = torch.from_numpy(\
    np.load("./diffusion_demo_data/camera_both_front.npy")).unsqueeze(-1)
sample_image = torch.cat([sample_image1, sample_image2, sample_image3], dim = -1)
# sample_image = sample_image1
sample_input = einops.rearrange(sample_input, '(h1 b) c -> b c h1', h1=4)
sample_image = einops.rearrange(sample_image, '(t b) h w c n -> b n c h w t', t=4)
batch_size = sample_image.shape[0]
T = sample_image.shape[-1]
H = sample_image.shape[2]
W = sample_image.shape[3]
action_dim = sample_input.shape[1]
sample_pos = torch.zeros(batch_size, 2, T)
model = ConditionalUnet2DImage(
    input_dim=action_dim,
    obs_horizon=T, 
    kernel_size=1,
    obs_number=sample_image.shape[1] # Condition on three images
)

diffusion = GaussianDiffusionImageConditional(
    model,
    seq_length = T,
    timesteps = 1000,
    objective = 'pred_noise'
)
# Normalize the actions
sample_input_max = torch.max(sample_input, 1)[0].unsqueeze(1)
sample_input_min = torch.min(sample_input, 1)[0].unsqueeze(1)
sample_input_norm = (sample_input - sample_input_min) / (sample_input_max - sample_input_min)
dataset = DatasetImageCond(sample_input_norm, sample_image, sample_pos)  

trainer = TrainerImageCond(
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
load_prev = True
trainer.train()
print("Training Complete! Exit")
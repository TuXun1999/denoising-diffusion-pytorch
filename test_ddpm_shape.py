from denoising_diffusion_pytorch.denoising_diffusion_pytorch_image_cond import \
    ConditionalUnet2DImage, GaussianDiffusionImageConditional, TrainerImageCond,\
    DatasetImageCond
import numpy as np
import torch

batch_size = 200
T = 4
C = 3
H = 26
W = 26
N = 3
action_dim = 7
sample_input = torch.zeros(batch_size, action_dim, T)
sample_image = torch.zeros(batch_size, N, C, H, W, T)
sample_pos = torch.zeros(batch_size, 2, T)
model = ConditionalUnet2DImage(
    input_dim=action_dim,
    obs_horizon=T, 
    kernel_size=1,
    obs_number=3 # Condition on three images
)

diffusion = GaussianDiffusionImageConditional(
    model,
    seq_length = T,
    timesteps = 1000,
    objective = 'pred_noise'
)
print(diffusion.forward(sample=sample_input, image=sample_image, pos=sample_pos))
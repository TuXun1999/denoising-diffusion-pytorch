import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
seq_length = 16
sample_num = 20
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 2
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = seq_length,
    timesteps = 1000,
    objective = 'pred_v'
)

# Create a circle
# theta_dense = np.linspace(0, np.pi, 50)
# circle_dense = 10 * np.vstack((np.cos(theta_dense), np.sin(theta_dense)))
# plt.plot(circle_dense[0, :], circle_dense[1, :])

# theta = np.linspace(0, np.pi, seq_length)
# circle = 10 * np.vstack((np.cos(theta), np.sin(theta)))

# Perturb the data
# noise = 2*torch.rand(sample_num, 2, seq_length) - 1 # noise range: [-1, 1]

# Create the sinusoidal curve
theta_dense = np.linspace(0, 2*np.pi, 50)
circle_dense = np.vstack((theta_dense, np.sin(theta_dense)))
print("what")
plt.plot(circle_dense[0, :], circle_dense[1, :])
print("what")
theta = np.linspace(0, 2*np.pi, seq_length)
circle = np.vstack((theta, np.sin(theta)))

# Perturb the data
noise = 2*torch.rand(sample_num, 2, seq_length) - 1 # noise range: [-1, 1]
noise = 0.2*noise

circle = torch.from_numpy(np.float32(circle))
circle_noisy = circle + noise

circle_noisy_plot = circle_noisy.numpy()
plt.scatter(circle_noisy_plot[0, 0, :], circle_noisy_plot[0, 1, :], color='g')
plt.show()


# Normalize the statistics
circle_noisy_min = torch.min(circle_noisy, dim=-1)[0].unsqueeze(dim=-1)
circle_noisy_max = torch.max(circle_noisy, dim=-1)[0].unsqueeze(dim=-1)
print(circle_noisy[0])
print(circle_noisy_min[0])
print(circle_noisy_max[0])
circle_noisy_normalize = (circle_noisy - circle_noisy_min) / (circle_noisy_max - circle_noisy_min)
training_sq = circle_noisy_normalize
print(circle_noisy_normalize[0])

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

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 2, 30)
print("==Initial Check===")
print(sampled_seq.shape)
print(sampled_seq[0])
print("==Inverse the mapping and check the plot==")
circle_recon = sampled_seq[0].cpu()
circle_noisy_max_batch_mean = torch.mean(circle_noisy_max, dim=0)
circle_noisy_min_batch_mean = torch.mean(circle_noisy_min, dim=0)
circle_recon = circle_recon * (\
    circle_noisy_max_batch_mean - circle_noisy_min_batch_mean) \
    + circle_noisy_min_batch_mean
circle_recon = circle_recon.numpy()
plt.plot(circle_dense[0, :], circle_dense[1, :])
plt.scatter(circle_recon[0, :], circle_recon[1, :], color='r')
plt.show()

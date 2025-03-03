from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_cond import \
    ConditionalUnet1D, linear_beta_schedule, cosine_beta_schedule, default, \
    extract, normalize_to_neg_one_to_one,unnormalize_to_zero_to_one,\
    ModelPrediction, partial, identity, num_to_groups, exists, \
    has_int_squareroot, cycle, Path, cpu_count, random
from denoising_diffusion_pytorch.version import __version__
import torch
from torch import nn, einsum, Tensor
import torchvision
import einops
from typing import Callable, Union
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm
# Dataset construction
class DatasetImageCond(Dataset):
    def __init__(self, tensor: Tensor, image_cond: Tensor = None, pos_cond: Tensor = None):
        super().__init__()
        self.tensor = tensor.clone()
        self.image_cond = None
        self.pos_cond = None
        if image_cond is not None:
            self.image_cond = image_cond.clone()
        if pos_cond is not None:
            self.pos_cond = pos_cond.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        a = self.tensor[idx].clone()
        b = None
        c = None
        if self.image_cond is not None:
            b = self.image_cond[idx].clone()
        if self.pos_cond is not None:
            c = self.pos_cond[idx].clone()
        return (a, b, c)
# Helper functions for the image encoder
def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
# New Noise prediction network that supports image encoding
class ConditionalUnet2DImage(nn.Module):
    def __init__(self, 
        input_dim,
        vision_feature_dim = 512, # ResNet18 has output dim of 512
        lowdim_obs_dim = 2, # agent_pos is 2 dimensional
        obs_horizon = 1,
        obs_number = 1, # Number of images to condition on
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        # Initialize the parent class
        super().__init__()

        # I. Construct the image encoder
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        self.obs_number = obs_number
        self.vision_encoder = nn.ModuleList([])
        for i in range(self.obs_number):
            vision_encoder = get_resnet('resnet18')
            # IMPORTANT!
            # replace all BatchNorm with GroupNorm to work with EMA
            # performance will tank if you forget to do this!
            vision_encoder = replace_bn_with_gn(vision_encoder)
            self.vision_encoder.append(vision_encoder)
        # observation feature has 514 dims in total per step
        obs_global_dim = obs_horizon * (vision_feature_dim * obs_number + lowdim_obs_dim)
        action_dim = input_dim

        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_global_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            image = None, pos = None, **kwargs):

        # Pass the images through the encoder
        # and then pass everything through the ConditionalUnet1D
        # to obtain the noise
        '''
        sample: (B x action_dim x T), T is the sequence length
        image: (B x N x 3 x H x W x T), N is number of images to condition on
        pos: (B x pos_dim x T)
        timestep: (B,) or int, diffusion step
        '''
        # Correct the data order for the networks
        image = einops.rearrange(image, 'b n c h w t -> n b t c h w')
        pos = einops.rearrange(pos, 'b h t -> b t h')
        
        # Feed the image to the vision encoder
        # (Parallelism?)
        image_features_all = []
        for i in range(self.obs_number):
            image_features = self.vision_encoder[i](
                image[i].flatten(end_dim=1))

            image_features = image_features.reshape(*image[i].shape[:2],-1)
            # (B,obs_horizon,T)
            image_features_all.append(image_features)

        
        # concatenate vision feature and low-dim obs
        obs_features = torch.cat(image_features_all + [pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim * obs_number)

        # Pred the noise
        noise_pred = self.noise_pred_net(
                    sample, timestep, global_cond = obs_cond)
        
        return noise_pred


class GaussianDiffusionImageConditional(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()
        self.model = model
        self.channels = 3 #self.model.out_channels
        self.self_condition = False # Brutally set up the value #self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        else:
            print("===Error===")
            print("The program only supports prediction on noise currently")
            exit(-1)
        # elif objective == 'pred_x0':
        #     loss_weight = snr
        # elif objective == 'pred_v':
        #     loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, image = None, pos = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, image, pos)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            # TODO: fix up the conversion between start & noise
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            print("===Error===")
            print("The program only supports prediction on noise currently")
            assert False
        # elif self.objective == 'pred_x0':
        #     x_start = model_output
        #     x_start = maybe_clip(x_start)
        #     pred_noise = self.predict_noise_from_start(x, t, x_start)

        # elif self.objective == 'pred_v':
        #     v = model_output
        #     x_start = self.predict_start_from_v(x, t, v)
        #     x_start = maybe_clip(x_start)
        #     pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    
    def p_mean_variance(self, x, t, image = None, pos = None, \
                        x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, image, pos, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, \
                 image = None, pos = None, \
                    x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(\
            x = x, t = batched_times, \
            image = image, pos = pos,\
            x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, image = None, pos = None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        image = image.to(device)
        pos = pos.to(device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, image, pos, self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, image, pos, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, \
                                        image, pos, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    # TODO: figure out how to add condition to sampling
    def sample(self,  batch_size = 16, image = None, pos = None):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), image, pos)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, image=None, pos=None, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        # NOTE: no need to add conditions here
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, image, pos).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, image, pos)
        if self.objective == 'pred_noise':
            target = noise
        # elif self.objective == 'pred_x0':
        #     target = x_start
        # elif self.objective == 'pred_v':
        #     v = self.predict_v(x_start, t, noise)
        #     target = v
        else:
            raise ValueError(f'unknown objective {self.objective} (only pred_noise is supported currently)')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sample, image = None, pos = None, *args, **kwargs):
        b, c, n, device, seq_length, = *sample.shape, sample.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        sample = self.normalize(sample)
        return self.p_losses(sample, t, image, pos, *args, **kwargs)

# trainer class
# TODO: modify the trainer class so as to support conditional training
class TrainerImageCond(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusionImageConditional,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data, image, pos = data
                    data = data.to(device)
                    if image is not None:
                        image = image.to(device)
                    if pos is not None:
                        pos = pos.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data, image, pos)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, image=image, pos=pos), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


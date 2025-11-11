import math
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from typing import Union, Optional, Tuple
from multiprocessing import cpu_count
import einops

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# data

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()
class Dataset1DCond(Dataset):
    def __init__(self, tensor: Tensor, local_cond: Tensor = None, global_cond: Tensor = None):
        super().__init__()
        self.tensor = tensor.clone()
        self.local_cond = None
        self.global_cond = None
        if local_cond is not None:
            self.local_cond = local_cond.clone()
        if global_cond is not None:
            self.global_cond = global_cond.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        a = self.tensor[idx].clone()
        b = None
        c = None
        if self.local_cond is not None:
            b = self.local_cond[idx].clone()
        if self.global_cond is not None:
            c = self.global_cond[idx].clone()
        return (a, b, c)
# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)
class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model 
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

# (backbone 1: residual unet)
class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.out_channels = input_dim

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        # NOTE: modify the data order (?)
        x: (B, input_dim, T)
        timestep: (B,) or int, diffusion step
        local_cond: (B, local_cond_dim, T)
        global_cond: (B,global_cond_dim)
        output: (B, input_dim, T)
        """
        # sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        # encode local features
        h_local = list()
        if local_cond is not None:
            # local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')
        return x


class TransformerForDiffusion(nn.Module):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            local_cond_dim: int = 0,
            global_cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            n_cond_layers: int = 0
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        local_obs_as_cond = local_cond_dim > 0
        global_obs_as_cond = global_cond_dim > 0
        if local_obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        if global_obs_as_cond:
            assert time_as_cond
            T_cond += 1

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if local_obs_as_cond:
            self.cond_obs_emb = nn.Linear(local_cond_dim, n_emb)
        if global_obs_as_cond:
            self.global_cond_emb = nn.Linear(global_cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and local_obs_as_cond and global_obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.local_obs_as_cond = local_obs_as_cond
        self.global_obs_as_cond = global_obs_as_cond
        self.encoder_only = encoder_only
        self.out_channels = input_dim

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond: Optional[torch.Tensor]=None, 
        global_cond: Optional[torch.Tensor]=None,
        **kwargs):
        """
        sample: (B,input_dim,T)
        timestep: (B,) or int, diffusion step
        local_cond: (B,cond_dim,T)
        global_cond: (B, global_dim)
        output: (B,input_dim,T)
        """
        # 0. correct the order
        sample = einops.rearrange(sample, 'b h t -> b t h')
        local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
        global_cond = global_cond.unsqueeze(-1) # Shape: (B, global_dim, 1)
        global_cond = einops.rearrange(global_cond, 'b h t -> b t h')
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.local_obs_as_cond and self.global_obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(local_cond)
                # (B,To,n_emb)
                global_obs_emb = self.global_cond_emb(global_cond)
                # (B, 1, n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb, global_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        # Correct the order
        x = einops.rearrange(x, 'b h t -> b t h')
        return x
# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1DConditionalRL(Module):
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
        auto_normalize = True,
        combine_DDPO_MSE = False,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.out_channels
        self.self_condition = False # Brutally set up the value #self.model.self_condition
        self.combine_DDPO_MSE = combine_DDPO_MSE
        self.mse_loss_coef = 0.5 if combine_DDPO_MSE else 0.0
        self.clip_range = 2e-1
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

    def model_predictions(self, x, t, local_cond = None, global_cond = None, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, local_cond, global_cond)
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

    # TODO: understand the codes here
    def p_mean_variance(self, x, t, local_cond = None, global_cond = None, \
                        x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, local_cond, global_cond, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)
        # Formula (6) in https://arxiv.org/pdf/2006.11239
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, \
                 local_cond = None, global_cond = None, \
                    x_self_cond = None, clip_denoised = True, img_prev = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(\
            x = x, t = batched_times, \
            local_cond = local_cond, global_cond = global_cond,\
            x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        # TODO: further understand the mathematics here & return log scores as well
        if prev_img is None:
            prev_img = model_mean + sigma * noise
        pred_img_prev = prev_img.clone()
        
        # Calculate the log probabilities
        log_prob = (
            -((prev_img - model_mean) ** 2) /  (2 * (sigma**2))
            - math.log(sigma)
            - math.log(math.sqrt(2 * math.pi))
        )
        # Mean over each individual "pixel" (assume indpendence between "pixels")
        log_prob = torch.mean(log_prob, axis=tuple(range(1, log_prob.ndim)))
        return pred_img_prev, x_start, log_prob
    """
    Modify the two sampling functions so that they can return log probability scores 
    as well
    """
    @torch.no_grad()
    def p_sample_loop(self, shape, local_cond = None, global_cond = None, verbose = False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        local_cond = local_cond.to(device)
        global_cond = global_cond.to(device)
        x_start = None
        if verbose is True:
            ts_lst = []
            img_lst = []
            img_next_lst = []
            log_probs_lst = []
        #for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            if verbose is True:
                img_lst.append(img)
            img, x_start, log_prob = self.p_sample(img, t, local_cond, global_cond, self_cond)
            
            if verbose is True:
                img_next_lst.append(img)
                ts_lst.append(t)
                log_probs_lst.append(log_prob)

        img = self.unnormalize(img)
        
        if verbose is True:
            # Transform the shapes:
            # img_lst, img_next_lst into shape (B, ts, D, T)
            # ts_lst into shape (B, ts)
            img_lst = torch.stack(img_lst).to(device).permute(1, 0, 2, 3)
            img_next_lst = torch.stack(img_next_lst).to(device).permute(1, 0, 2, 3)
            log_probs_lst = torch.stack(log_probs_lst).to(device).permute(1, 0)
            
            # Already of the length sampling_timesteps
            ts_lst = torch.broadcast_to(ts_lst, (img_lst.shape[0], ts_lst.shape[0]))
            # Unnormalize img_lst & img_next_lst
            img_lst = self.unnormalize(img_lst)
            img_next_lst = self.unnormalize(img_next_lst)
        
            return img, img_lst, img_next_lst, ts_lst, log_probs_lst
        return img
    @torch.no_grad()
    def ddim_sample_step(self, eta, time, time_next, x_start, pred_noise, img_prev = None):
        """
        Function to do one step of sampling in DDIM
        """
        # Check formula (12) from https://arxiv.org/pdf/2010.02502.pdf (DDIM)
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        
        # Predict the mean of the previous sample
        img_prev_mean = x_start * alpha_next.sqrt() + c * pred_noise
       
        
        # Follow the formula
        if img_prev is None:
            noise = torch.randn_like(x_start)
            img_prev =  img_prev_mean + sigma * noise

        # Understand the mathematics & finalize the function to return the list of log_prob as well
        log_prob = (
            -((img_prev - img_prev_mean) ** 2) / (2 * (sigma**2))
            - math.log(sigma)
            - math.log(math.sqrt(2 * math.pi))
        )
        # Mean over each individual "pixel" (assume indpendence between "pixels")
        log_prob = torch.mean(log_prob, axis=tuple(range(1, log_prob.ndim)))
        
        img = img_prev.clone()
        return img, x_start, log_prob
    
    @torch.no_grad()
    def ddim_sample(self, shape, local_cond, global_cond, clip_denoised = True, verbose = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # The sample to genereta
        img = torch.randn(shape, device = device)

        # The predicted initial sample
        x_start = None
        if verbose is True:
            ts_lst = []
            img_lst = []
            img_next_lst = []
            log_probs_lst = []
        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            if verbose is True:
                ts_lst.append(torch.tensor(time))
                img_lst.append(img)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, \
                                        local_cond, global_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0: 
                # At the final stage, the predicted sample is the inital sample
                img = x_start
                continue
            img, _, log_prob = self.ddim_sample_step(eta, time, time_next, x_start, pred_noise, img_prev = None)
            if verbose is True:
                img_next_lst.append(img)
                log_probs_lst.append(log_prob)
        # Parameters to return: latents, latents_lst, next_latents_lst, ts_lst, log_probs_lst, images

        # The final generated sample (also the first initial sample)
        img = self.unnormalize(img) # (B, D, T)
        
        if verbose is True:
            # Transform the shapes:
            # img_lst, img_next_lst into shape (B, ts, D, T)
            # ts_lst into shape (B, ts)
            img_lst = torch.stack(img_lst).to(device).permute(1, 0, 2, 3)
            img_next_lst = torch.stack(img_next_lst).to(device).permute(1, 0, 2, 3)
            log_probs_lst = torch.stack(log_probs_lst).to(device).permute(1, 0)
            ts_lst = torch.stack(ts_lst).to(device)
            # Already of the length sampling_timesteps
            ts_lst = torch.broadcast_to(ts_lst, (img_lst.shape[0], ts_lst.shape[0]))
            # Unnormalize img_lst & img_next_lst
            img_lst = self.unnormalize(img_lst)
            img_next_lst = self.unnormalize(img_next_lst)
        
            return img, img_lst, img_next_lst, ts_lst, log_probs_lst
        return img

    @torch.no_grad()
    def sample(self,  batch_size = 16, local_cond = None, global_cond = None):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), local_cond, global_cond)
    
    
    @torch.no_grad()
    def sample_verbose(self, batch_size = 16, local_cond = None, global_cond = None):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), local_cond, global_cond, verbose=True)
    
    
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

    def p_losses(self, x_start, t, local_cond=None, global_cond=None, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        # NOTE: no need to add conditions here (sample the random timesteps)
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, local_cond, global_cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, local_cond, global_cond)

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
    
    
    def forward(self, img, local_cond = None, global_cond = None, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, local_cond, global_cond, *args, **kwargs)
    
    def forward_ppo_loss(self, img, local_cond = None, global_cond = None, *args, **kwargs):
        if img is not None:
            b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
            assert n == seq_length, f'seq length must be {seq_length}'
            img = self.normalize(img)
        else:
            b = global_cond.shape[0]
            device = global_cond.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        

        img_lst = kwargs["img_lst"] # Shape: (B, ts, D, T)
        img_next_lst = kwargs["img_next_lst"] # Shape: (B, ts, D, T)
        ts_lst = kwargs["ts_lst"] # Shape: (B, ts)
        log_probs_lst = kwargs["log_probs_lst"] # Shape: (B, ts)

        advantages = kwargs["advantages"] # Shape: (B, )
        
        logger = kwargs["logger"]
        
        train_steps = 5
        total_loss = 0
        # for i in random.sample(range(img_next_lst.shape[1]), k=train_steps):  # img_lst.shape: torch.Size([6, 10, 4, 64, 64])
        for i in range(img_next_lst.shape[1]):
            img_i, img_next_i, t_i = img_lst[:, i], img_next_lst[:, i], ts_lst[:, i]
            # Normalize the inputs
            img_i = self.normalize(img_i)
            img_next_i = self.normalize(img_next_i)

            # Prepare the inputs (TODO: classifier-free guidance?)
            img_old_model_noisy = img_i
            t_input = t_i
            if self.combine_DDPO_MSE is True:
                # Get the ground-truth image
                img_gt = img

                # Generate noisy samples from the ground-truth image
                noise = default(noise, lambda: torch.randn_like(img_gt))

                # noise sample
                # NOTE: no need to add conditions here
                # We are adding noise to the original sample for a noisy sample
                img_gt_noisy = self.q_sample(x_start = img_gt, t = t_input, noise = noise)


                if self.self_condition:
                    raise NotImplementedError("DDPO with self-conditioning is not implemented yet.")
                
                
                # Get the target for loss depending on the prediction type
                if self.objective == 'pred_noise':
                    target = noise
                # elif self.objective == 'pred_x0':
                #     target = x_start
                # elif self.objective == 'pred_v':
                #     v = self.predict_v(x_start, t, noise)
                #     target = v
                else:
                    raise ValueError(f'unknown objective {self.objective} (only pred_noise is supported currently)')

                
                # Predict the noise residual and compute loss
                noise_pred_old_model, x_start, *_ = self.model_predictions(img_old_model_noisy, t_input, \
                        local_cond, global_cond, None, clip_x_start = False)
                noise_pred_gt_img, _, *_ = self.model_predictions(img_gt_noisy, t_input, \
                        local_cond, global_cond, None, clip_x_start = False)
            else:
                # Predict the noise residual and compute loss based on old model directly
                noise_pred_old_model = self.model(img_old_model_noisy, t_input, local_cond, global_cond)
                x_start = self.predict_start_from_noise(img_old_model_noisy, t_input, noise_pred_old_model)
                
          
            if self.is_ddim_sampling is True and t_input[0] > 0:
                
                # _, _, log_prob = self.ddim_sample_step(\
                #     self.ddim_sampling_eta, 
                #     t_input[0], t_input[0]-1, 
                #     x_start, noise_pred_old_model, 
                #     img_prev=img_next_i)
                """
                The function has no torch_grad support. So, I replicate its codes here
                to enable the torch_grad support
                """
                alpha = self.alphas_cumprod[t_input[0]]
                alpha_next = self.alphas_cumprod[t_input[0]-1]

                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                
                # Predict the mean of the previous sample
                img_prev_mean = x_start * alpha_next.sqrt() + c * noise_pred_old_model
 
                # Understand the mathematics & finalize the function to return the list of log_prob as well
                log_prob = (
                    -((img_next_i - img_prev_mean) ** 2) / (2 * (sigma**2))
                    - math.log(sigma)
                    - math.log(math.sqrt(2 * math.pi))
                )
                # Mean over each individual "pixel" (assume indpendence between "pixels")
                log_prob = torch.mean(log_prob, axis=tuple(range(1, log_prob.ndim)))
            else:
                # _, _, log_prob = self.p_sample(\
                #     img_old_model_noisy, 
                #     t_input, 
                #     local_cond, global_cond, 
                #     x_self_cond=None, 
                #     clip_denoised=False, 
                #     img_prev=img_next_i)
                """
                Move the block of code out of a no_grad function
                """
                x = img_old_model_noisy
                b, *_, device = *x.shape, x.device
                batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
                model_mean, _, model_log_variance, x_start = self.p_mean_variance(\
                    x = x, t = batched_times, \
                    local_cond = local_cond, global_cond = global_cond,\
                    x_self_cond = None, clip_denoised = False)
                noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
                sigma = (0.5 * model_log_variance).exp()
                
                
                # Calculate the log probabilities
                log_prob = (
                    -((img_next_i - model_mean) ** 2) /  (2 * (sigma**2))
                    - math.log(sigma)
                    - math.log(math.sqrt(2 * math.pi))
                )
                # Mean over each individual "pixel" (assume indpendence between "pixels")
                log_prob = torch.mean(log_prob, axis=tuple(range(1, log_prob.ndim)))

    
            ratio_ddpo = torch.exp(log_prob - log_probs_lst[:, i])


            # Compute PPOClip loss
            unclipped_loss_ddpo = -advantages * ratio_ddpo
            clipped_loss_ddpo = -advantages * torch.clamp(ratio_ddpo, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_ddpo = torch.sum(torch.max(unclipped_loss_ddpo, clipped_loss_ddpo))

            if self.combine_DDPO_MSE:
                loss = F.mse_loss(noise_pred_gt_img, target, reduction = 'none')
                loss = reduce(loss, 'b ... -> b', 'mean')

                loss = loss * extract(self.loss_weight, t, loss.shape)
                loss_mse_reconstruction = loss.mean()

                total_loss += loss_mse_reconstruction * self.mse_loss_coef + loss_ddpo
            else:
                total_loss += loss_ddpo
        
        return total_loss
    
    def forward_ppod_loss(self, local_cond = None, global_cond = None, *args, **kwargs):

        # Extract out the necessary parameters
        img_w_lst = kwargs["img_w_lst"] # Shape: (B, ts, D, T)
        img_l_lst = kwargs["img_l_lst"] # Shape: (B, ts, D, T)
        img_next_w_lst = kwargs["img_next_w_lst"] # Shape: (B, ts, D, T)
        img_next_l_lst = kwargs["img_next_l_lst"] # Shape: (B, ts, D, T)
        ts_lst = kwargs["ts_lst"] # Shape: (B, ts)
        log_probs_w_lst = kwargs["log_probs_w_lst"] # Shape: (B, ts)
        log_probs_l_lst = kwargs["log_probs_l_lst"] # Shape: (B, ts)
        
        logger = kwargs["logger"]
        
        train_steps = 70
        total_loss = 0
        # for i in random.sample(range(img_next_w_lst.shape[1]), k=train_steps):  # img_lst.shape: torch.Size([6, 10, 4, 64, 64])
        for i in range(img_next_w_lst.shape[1]):
            img_w_i, img_next_w_i, t_i = img_w_lst[:, i], img_next_w_lst[:, i], ts_lst[:, i]
            img_l_i, img_next_l_i = img_l_lst[:, i], img_next_l_lst[:, i]
            # Normalize the inputs
            img_w_i = self.normalize(img_w_i)
            img_next_w_i = self.normalize(img_next_w_i)
            img_l_i = self.normalize(img_l_i)
            img_next_l_i = self.normalize(img_next_l_i)

            # Prepare the inputs (TODO: classifier-free guidance?)
            img_ref_w_noisy = img_w_i
            img_ref_l_noisy = img_l_i
            t_input = t_i
            

            # Predict the probailities of generating the samples from referece model for the new model
            noise_pred_ref_w = self.model(img_ref_w_noisy, t_input, local_cond, global_cond)
            x_start_ref_w = self.predict_start_from_noise(img_ref_w_noisy, t_input, noise_pred_ref_w)
            noise_pred_ref_l = self.model(img_ref_l_noisy, t_input, local_cond, global_cond)
            x_start_ref_l = self.predict_start_from_noise(img_ref_l_noisy, t_input, noise_pred_ref_l)

            if self.is_ddim_sampling is True and t_input[0] > 0:
                
                # _, _, log_prob = self.ddim_sample_step(\
                #     self.ddim_sampling_eta, 
                #     t_input[0], t_input[0]-1, 
                #     x_start, noise_pred_old_model, 
                #     img_prev=img_next_i)
                """
                The function has no torch_grad support. So, I replicate its codes here
                to enable the torch_grad support
                """
                alpha = self.alphas_cumprod[t_input[0]]
                alpha_next = self.alphas_cumprod[t_input[0]-1]

                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                
                # Predict the mean of the previous sample
                img_prev_mean_w = x_start_ref_w * alpha_next.sqrt() + c * noise_pred_ref_w
                img_prev_mean_l = x_start_ref_l * alpha_next.sqrt() + c * noise_pred_ref_l

                # Understand the mathematics & finalize the function to return the list of log_prob as well
                log_prob_w = (
                    -((img_next_w_i - img_prev_mean_w) ** 2) / (2 * (sigma**2))
                    - math.log(sigma)
                    - math.log(math.sqrt(2 * math.pi))
                )
                # Mean over each individual "pixel" (assume indpendence between "pixels")
                log_prob_w = torch.mean(log_prob_w, axis=tuple(range(1, log_prob_w.ndim)))
                
                # Repeat the for the "lose" sample
                log_prob_l = (
                    -((img_next_l_i - img_prev_mean_l) ** 2) / (2 * (sigma**2))
                    - math.log(sigma)
                    - math.log(math.sqrt(2 * math.pi))
                )
                # Mean over each individual "pixel" (assume indpendence between "pixels")
                log_prob_l = torch.mean(log_prob_l, axis=tuple(range(1, log_prob_l.ndim)))
            else:
                print("PPO-D only supports DDIM sampling currently.")
                raise NotImplementedError("PPO-D only supports DDIM sampling currently.")

            ratio_ddpo_w = log_prob_w - log_probs_w_lst[:, i]
            ratio_ddpo_l = log_prob_l - log_probs_l_lst[:, i]
            eta = 0.5
            ratio_ddpo = eta * torch.clip(ratio_ddpo_w, min = -10, max = 10) - torch.clip(ratio_ddpo_l, min = -10, max = 10)

            # Compute PPOD loss
            loss_ddpo = ratio_ddpo

            
            total_loss += loss_ddpo

        return -torch.log(torch.sigmoid(total_loss))

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size: int | list, output_size):
        super(SimpleMLP, self).__init__()
        self.model = nn.ModuleList([])
        if isinstance(hidden_size, list):
            self.model.append(nn.Linear(input_size, hidden_size[0]))
            self.model.append(nn.ReLU())
            if len(hidden_size) > 2:
                for i in range(1, len(hidden_size)):
                    self.model.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
                    self.model.append(nn.ReLU())
            self.model.append(nn.Linear(hidden_size[-1], output_size))
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
class RewardModel(nn.Module):
    # The function to give the rewarding on the given observation
    def __init__(self, scale=0.4, state_dim = 4, **kwargs):
        super().__init__()
        self.scale = scale
        
        
        device = kwargs["device"]
        self.device = device
        
        # 3. Instantiate Model, Loss Function, and Optimizer
        self.model = SimpleMLP(state_dim, [32, 32], 1).to(device)
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        # 4. Training Loop
        self.num_epochs = 500
        # Record the normalization statistics for action
        v_max = kwargs["v_max"]
        angular_v_max = kwargs["angular_v_max"]
        v_min = kwargs["v_min"]
        angular_v_min = kwargs["angular_v_min"]

        self.action_upper = torch.tensor([v_max, v_max, angular_v_max]).to(device)
        self.action_lower = torch.tensor([v_min, v_min, angular_v_min]).to(device)
    
    def load_dataset(self, **kwargs):
        """
        The object's own function to train the models
        """
        # Extract out the states & rewards
        states = kwargs["states"].to(self.device)
        actions = kwargs["actions"].to(self.device)
        rewards = kwargs["rewards"].to(self.device)
        self.state_dim = states.shape[-1]  # x, y, theta
        X = torch.hstack((states, actions))
        y = rewards
        # Create a TensorDataset and DataLoader
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
    def train(self):
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in self.dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(-1), batch_y)

                # Backward and optimize
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()        # Compute gradients
                self.optimizer.step()       # Update model parameters
    def save_model(self, path = "./results/rewarding_model.pth"):
        torch.save(self.model.state_dict(), path)
    def load_model(self, path = "./results/rewarding_model.pth"):
        self.model.load_state_dict(torch.load(path, weights_only=False))
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        
        # action shape: (B, D, T)
        # state_shape: (B, state_dim)
        # Extract x, y, theta from the global_cond
        action = action.mean(dim=-1).to(self.device)  # (B, D)
        action = action * (self.action_upper - self.action_lower) + self.action_lower
        rewards = self.model(torch.hstack((state, action)))
        return rewards
    
class NaiveCriticModel(nn.Module):
    # The function to give the rewarding on the given observation
    def __init__(self, scale=0.4, **kwargs):
        super().__init__()
        self.scale = scale
        self.state_dim = 3  # x, y, theta
        
        v_max = kwargs["v_max"]
        angular_v_max = kwargs["angular_v_max"]
        v_min = kwargs["v_min"]
        angular_v_min = kwargs["angular_v_min"]
        device = kwargs["device"]
        self.action_upper = torch.tensor([v_max, v_max, angular_v_max]).to(device)
        self.action_lower = torch.tensor([v_min, v_min, angular_v_min]).to(device)
        self.device = kwargs["device"]


    def forward(self,state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Calculate the reward directly at x^2 + y^2 + 0.4*theta^2
        # action shape: (B, D, T)
        # state_shape: (B, state_dim)
        # Extract x, y, theta from the global_cond
        
        x = state[:, -3]
        y = state[:, -2]
        theta = state[:, -1]
        state_values = torch.sqrt(x**2 + y**2) + self.scale * torch.norm(theta)  # (B, )
        # At first, decode the action into their real-values
        action = action.mean(dim=-1).to(self.device)  # (B, D)
        action = action * (self.action_upper - self.action_lower) + self.action_lower

        # Analytical formula to estimate the reward
        x = x + action[:, 0]
        y = y + action[:, 1]
        theta = theta + action[:, 2]
        rewards = torch.sqrt(x**2 + y**2) + self.scale * torch.norm(theta)  # (B, )
        rewards = -(rewards - state_values)
        rewards = rewards.unsqueeze(-1)  # (B, 1)
        # NOTE: Encourage larger rewards
        rewards = 1000 * rewards
        return rewards
    
# trainer class
class Trainer1DCondRL(object):
    def __init__(
        self,
        # Diffusion model
        diffusion_model_baseline: GaussianDiffusion1DConditionalRL,
        diffusion_model: GaussianDiffusion1DConditionalRL,
        
        # Dataset
        dataset: Dataset,
        *,
        # ppo_dataset = None,
        # Training hyperparameters
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        PPO_train_num_steps = 100000,
        
        # EMA updates
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        
        # RL rewarding model
        rewarding_model = None,
        
        # Wandb logger
        wandb_logger = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model_baseline = diffusion_model_baseline
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.reward_sample_ppo_every = 10  # after how many PPO steps to resample rewards

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.PPO_train_num_steps = PPO_train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        self.batch_size = train_batch_size
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # ppo_dataset = ppo_dataset
        ppo_dl = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = cpu_count())
        ppo_dl = self.accelerator.prepare(ppo_dl)
        self.ppo_dl = cycle(ppo_dl)
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        # PPO is only used for finetuning
        self.opt_ppo = Adam(diffusion_model.parameters(), lr = 0.1 * train_lr,  betas = adam_betas)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.PPO_step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model_baseline, self.model, self.opt, self.opt_ppo= \
            self.accelerator.prepare(self.model_baseline, self.model, self.opt, self.opt_ppo)

        # RL rewarding finetuning
        self.rewarding_model = rewarding_model
        
        # Wandb logger
        self.wandb_logger = wandb_logger
        
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
            'opt_ppo': self.opt_ppo.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model_baseline = self.accelerator.unwrap_model(self.model_baseline)
        model_baseline.load_state_dict(data['model'])
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = 0 #data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    
    def train(self):
        """
        The function to train the baseline diffusion model following traditional 
        noise reconstruction loss
        """
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data, local_cond, global_cond = data
                    data = data.to(device)
                    if local_cond is not None:
                        local_cond = local_cond.to(device)
                    if global_cond is not None:
                        global_cond = global_cond.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data, local_cond, global_cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'MSE loss for Noise Estimation: {total_loss:.4f}')

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
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, local_cond=local_cond, global_cond=global_cond), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
    
    def finetune_PPO(self):
        """
        Finetune the diffusion model use PPO
        """
        # Freeze the baseline diffusion model
        self.model_baseline.requires_grad_(False)
        accelerator = self.accelerator
        device = accelerator.device

        """
        The crazy idea: test only one data sample!
        If it still failed, then probably PPO loss is not calculated correctly?
        """
        # data = next(self.ppo_dl)
        # data, local_cond, global_cond = data
        # data = data.to(device)
        with tqdm(initial = self.PPO_step, total = self.PPO_train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.PPO_step < self.PPO_train_num_steps:
                self.model_baseline.eval()
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data, local_cond, global_cond = data
                    data = data.to(device)

                    # Stack the global_cond
                    with self.accelerator.autocast():
                        with torch.no_grad():
                            # Stack the global_cond (from batch_size x state_dim into batch_size * sample_batch_size x state dim)
                            batch_size_sample = 32
                            local_cond_sample = torch.repeat_interleave(local_cond, repeats=batch_size_sample, dim=0) # For the local_cond, only to stack them
                            global_cond_sample = torch.repeat_interleave(global_cond, repeats=batch_size_sample, dim=0)
                            data_sample = torch.repeat_interleave(data, repeats=batch_size_sample, dim=0)
                            # Sample a batch of data using the baseline policy
                            img, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model_baseline.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores = self.rewarding_model(global_cond_sample, img)
                            
                            # Size of reward_scores: (bs, 1) => (bs, )
                            reward_scores = torch.Tensor(reward_scores).squeeze(-1).to(self.accelerator.device)
                            # avg_reward_step = all_rewards_valid.mean().item() /self.gradient_accumulate_every
                            
                            # (batch_size*batch_size_sample, ) => (batch_size, batch_size_sample, )
                            reward_scores = reward_scores.view(global_cond.shape[0], batch_size_sample)
                            
                            all_rewards = self.accelerator.gather(reward_scores)
                            # add a small offset 1e-7 to avoid denominator being 0
                            advantages = (reward_scores - torch.mean(all_rewards, dim=1, keepdim=True)) / (
                                    torch.std(all_rewards, dim=1, keepdim=True) + 1e-7)
                            # Convert advantages back to (batch_size*batch_size_sample, )
                            advantages = advantages.view(-1, 1)
                           
                            # Clip the advantages
                            adv_clip_max = 10.0
                            advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
                        # Obtain the DD-PPO loss
                        loss = self.model.forward_ppo_loss(data_sample, local_cond_sample, global_cond_sample,
                                          img_lst=img_lst,
                                          img_next_lst=img_next_lst,
                                          ts_lst=ts_lst,
                                          log_probs_lst=log_probs_lst,
                                          advantages=advantages.squeeze(-1),
                                          logger = self.wandb_logger)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                pbar.set_description(f'PPO loss: {total_loss:.4f}')
                if self.wandb_logger is not None:
                    self.wandb_logger.log({"PPO Objective": total_loss})
                    

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt_ppo.step()
                self.opt_ppo.zero_grad()

                accelerator.wait_for_everyone()

                self.PPO_step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.PPO_step != 0 and self.PPO_step % self.reward_sample_ppo_every == 0 \
                        and self.wandb_logger is not None:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            data = next(self.dl)
                            data, local_cond, global_cond = data
                            data = data.to(device)
                            batch_size_sample = 32
                            local_cond_sample = torch.repeat_interleave(local_cond, repeats=batch_size_sample, dim=0) # For the local_cond, only to stack them
                            global_cond_sample = torch.repeat_interleave(global_cond, repeats=batch_size_sample, dim=0)
                            data_sample = torch.repeat_interleave(data, repeats=batch_size_sample, dim=0)
                            # Sample a batch of data using the latest policy
                            img_new, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores_new = self.rewarding_model(global_cond_sample, img_new)
                            
                            
                            reward_new = reward_scores_new.mean().item()
                             # Sample a batch of data using the baseline policy
                            img, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model_baseline.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores = self.rewarding_model(global_cond_sample, img)
                            self.wandb_logger.log({"PPO Sampled Reward (Mean, baseline - new)": reward_scores.mean().item() - reward_new})
                            self.wandb_logger.log({"PPO Sampled Reward (baseline)": reward_scores.mean().item()})
                            self.wandb_logger.log({"PPO Sampled Reward (new)": reward_new})
                           
                            self.wandb_logger.log({"PPO Sampled Reward (Std, new)": reward_scores_new.std().item()})

                pbar.update(1)

        accelerator.print('PPO training complete')
        self.PPO_step = 0
        
    def finetune_PPOD(self):
        """
        Finetune the diffusion model use PPO
        """
        # Freeze the baseline diffusion model
        self.model_baseline.requires_grad_(False)
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.PPO_step, total = self.PPO_train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.PPO_step < self.PPO_train_num_steps:
                self.model_baseline.eval()
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data, local_cond, global_cond = data
                    data = data.to(device)
                    
                    # Stack the global_cond
                    with self.accelerator.autocast():
                        with torch.no_grad():
                            # Stack the global_cond (from batch_size x state_dim into batch_size * sample_batch_size x state dim)
                            batch_size_sample = 32
                            local_cond_sample = torch.repeat_interleave(local_cond, repeats=batch_size_sample, dim=0) # For the local_cond, only to stack them
                            global_cond_sample = torch.repeat_interleave(global_cond, repeats=batch_size_sample, dim=0)
                            data = torch.repeat_interleave(data, repeats=batch_size_sample, dim=0)
                            # Sample a batch of data using the baseline policy
                            img, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model_baseline.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores = self.rewarding_model(global_cond_sample, img)
                            
                            # Size of reward_scores: (bs, 1) => (bs, )
                            reward_scores = torch.Tensor(reward_scores).squeeze(-1).to(self.accelerator.device)
                            # avg_reward_step = all_rewards_valid.mean().item() /self.gradient_accumulate_every
                            
                            # (batch_size*batch_size_sample, ) => (batch_size, batch_size_sample, )
                            reward_scores = reward_scores.view(global_cond.shape[0], batch_size_sample)
                            
                            # Find the best reward & the worst reward
                            best_reward_idx = torch.argmax(reward_scores, dim=1)
                            worst_reward_idx = torch.argmin(reward_scores, dim=1)

                            # Reshape the tensors
                            img_lst = img_lst.view(global_cond.shape[0], batch_size_sample, *img_lst.shape[1:])  # (B, sample_bs, D, T)
                            img_next_lst = img_next_lst.view(global_cond.shape[0], batch_size_sample, *img_next_lst.shape[1:])  # (B, sample_bs, D, T)

                            ts_lst = ts_lst.view(global_cond.shape[0], batch_size_sample, -1)  # (B, sample_bs, T)
                            log_probs_lst = log_probs_lst.view(global_cond.shape[0], batch_size_sample, -1)  # (B, sample_bs, T)
                            ts_lst = ts_lst[:, 0, :]  # Just take the ts from the first sample
                            
                            img_lst_w = img_lst[torch.arange(global_cond.shape[0]), best_reward_idx, ...]
                            img_next_w_lst = img_next_lst[torch.arange(global_cond.shape[0]), best_reward_idx, ...]
                            log_probs_w_lst = log_probs_lst[torch.arange(global_cond.shape[0]), best_reward_idx]
                            
                            # The "bad" or "lose" samples
                            img_lst_l = img_lst[torch.arange(global_cond.shape[0]), worst_reward_idx, ...]
                            img_next_l_lst = img_next_lst[torch.arange(global_cond.shape[0]), worst_reward_idx, ...]
                            log_probs_l_lst = log_probs_lst[torch.arange(global_cond.shape[0]), worst_reward_idx]
                            
                        # Obtain the PPOD loss
                        loss = self.model.forward_ppod_loss(local_cond, global_cond,
                                          img_w_lst = img_lst_w,
                                          img_l_lst = img_lst_l,
                                          img_next_w_lst = img_next_w_lst,
                                          img_next_l_lst = img_next_l_lst,
                                          ts_lst = ts_lst,
                                          log_probs_w_lst = log_probs_w_lst,
                                          log_probs_l_lst = log_probs_l_lst,
                                          logger = self.wandb_logger)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.mean().item()
                    self.accelerator.backward(loss.mean())

                pbar.set_description(f'PPOD loss: {total_loss:.4f}')
                if self.wandb_logger is not None:
                    self.wandb_logger.log({"PPOD Objective": total_loss})
                    

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt_ppo.step()
                self.opt_ppo.zero_grad()

                accelerator.wait_for_everyone()

                self.PPO_step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.PPO_step != 0 and self.PPO_step % self.reward_sample_ppo_every == 0 \
                        and self.wandb_logger is not None:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            data = next(self.dl)
                            data, local_cond, global_cond = data
                            data = data.to(device)
                            batch_size_sample = 32
                            local_cond_sample = torch.repeat_interleave(local_cond, repeats=batch_size_sample, dim=0) # For the local_cond, only to stack them
                            global_cond_sample = torch.repeat_interleave(global_cond, repeats=batch_size_sample, dim=0)
                            data = torch.repeat_interleave(data, repeats=batch_size_sample, dim=0)
                            # Sample a batch of data using the latest policy
                            img_new, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores_new = self.rewarding_model(global_cond_sample, img_new)
                            
                            
                            reward_new = reward_scores_new.mean().item()
                             # Sample a batch of data using the baseline policy
                            img, img_lst, img_next_lst, ts_lst, log_probs_lst = \
                                self.model_baseline.sample_verbose(global_cond.shape[0] * batch_size_sample, local_cond_sample, global_cond_sample)
                            reward_scores = self.rewarding_model(global_cond_sample, img)
                            self.wandb_logger.log({"PPOD Sampled Reward (Mean, baseline - new)": reward_scores.mean().item() - reward_new})
                pbar.update(1)

        accelerator.print('PPOD training complete')
        self.PPO_step = 0


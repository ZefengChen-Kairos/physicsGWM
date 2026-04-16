from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# from torch_cluster import fps
from pytorch3d.ops import sample_farthest_points as fps

from timm.models.layers import DropPath

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, point_dim=14):
        """
        Args:
            hidden_dim: size of sinusoidal XYZ positional encoding
            dim:        output embedding dimension
            point_dim:  total number of features per point.
                        Default 14 (original GWM).
                        Set to 26 when physics features are appended
                        (14 geometry + 12 physics).
        """
        super().__init__()
        assert hidden_dim % 6 == 0, "Hidden dim must be divisible by 6 for XYZ positional encoding"
        self.embedding_dim = hidden_dim
        self.point_dim = point_dim

        # Positional encoding basis for XYZ coordinates
        e = torch.pow(2, torch.arange(hidden_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(hidden_dim//6), torch.zeros(hidden_dim//6)]),  # X
            torch.cat([torch.zeros(hidden_dim//6), e, torch.zeros(hidden_dim//6)]),   # Y
            torch.cat([torch.zeros(hidden_dim//6), torch.zeros(hidden_dim//6), e]),   # Z
        ])
        self.register_buffer('basis', e)  # 3 x (hidden_dim//2)

        # MLP processes: positional_embeddings(hidden_dim) + all point features(point_dim)
        self.mlp = nn.Linear(hidden_dim + point_dim, dim)

    @staticmethod
    def embed(input, basis):
        # input: [B, N, 3] XYZ coordinates
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings  # [B, N, hidden_dim]
    
    def forward(self, input):
        # input: [B, N, point_dim] full point cloud features
        xyz = input[..., :3]   # [B, N, 3] — always the first 3 dims are XYZ

        # Get positional embeddings for XYZ
        pos_embed = self.embed(xyz, self.basis)  # [B, N, hidden_dim]

        # Concat positional encoding with ALL input features → (hidden_dim + point_dim) → dim
        combined = torch.cat([pos_embed, input], dim=-1)
        return self.mlp(combined)


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])  # bug fix: mean is [B,M,D] (3D), not 4D

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        point_dim=14,   # 14 = original geometry; 26 = geometry + physics
    ):
        super().__init__()

        self.depth = depth  # control the depth of the decoder

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim, point_dim=point_dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x D (D = point_dim, default 14; 26 with physics features)
        B, N, D = pc.shape
        assert N == self.num_inputs, f"Expected {self.num_inputs} point cloud inputs, got {N}"

        ###### FPS sampling based on XYZ coordinates
        _, sampled_indices = fps(pc[..., :3], K=self.num_latents)
        sampled_pc = torch.gather(pc, 1, 
            sampled_indices.unsqueeze(-1).expand(-1,-1,D))  # [B, K, D]

        ###### Embed full D-dim features
        sampled_pc_embeddings = self.point_embed(sampled_pc)  # [B, K, C]
        pc_embeddings = self.point_embed(pc)  # [B, N, C]

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x  # [B, K, C]

        return x


    def decode(self, x, queries=None):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        if queries is not None:
            queries_embeddings = self.point_embed(queries)
            latents = self.decoder_cross_attn(queries_embeddings, context = x)
        else:
            latents = x

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        x = self.encode(pc)

        # print(f"{x.shape=}")  # [B, 128, 128]

        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o}

class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        point_dim = 14,   # 14 = original geometry; 26 = geometry + physics
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim, point_dim=point_dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        # pc: [B, N, D]   D = point_dim (14 geometry, or 26 with physics)
        B, N, D = pc.shape
        assert N == self.num_inputs, f"Expected {self.num_inputs} points, got {N}"

        # FPS on XYZ only (first 3 dims) — bug-fix: was running FPS on all D dims
        _, sampled_idx = fps(pc[..., :3], K=self.num_latents)  # [B, M]
        sampled_pc = torch.gather(
            pc, 1,
            sampled_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, M, D]  — bug-fix: was hardcoded view(..., 3)

        sampled_pc_embeddings = self.point_embed(sampled_pc)  # [B, M, dim]
        pc_embeddings          = self.point_embed(pc)          # [B, N, dim]

        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings) + sampled_pc_embeddings
        x = cross_ff(x) + x   # [B, M, dim]

        mean   = self.mean_fc(x)    # [B, M, latent_dim]
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x  = posterior.sample()
        kl = posterior.kl()

        # Also return sampled_pc so callers can use it as decoder queries
        return kl, x, sampled_pc


    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries=None):
        kl, x, sampled_pc = self.encode(pc)
        # Use sampled_pc as decoder queries if none provided
        queries = queries if queries is not None else sampled_pc
        o = self.decode(x, queries).squeeze(-1)
        return {'logits': o, 'kl': kl}

def create_autoencoder(
        dim=512, M=512, depth=24, latent_dim=64, output_dim=1, N=2048,
        deterministic=False, point_dim=14
    ):
    if deterministic:
        model = AutoEncoder(
            depth=depth,
            dim=dim,
            queries_dim=dim,
            output_dim=output_dim,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
        )
    else:
        model = KLAutoEncoder(
            depth=depth,
            dim=dim,
            queries_dim=dim,
            output_dim=output_dim,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            point_dim=point_dim,
        )
    return model

def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, deterministic=False)
    
def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, deterministic=False)

def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, deterministic=False)

def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, deterministic=False)

def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, deterministic=False)

def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, deterministic=False)

def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, deterministic=False)

def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, deterministic=False)

###
def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, deterministic=True)

def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, deterministic=True)

def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, deterministic=True)

def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, deterministic=True)

###
def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, deterministic=True)

def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, deterministic=True)

def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, deterministic=True)

# low-resolution autoencoder
def ae_d64_m64(N=2048):
    return create_autoencoder(dim=128, M=128, depth=4, output_dim=3, N=N, deterministic=True)
import os
import sys
import math

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed

from safetensors.torch import load_file
# from accelerate import Accelerator
from gaussianwm.vq_model import LPIPS

from gaussianwm.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from gaussianwm.physics import PhysicsAugmentor
from gaussianwm.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from gaussianwm.diffusion.models import DiT_models, InnerModelConfig
from gaussianwm.reward.reward_model import RewardModel, RewardModelConfig
from dotmap import DotMap
from termcolor import cprint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.ops import sample_farthest_points as fps
import einops


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class GaussianPredictor(nn.Module):
    # def __init__(self, **kwargs) -> None:
    def __init__(self, args) -> None:
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.device = device

        # Initialize diffusion sampler
        denoiser_config = DenoiserConfig(
            inner_model=InnerModelConfig(
                input_size=args.model.input_size,
                patch_size=args.model.patch_size,
                in_channels=args.model.in_channels,
                action_dim=args.action_dim,
                hidden_size=args.model.hidden_size,
                depth=args.model.depth,
                num_heads=args.model.num_heads,
                mlp_ratio=args.model.mlp_ratio,
                class_dropout_prob=args.model.class_dropout_prob,
                learn_sigma=args.model.learn_sigma,
                context_length=args.context_length,
            ),
            sigma_data=args.diffusion.sigma_data,
            sigma_offset_noise=args.diffusion.sigma_offset_noise,
            noise_previous_obs=args.diffusion.noise_previous_obs,
        )
        reward_model_config = RewardModelConfig(
                lstm_dim=args.model.hidden_size,
                img_channels=args.model.in_channels,
                img_size=args.model.input_size,
                cond_channels=args.model.hidden_size,
                depths=[2, 2, 2],
                channels=[32, 32, 32],
                attn_depths=[0, 0, 0],
                action_dim=args.action_dim,
            )

        # Physics augmentation
        self.physics_augmentor = PhysicsAugmentor(args.physics) if hasattr(args, 'physics') else None
        if self.physics_augmentor and self.physics_augmentor.enabled:
            from gaussianwm.physics.gpt4v_predictor import get_physics_dim
            physics_dim = get_physics_dim(args.physics.feature_mode)
        else:
            physics_dim = 0

        # Splatt3r and VAE
        self.gaussian_feature_dim = 14 + physics_dim
        if args.observation.use_gs:
            from gaussianwm.processor.regressor import Splatt3rRegressor, gaussian_feature_to_dim
            self.splatt3r = Splatt3rRegressor().to(device).eval()
        if args.vae.use_vae:
            from gaussianwm.encoder.models_ae import create_autoencoder
            self.latent_dim = args.vae.latent_dim
            self.num_latents = args.vae.num_latents
            self.vae = create_autoencoder(
                depth=args.vae.vae_depth,
                dim=self.latent_dim,
                M=self.num_latents,
                latent_dim=self.latent_dim,
                output_dim=self.gaussian_feature_dim,
                N=args.observation.point_cloud_size,
                deterministic=not args.vae.use_kl,
                point_dim=self.gaussian_feature_dim,  # 14 or 26
            ).to(device)
            self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=args.optimizer.tok_lr)
            cprint(f"[VAE] Trainable parameters: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad)/1e6}M", 'yellow')
            cprint(f"[VAE] Total parameters: {sum(p.numel() for p in self.vae.parameters())/1e6}M", 'yellow')

        # Modify denoiser config for latent space if using either component
        if args.observation.use_gs:
            denoiser_config.inner_model.in_channels = self.gaussian_feature_dim
            if args.reward.use_reward_model:
                reward_model_config.img_channels = 14
        if args.vae.use_vae:
            denoiser_config.inner_model.in_channels = args.vae.latent_dim
            denoiser_config.inner_model.input_size = args.vae.num_latents
            denoiser_config.inner_model.patch_size = 1
            
            # Pre-compute spatial dimensions for reshaping when using VAE
            self.nh = int(math.sqrt(args.vae.num_latents))
            self.nw = self.nh  # Assuming square spatial dimensions
            # Update input_size to spatial dimensions
            denoiser_config.inner_model.input_size = self.nh

            if args.reward.use_reward_model:
                reward_model_config.img_size = self.nh
                reward_model_config.img_channels = args.vae.latent_dim

        self.model = Denoiser(denoiser_config).to(device)
        self.model.setup_training(
            SigmaDistributionConfig(
                loc=args.diffusion.sigma_loc,
                scale=args.diffusion.sigma_scale,
                sigma_min=args.diffusion.sigma_min,
                sigma_max=args.diffusion.sigma_max,
            )
        )
        cprint(f"[Model] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6}M", 'yellow')
        cprint(f"[Model] Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6}M", 'yellow')

        sampler_config = DiffusionSamplerConfig(
            num_steps_denoising=args.diffusion.num_steps_denoising,
            sigma_min=args.diffusion.sigma_min,
            sigma_max=args.diffusion.sigma_max,
            rho=args.diffusion.rho,
            order=args.diffusion.order,
        )
        self.diffusion_sampler = DiffusionSampler(self.model, sampler_config)

        # prepare for tokenizer training
        self.lpips = LPIPS().to(device).eval()
        # if args.selected_params:
        #     params = [parameter for name, parameter in self.tokenizer.named_parameters() if 'quantize' not in name]
        # else:
        #     params = list(self.tokenizer.parameters())
        # self.tok_optimizer = torch.optim.AdamW(
        #     params,
        #     lr=args.optimizer.tok_lr,
        #     betas=(args.optimizer.tok_beta1, args.optimizer.tok_beta2),
        #     weight_decay=args.optimizer.tok_wd,
        #     eps=1e-8,
        # )

        # prepare for model training
        self.model_optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.optimizer.model_lr)

        if args.reward.use_reward_model:
            self.reward_model = RewardModel(reward_model_config)
            self.reward_model_optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=args.optimizer.reward_model_lr)


    def _process_obs(self, obs, precomputed_points=None):
        """Convert RGB obs to latent embeddings with Gaussian processing (batched version)"""
        B, T, C, H, W = obs.shape

        if not self.args.observation.use_gs:
            return obs  # [B, T, C, H, W]

        obs_flat = obs.view(B*T, C, H, W)

        # ── Step 1: Splatt3R → 14-dim Gaussian features ──────────────────────
        # bug fix：接受预计算的 points，避免 forward() 里重复跑 Splatt3R
        if precomputed_points is not None:
            points = precomputed_points
        else:
            with torch.no_grad():
                points, _ = self.splatt3r.forward_tensor(obs_flat)  # [B*T, N, 14]

        # ── Step 2: physics augmentation (optional) ───────────────────────────
        # feature_mode="material" → +10 dims (one-hot only)   → [B*T, N, 24]
        # feature_mode="full"     → +12 dims (one-hot+hardness+shore) → [B*T, N, 26]
        # use_physics=false       → skip, stay at [B*T, N, 14]
        if self.physics_augmentor is not None and self.physics_augmentor.enabled:
            points = self.physics_augmentor.augment(obs_flat * 255., points)

        # ── Step 3: VAE encoding (optional) ───────────────────────────────────
        if self.args.vae.use_vae:
            enc = self.vae.encode(points)
            if isinstance(enc, tuple):
                # KLAutoEncoder returns (kl, z, sampled_pc); we want z
                enc = enc[1]                          # [B*T, M, latent_dim]
            enc = enc.view(B, T, -1, enc.shape[-1])  # [B, T, M, latent_dim]
            embeddings = enc.permute(0, 1, 3, 2).contiguous().view(
                B, T, enc.shape[-1], self.nh, self.nw
            )                                         # [B, T, latent_dim, nh, nw]
        else:
            # no VAE: reshape [B*T, N, D] → [B, T, D, H, W]
            D = points.shape[-1]
            embeddings = points.view(B, T, -1, D).permute(0, 1, 3, 2).contiguous()
            embeddings = embeddings.view(B, T, D, H, W)

        return embeddings

    def update(self, batch, update_tokenizer=True, update_model=True):
        start = time.time()
        metrics = {}
        if len(batch) == 3:
            obs, action, reward = batch
            pad_mask = None
        else:
            obs, action, reward, pad_mask = batch
            pad_mask = pad_mask.to(self.device)
        obs = obs.to(self.device) / 255.    # [B, T, C, H, W]
        action = action.to(self.device)     # [B, T, A]
        reward = reward.to(self.device)     # [B, T]
        if self.args.symlog:
            reward = symlog(reward)

        if update_tokenizer and self.args.vae.use_vae:
            metrics.update(self.update_vae(self.args, obs))
        if update_model:
            metrics.update(self.update_model(self.args, obs, action, reward, pad_mask))
        metrics.update({'model_update_time': time.time() - start})
        return metrics

    def update_vae(self, args, obs):
        """Updated VAE training with Gaussian features"""
        self.vae_optimizer.zero_grad()
        
        B, T, C, H, W = obs.shape
        obs_flat = obs.reshape(B*T, C, H, W)
        
        # Get Gaussian features from Splatt3r
        with torch.no_grad():
            points, _ = self.splatt3r.forward_tensor(obs_flat) # e.g., [160, 4096, 14]

        # Physics augmentation（与 _process_obs 保持一致）
        if self.physics_augmentor is not None and self.physics_augmentor.enabled:
            points = self.physics_augmentor.augment(obs_flat * 255., points)

        # VAE reconstruction
        enc = self.vae.encode(points)   # AutoEncoder → tensor; KLAutoEncoder → (kl, z, sampled_pc)
        if isinstance(enc, tuple):
            kl_loss, z, sampled_pc = enc
            recon = self.vae.decode(z, queries=sampled_pc)  # decode back to full point cloud
        else:
            kl_loss = torch.tensor(0.0, device=self.device)
            z       = enc
            recon   = self.vae.decode(z)

        # Reconstruction loss on Gaussian parameters
        recon_loss = F.mse_loss(recon, points)
        kl_weight  = getattr(args.vae, 'kl_weight', 1e-4)
        loss = recon_loss + kl_weight * kl_loss.mean()

        loss.backward()
        self.vae_optimizer.step()

        return {
            'tokenizer_loss': loss.item(),
            'recon_loss':     recon_loss.item(),
            'kl_loss':        kl_loss.mean().item(),
        }

    def update_model(self, args, obs, action, reward, pad_mask=None):
        """Update the diffusion model using denoising loss"""
        self.model.train()
        self.model_optimizer.zero_grad()
        if args.reward.use_reward_model:
            self.reward_model.train()
            self.reward_model_optimizer.zero_grad()
        
        # Process observations to latent space
        latent_embeddings = self._process_obs(obs)  # [B, T, C, H, W]
        
        # Forward through diffusion model
        diff_loss = self.model(
            latent_embeddings, 
            action,
            batch_mask_padding=pad_mask
        )
        reward_loss, reward_pred = 0.0, None
        if args.reward.use_reward_model:
            reward_loss, reward_pred = self.reward_model(
                latent_embeddings[:, args.context_length:-1], 
                action[:, args.context_length:-1],
                latent_embeddings[:, args.context_length+1:], 
                reward[:, args.context_length:-1]
            )

        # Calculate losses
        loss = diff_loss
        if args.reward.use_reward_model:
            loss += args.reward.reward_weight * reward_loss
        
        # Backward and optimize
        loss.backward()
        grad_norm, grad_norms = 0, {}
        if args.optimizer.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.optimizer.max_grad_norm)
            grad_norm = grad_norm.item()
        self.model_optimizer.step()
        if args.reward.use_reward_model:
            self.reward_model_optimizer.step()

        return {
            'diff_loss': diff_loss.item(),
            'model_loss': loss.item(), 
            **({'reward_loss': reward_loss.item(),
               'model_train/reward_mean': reward[:, args.context_length:].mean().item(),
               'model_train/reward_pred_mean': reward_pred.mean().item()} if args.reward.use_reward_model else {}),
            'model_train/grad_norm': grad_norm,
            **grad_norms,
        }

    def forward(self, batch, update_tokenizer=True, update_model=True):
        start = time.time()
        metrics = {}
        total_loss = torch.tensor(0.0).to(self.device)
        
        if len(batch) == 3:
            obs, action, reward = batch
            pad_mask = None
        else:
            obs, action, reward, pad_mask = batch
            pad_mask = pad_mask.to(self.device)
        obs = obs.to(self.device) / 255.    # [B, T, C, H, W]
        action = action.to(self.device)     # [B, T, A]
        reward = reward.to(self.device)     # [B, T]
        if self.args.symlog:
            reward = symlog(reward)

        # Calculate VAE loss without optimization
        if update_tokenizer and self.args.vae.use_vae:
            B, T, C, H, W = obs.shape
            obs_flat = obs.reshape(B*T, C, H, W)
            
            # Get Gaussian features from Splatt3r
            with torch.no_grad():
                points, _ = self.splatt3r.forward_tensor(obs_flat)

            # Physics augmentation（与 _process_obs 保持一致）
            if self.physics_augmentor is not None and self.physics_augmentor.enabled:
                points = self.physics_augmentor.augment(obs_flat * 255., points)

            # VAE reconstruction (same fix as update_vae)
            enc = self.vae.encode(points)
            if isinstance(enc, tuple):
                kl_loss, z, sampled_pc = enc
                recon = self.vae.decode(z, queries=sampled_pc)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)
                z       = enc
                recon   = self.vae.decode(z)

            kl_weight  = getattr(self.args.vae, 'kl_weight', 1e-4)
            recon_loss = F.mse_loss(recon, points)
            vae_loss   = recon_loss + kl_weight * kl_loss.mean()
            total_loss += vae_loss

            metrics.update({
                'tokenizer_loss': vae_loss.item(),
                'recon_loss':     recon_loss.item(),
                'kl_loss':        kl_loss.mean().item(),
            })

        # Calculate model loss without optimization
        if update_model:
            self.model.train()
            if self.args.reward.use_reward_model:
                self.reward_model.train()

            # bug fix：如果 VAE 分支已经算过 points，直接复用，避免 Splatt3R 跑两次
            precomputed = points if (update_tokenizer and self.args.vae.use_vae
                                     and self.args.observation.use_gs) else None
            latent_embeddings = self._process_obs(obs, precomputed_points=precomputed)
            
            # Forward through diffusion model
            diff_loss = self.model(
                latent_embeddings, 
                action,
                batch_mask_padding=pad_mask
            )
            total_loss += diff_loss
            
            reward_loss, reward_pred = 0.0, None
            if self.args.reward.use_reward_model:
                reward_loss, reward_pred = self.reward_model(
                    latent_embeddings[:, self.args.context_length:-1], 
                    action[:, self.args.context_length:-1],
                    latent_embeddings[:, self.args.context_length+1:], 
                    reward[:, self.args.context_length:-1]
                )

            # Calculate total model loss
            if self.args.reward.use_reward_model:
                total_loss += self.args.reward.reward_weight * reward_loss

            metrics.update({
                'diff_loss': diff_loss.item(),
                **({'reward_loss': reward_loss.item(),
                   'model_train/reward_mean': reward[:, self.args.context_length:].mean().item(),
                   'model_train/reward_pred_mean': reward_pred.mean().item()} 
                   if self.args.reward.use_reward_model else {}),
            })

        metrics.update({
            "total_loss": total_loss.item(),
            # bug fix: diff_loss 只在 update_model=True 时存在
            **({"diff_loss": diff_loss.item()} if update_model else {}),
        })
        
        return total_loss, metrics
    
    @torch.no_grad()
    def rollout(self, obs, policy, horizon):
        self.model.eval()
        args = self.args

        x = obs.to(self.device).float()
        B, Ctot, H, W = x.shape

        if args.observation.use_gs:
            ch_per_frame = (args.vae.latent_dim if args.vae.use_vae else 14)
            assert Ctot % args.context_length == 0
            frames_img = [x[:, i*(Ctot//args.context_length):(i+1)*(Ctot//args.context_length)] for i in range(args.context_length)]
            context_imgs = torch.stack(frames_img, dim=1)  # [B, T, C_img, H, W]
            context_latents = self._process_obs(context_imgs / 255.)  # [B, T, Cg, H', W']
            frames = [context_latents[:, i] for i in range(args.context_length)]  # list of [B, Cg, H', W']

            obss = [torch.cat(frames, dim=1)]
            actions, rewards = [], []

            for t in range(horizon):
                ctx = torch.stack(frames[-args.context_length:], dim=1)
                obs_for_policy = torch.cat(frames[-args.context_length:], dim=1)
                action = policy(obs_for_policy, t)
                next_latent = self.diffusion_sampler.sample(ctx, action)[0]

                if args.reward.use_reward_model:
                    prev_lat = ctx[:, -1].unsqueeze(1)
                    rew_pred, _ = self.reward_model.predict_rew(prev_lat, action, next_latent)
                    rew_pred = rew_pred.squeeze(1)
                else:
                    rew_pred = torch.zeros(action.size(0), device=self.device)

                frames.append(next_latent)
                frames.pop(0)

                obss.append(torch.cat(frames[-args.context_length:], dim=1))
                actions.append(action)
                rewards.append(rew_pred)

            actions = [torch.zeros_like(actions[0])] + actions
            rewards = [torch.zeros_like(rewards[0])] + rewards
            if args.symlog:
                rewards = [symexp(r) for r in rewards]

            return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

        frames = [x[:, i*(Ctot//args.context_length):(i+1)*(Ctot//args.context_length)] for i in range(args.context_length)]
        obss, actions, rewards = [torch.cat(frames, dim=1)], [], []

        for t in range(horizon):
            ctx_imgs = torch.stack(frames[-args.context_length:], dim=1)  # [B,T,C,H,W]
            ctx_latents = self._process_obs(ctx_imgs / 255.)
            action = policy(torch.cat(frames[-args.context_length:], dim=1), t)
            next_latent = self.diffusion_sampler.sample(ctx_latents, action)[0]
            next_obs = next_latent

            if args.reward.use_reward_model:
                reward_pred, _ = self.reward_model.predict_rew(ctx_latents[:, -1].unsqueeze(1), action, next_obs)
                reward_pred = reward_pred.squeeze(1)
            else:
                reward_pred = torch.zeros_like(action[:, 0])

            frames.append(next_obs)  # bug fix: latent 不应被 clamp(0,1)，会截断负值
            frames.pop(0)
            obss.append(torch.cat(frames[-args.context_length:], dim=1))
            actions.append(action)
            rewards.append(reward_pred)

        actions = [torch.zeros_like(actions[0])] + actions
        rewards = [torch.zeros_like(rewards[0])] + rewards
        if args.symlog:
            rewards = [symexp(reward) for reward in rewards]

        return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

    def save_snapshot(self, workdir, suffix=''):
        # bug fix：原来只存 diffusion model，VAE 和 reward model 丢失
        # bug fix：DDP 解包判断对象应为传入的 wrapper，这里用 getattr 兼容两种情况
        unwrapped = getattr(self, 'module', self)
        snapshot = {'model': unwrapped.model.state_dict()}
        if self.args.vae.use_vae:
            snapshot['vae'] = unwrapped.vae.state_dict()
        if self.args.reward.use_reward_model:
            snapshot['reward_model'] = unwrapped.reward_model.state_dict()
        torch.save(snapshot, os.path.join(workdir, f'snapshot{suffix}.pt'))

    def load_snapshot(self, workdir, suffix=''):
        map_loc = f'cuda:{dist.get_rank()}' if dist.is_initialized() else 'cpu'
        snapshot = torch.load(os.path.join(workdir, f'snapshot{suffix}.pt'), map_location=map_loc)
        self.model.load_state_dict(snapshot['model'])
        if self.args.vae.use_vae and 'vae' in snapshot:
            self.vae.load_state_dict(snapshot['vae'])
        if self.args.reward.use_reward_model and 'reward_model' in snapshot:
            self.reward_model.load_state_dict(snapshot['reward_model'])
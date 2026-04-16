import hashlib
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from .gpt4v_predictor import query_vlm, build_physics_feature, get_physics_dim
from .sam_segmentor import segment, build_mask_generator


def _frame_hash(rgb_np: np.ndarray) -> str:
    """RGB numpy array → MD5 hash，用作缓存文件名。"""
    return hashlib.md5(rgb_np.tobytes()).hexdigest()[:16]


def _tensor_to_numpy_rgb(t: torch.Tensor) -> np.ndarray:
    """
    [C, H, W] float tensor → uint8 [H, W, 3] numpy。

    bug fix：原来直接 astype(np.uint8)，如果上游传入 0-1 归一化的图
    会全部截断为 0（全黑），SAM 和 GPT-4V 拿到黑图不报错。
    现在自动检测量程并 scale。
    """
    arr = t.permute(1, 2, 0).cpu().numpy()[..., :3]
    if arr.max() <= 1.0:
        print("[PhysicsAugmentor] Warning: obs looks like 0-1 float, scaling to 0-255")
        arr = arr * 255.0
    return arr.clip(0, 255).astype(np.uint8)


class PhysicsAugmentor:
    """
    把 [B, N, 14] Gaussian 点云增强为 [B, N, 14+physics_dim]。

    Config (args.physics):
        use_physics    bool  — 总开关
        feature_mode   str   — "material"（10维）或 "full"（12维）
        gpt4v_api_key  str   — OpenAI key，null 则全部填零
        vlm_type       str   — "gpt" 或 "qwen"
        gpt4v_model    str   — 例如 "gpt-4o-mini"
        sam_ckpt_path  str   — SAM checkpoint 路径
        cache_dir      str   — 缓存目录
        device         str   — "cuda" 或 "cpu"
    """

    def __init__(self, cfg):
        self.enabled      = getattr(cfg, "use_physics",   False)
        self.feature_mode = getattr(cfg, "feature_mode",  "material")
        self.physics_dim  = get_physics_dim(self.feature_mode)

        if not self.enabled:
            return

        self.api_key   = getattr(cfg, "gpt4v_api_key", None)
        self.vlm_type  = getattr(cfg, "vlm_type",      "gpt")
        self.model     = getattr(cfg, "gpt4v_model",   "gpt-4o-mini")
        self.device    = getattr(cfg, "device",        "cuda")
        self.cache_dir = Path(getattr(cfg, "cache_dir", "./physics_cache"))
        self.sam_ckpt  = getattr(cfg, "sam_ckpt_path", "./sam_vit_h_4b8939.pth")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mask_generator = None

    # ── SAM 初始化 ────────────────────────────────────────────────────────────

    def _get_mask_generator(self):
        if self._mask_generator is None:
            print("[PhysicsAugmentor] Loading SAM...")
            self._mask_generator = build_mask_generator(self.sam_ckpt, self.device)
        return self._mask_generator

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _load_cache(self, frame_hash: str):
        path = self.cache_dir / f"{frame_hash}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, frame_hash: str, feat: np.ndarray):
        with open(self.cache_dir / f"{frame_hash}.pkl", "wb") as f:
            pickle.dump(feat, f)

    # ── 单帧处理 ──────────────────────────────────────────────────────────────

    def _process_frame(self, rgb_np: np.ndarray, gaussians_np: np.ndarray) -> np.ndarray:
        """
        对一帧图像运行 SAM + GPT-4V，把每个零件的物理属性
        投影到对应的 Gaussian 点上。

        返回: [N, physics_dim] float32
        """
        H, W = rgb_np.shape[:2]
        N    = gaussians_np.shape[0]

        # 查缓存
        fhash  = _frame_hash(rgb_np)
        cached = self._load_cache(fhash)
        if cached is not None:
            if cached.shape == (N, self.physics_dim):
                return cached
            # bug fix：shape 不匹配说明缓存过期，打 warning 后重新计算
            print(f"[PhysicsAugmentor] cache shape mismatch "
                  f"{cached.shape} != ({N}, {self.physics_dim}), recomputing...")

        # Step 1: SAM 分割
        pil_img = Image.fromarray(rgb_np)
        seg_map, triple_images = segment(pil_img, self._get_mask_generator())

        # Step 2: 每个零件调用 GPT-4V
        part_features = {}
        raw_preds     = {}   # 保存原始 pred，用于判断是否有失败项

        if self.api_key is not None:
            for part_id, triple_img in triple_images.items():
                pred = query_vlm(
                    triple_img,
                    api_key=self.api_key,
                    vlm_type=self.vlm_type,
                    model=self.model,
                )
                part_features[part_id] = build_physics_feature(pred, mode=self.feature_mode)
                raw_preds[part_id]     = pred
        else:
            for part_id in triple_images:
                part_features[part_id] = np.zeros(self.physics_dim, dtype=np.float32)

        # Step 3: 投影到每个 Gaussian 点
        physics  = np.zeros((N, self.physics_dim), dtype=np.float32)
        seg_flat = seg_map.reshape(-1)

        if N == H * W:
            # 直接用像素索引查 seg_map
            for part_id, feat in part_features.items():
                physics[seg_flat == part_id] = feat
        else:
            # N != H*W，用 Gaussian 中心坐标近似投影
            # bug fix：加断言保护列索引，加 warning 告知这是近似
            assert gaussians_np.shape[1] >= 6, \
                f"[PhysicsAugmentor] Unexpected Gaussian dim {gaussians_np.shape[1]}, expected >= 6"
            print(f"[PhysicsAugmentor] Warning: N({N}) != H*W({H*W}), "
                  f"using approximate projection from means_in_other_view")

            means_cam = gaussians_np[:, 3:6]   # means_in_other_view（相机坐标系）
            mn, mx = means_cam.min(0), means_cam.max(0)
            denom  = np.where((mx - mn) > 1e-6, mx - mn, 1.0)
            norm   = (means_cam - mn) / denom
            rows   = np.clip((norm[:, 1] * (H - 1)).astype(int), 0, H - 1)
            cols   = np.clip((norm[:, 0] * (W - 1)).astype(int), 0, W - 1)
            part_ids = seg_map[rows, cols]
            for part_id, feat in part_features.items():
                physics[part_ids == part_id] = feat

        # bug fix：只有全部 part 都成功时才缓存
        # 有失败项时不缓存，下次会重试，避免把全零结果永久缓存
        has_failure = any(p.get("failed", False) for p in raw_preds.values())
        if not has_failure:
            self._save_cache(fhash, physics)
        else:
            print(f"[PhysicsAugmentor] some parts failed, skipping cache write")

        return physics

    # ── 对外接口 ──────────────────────────────────────────────────────────────

    def augment(self, obs: torch.Tensor, gaussians: torch.Tensor) -> torch.Tensor:
        """
        输入:
            obs:       [B, C, H, W]  float  0-255
            gaussians: [B, N, 14]

        输出:
            [B, N, 14 + physics_dim]
        """
        B, N, _ = gaussians.shape

        if not self.enabled:
            zeros = torch.zeros(B, N, self.physics_dim,
                                dtype=gaussians.dtype, device=gaussians.device)
            return torch.cat([gaussians, zeros], dim=-1)

        physics_batch = []
        for b in range(B):
            rgb_np       = _tensor_to_numpy_rgb(obs[b])
            gaussians_np = gaussians[b].cpu().numpy()
            feat_np      = self._process_frame(rgb_np, gaussians_np)
            physics_batch.append(feat_np)

        physics = torch.from_numpy(np.stack(physics_batch))
        physics = physics.to(dtype=gaussians.dtype, device=gaussians.device)
        return torch.cat([gaussians, physics], dim=-1)
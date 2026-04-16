import cv2
import torch
import numpy as np
from PIL import Image


# ── 以下函数移植自 GaussianProperty sam_utils.py ─────────────────────────────

def get_seg_img(image, mask, bbox):
    """Extracts a segmented image using the mask and bounding box."""
    image = image.copy()
    image[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(bbox)
    return image[y:y+h, x:x+w, ...]


def pad_img(img):
    """Pads the image to make it square."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> list:
    """Filters masks based on the indices in `keep`."""
    keep = keep.int().cpu().numpy()
    return [m for i, m in enumerate(masks_result) if i in keep]


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """Performs non-maximum suppression on masks."""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # bug fix：原始代码用 [index, 0] 做 1D tensor 的索引，维度错误
    # 同时加 min(3, num_masks) 防止 mask 数量不足 3 时 topk 崩溃
    k = min(3, num_masks)
    if keep_conf.sum() == 0:
        keep_conf[scores.topk(k).indices] = True
    if keep_inner_u.sum() == 0:
        keep_inner_u[scores.topk(k).indices] = True
    if keep_inner_l.sum() == 0:
        keep_inner_l[scores.topk(k).indices] = True

    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    return idx[keep]


def masks_update(*args, **kwargs):
    """Removes redundant masks based on scores and overlap rate."""
    masks_new = ()
    for masks_lvl in (args):
        seg_pred  = torch.from_numpy(np.stack([m['segmentation']    for m in masks_lvl], axis=0))
        iou_pred  = torch.from_numpy(np.stack([m['predicted_iou']   for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def get_location(image, foreground_mask):
    """Finds the bounding box for the largest contour in the mask."""
    contours, _ = cv2.findContours(
        foreground_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # bug fix：轮廓为空时退化为全图 bbox，避免 contours[0] 崩溃
    if not contours:
        h, w = foreground_mask.shape
        return [0, 0, w, h]

    x, y, w, h = cv2.boundingRect(contours[0])
    return [x, y, w, h]


# ── 以下是改动部分 ────────────────────────────────────────────────────────────

def _build_triple_image(image: np.ndarray, seg_map: np.ndarray,
                        part_id: int) -> Image.Image:
    """
    移植自 GaussianProperty save_gpt_input() 里的三联图逻辑。
    改动：原始代码用 matplotlib 存磁盘，这里改为返回内存里的 PIL Image。
    返回：[原图 | mask overlay | 裁剪零件] 三联图
    """
    H, W = image.shape[:2]
    mask = seg_map == part_id

    # 左：原图
    panel_orig = image.copy()

    # 中：mask overlay（红色，alpha=0.4）
    panel_overlay = image.copy()
    colored = panel_overlay.copy()
    colored[mask] = [255, 0, 0]
    panel_overlay = cv2.addWeighted(panel_overlay, 0.6, colored, 0.4, 0)

    # 右：裁剪零件
    ys, xs = np.where(mask)
    if len(ys) == 0:
        panel_part = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        bbox = get_location(image, mask)
        seg_img = get_seg_img(image, mask, bbox)
        padded  = pad_img(seg_img)
        panel_part = cv2.resize(padded, (W, H))

    triple = np.concatenate([panel_orig, panel_overlay, panel_part], axis=1)
    return Image.fromarray(triple)


def segment(image: Image.Image, mask_generator) -> tuple:
    """
    移植自 GaussianProperty sam_encoder()。
    改动 1：入参从 torch tensor 改为 PIL Image。
    改动 2：不存任何文件到磁盘。
    改动 3：返回 (seg_map, triple_images)，triple_images 是 {part_id: PIL Image} 字典。

    返回：
        seg_map       np.ndarray [H, W] int32，-1 = 背景
        triple_images dict {part_id: PIL.Image}
    """
    image_np = np.array(image.convert("RGB"))

    if image.mode == "RGBA":
        alpha = np.array(image)[:, :, 3]
        alpha = (alpha >= 125).astype(np.uint8) * 255
    else:
        alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image_bgr)
    masks_m = masks_update(masks_m, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

    seg_map = -np.ones(image_np.shape[:2], dtype=np.int32)
    seg_map[alpha == 255] = 0

    masks_m = sorted(masks_m, key=lambda x: x['area'], reverse=True)
    for kk, mask in enumerate(masks_m):
        if kk == 0 or mask['segmentation'].sum() < 300:
            continue
        seg_map[mask['segmentation']] = kk

    seg_map[alpha == 0] = -1

    triple_images = {}
    for part_id in np.unique(seg_map):
        # bug fix：跳过 -1（背景）和 0（整体前景）
        # part_id=0 覆盖整个前景，喂给 VLM 意义不大，还浪费 API 调用
        if part_id <= 0:
            continue
        triple_images[int(part_id)] = _build_triple_image(image_np, seg_map, int(part_id))

    return seg_map, triple_images


def build_mask_generator(sam_ckpt_path: str, device: str = "cuda"):
    """
    初始化 SAM mask generator。
    参数与 GaussianProperty sam_preprocess.py 完全一致。
    """
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=300,
    )
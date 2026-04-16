import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI


# 与 GaussianProperty vlm_predict.py 完全一致的材质库
MATERIAL_LIST = [
    "wood", "metal", "plastic", "glass",
    "fabric", "foam", "food", "ceramic", "paper", "leather",
]
MATERIAL_TO_IDX = {m: i for i, m in enumerate(MATERIAL_LIST)}


def encode_image(img: Image.Image) -> str:
    """PIL Image → base64 JPEG string。原始代码从文件读，这里改为从 PIL Image 读。"""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_prompt() -> str:
    """与 GaussianProperty vlm_predict.py 的 prompt 一字不差。"""
    material_library = "{" + ", ".join(MATERIAL_LIST) + "}"
    return f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object. 
    Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to predict the hardness of the part. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. 

    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, hardness, Shore A/D) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    common material library: {material_library}. 
    Your answer must look like: caption, material, hardness low-high, <Shore A or Shore D>. 
    The material type must be chosen from the above common material library. Make sure to use Shore A or Shore D hardness, not Mohs hardness."""


def GPT4V(image: Image.Image, prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    移植自 GaussianProperty GPT4V()。
    改动 1：入参从文件路径改为 PIL Image。
    改动 2：返回 .message.content 字符串，原始代码返回 Choice 对象（bug）。
    改动 3：api_key 作为参数传入，原始代码是硬编码。
    """
    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def Qwen(image: Image.Image, prompt: str, api_key: str) -> str:
    """
    移植自 GaussianProperty Qwen()。
    改动 1：入参从文件路径改为 PIL Image。
    改动 2：api_key 作为参数传入，原始代码是硬编码。
    改动 3：删掉内部 print（bug fix：原来 Qwen 会打印两次，GPT 只打印一次）。
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    base64_image = encode_image(image)

    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


def query_vlm(image: Image.Image, api_key: str,
              vlm_type: str = "gpt", model: str = "gpt-4o-mini") -> dict:
    """
    对一张三联图调用 VLM，返回解析后的 dict。
    失败时返回 failed=True 的 dict（对应原始代码的 "error,-1"）。

    Bug fix：raw 在 try 前初始化为空字符串，
    保证 API 失败时 except 里能拿到 partial 信息用于调试。
    """
    prompt = _build_prompt()
    raw = ""  # bug fix：提前初始化，避免 except 里 UnboundLocalError

    try:
        if vlm_type == "qwen":
            raw = Qwen(image, prompt, api_key)
        else:
            raw = GPT4V(image, prompt, api_key, model)
        print(f"[VLM] {raw}")
    except Exception as e:
        print(f"[VLM] error: {e}")
        return _failed_pred(raw_response=f"error: {e} | partial_raw: {raw}")

    return _parse_response(raw)


def _failed_pred(raw_response: str = "") -> dict:
    """解析失败时的返回值，对应原始代码的 'error,-1'。"""
    return {
        "caption":      "unknown",
        "material":     None,
        "hardness_mid": 0.0,
        "shore_type":   0.0,
        "raw_response": raw_response,
        "failed":       True,
    }


def _parse_response(raw: str) -> dict:
    """
    解析 GPT-4V 返回的字符串。
    格式：caption, material, hardness low-high, Shore A/D
    失败直接返回 _failed_pred，不做任何猜测。
    """
    result = _failed_pred(raw_response=raw)

    try:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            print(f"[VLM] parse failed (need 4 fields): {raw}")
            return result

        caption    = parts[0]
        material   = parts[1].lower().replace(" ", "")
        hardness_s = parts[2]
        shore_s    = parts[3].lower()

        # material 校验，不在库内就尝试子串匹配
        if material not in MATERIAL_TO_IDX:
            matched = [m for m in MATERIAL_LIST if m in material]
            if not matched:
                print(f"[VLM] unknown material '{material}', returning zero vector")
                return result
            # bug fix：有多个匹配时打 warning，便于排查
            if len(matched) > 1:
                print(f"[VLM] ambiguous material '{material}', matched {matched}, using '{matched[0]}'")
            material = matched[0]

        # hardness 解析
        nums = re.findall(r"[\d.]+", hardness_s)
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
        elif len(nums) == 1:
            lo = hi = float(nums[0])
        else:
            print(f"[VLM] cannot parse hardness '{hardness_s}', returning zero vector")
            return result

        shore_type = 0.0 if "shore a" in shore_s else 1.0

        # bug fix：Shore A 和 Shore D 量纲不同，分开归一化到统一尺度
        # Shore A（软材料）→ [0.0, 0.5]
        # Shore D（硬材料）→ [0.5, 1.0]
        mid = float(np.clip((lo + hi) / 2.0 / 100.0, 0.0, 1.0))
        if shore_type == 0.0:   # Shore A
            hardness_mid = mid * 0.5
        else:                   # Shore D
            hardness_mid = 0.5 + mid * 0.5

        result.update({
            "caption":      caption,
            "material":     material,
            "hardness_mid": hardness_mid,
            "shore_type":   shore_type,
            "failed":       False,
        })

    except Exception as e:
        print(f"[VLM] parse error on '{raw}': {e}")

    return result


def build_physics_feature(pred: dict, mode: str = "material") -> np.ndarray:
    """
    把 VLM 返回的 dict 转成 numpy 向量。

    mode="material" → 10 维，只用材质 one-hot
    mode="full"     → 12 维，材质 one-hot + 硬度（统一尺度） + Shore 类型

    失败（failed=True）→ 全零，不引入任何先验。
    """
    dim = get_physics_dim(mode)

    if pred.get("failed", True) or pred.get("material") is None:
        return np.zeros(dim, dtype=np.float32)

    onehot = np.zeros(len(MATERIAL_LIST), dtype=np.float32)
    onehot[MATERIAL_TO_IDX[pred["material"]]] = 1.0

    if mode == "material":
        return onehot                                           # (10,)

    # mode == "full"
    return np.concatenate([
        onehot,
        np.array([pred["hardness_mid"]], dtype=np.float32),    # (1,) 统一尺度 [0,1]
        np.array([pred["shore_type"]],   dtype=np.float32),    # (1,) 0=Shore A, 1=Shore D
    ])                                                          # (12,)


def get_physics_dim(mode: str) -> int:
    if mode == "material":
        return 10
    if mode == "full":
        return 12
    raise ValueError(f"Unknown feature_mode '{mode}'. Choose 'material' or 'full'.")
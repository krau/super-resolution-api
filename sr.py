import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from onnx_infer import OnnxSRInfer


@dataclass
class ModelInfo:
    name: str = ""
    path: str = ""
    scale: int = 4
    algo: str = ""


model = ModelInfo(
    "x4_Anime_6B-Official", "models/x4_Anime_6B-Official.onnx", 4, "real-esrgan"
)


def process_image(
    tile_size: int = 64,  # 分块大小
    scale: int = 4,  # 放大倍数
    skip_alpha: bool = False,  # 是否跳过alpha通道
    resize_to: str = None,  # 调整大小 两种格式: 1. 1920x1080 2. 1/2
    input_image: Path = None,
    output_path: Path | str = "output",
    gpuid: int = 0,
) -> Path:
    logger.info(f"input_image: {input_image}")
    global model
    try:
        provider_options = None
        if int(gpuid) >= 0:
            provider_options = [{"device_id": int(gpuid)}]

        sr_instance = OnnxSRInfer(
            model.path,
            model.scale,
            model.name,
            provider_options=provider_options,
        )
        if skip_alpha:
            logger.debug("Skip Alpha Channel")
            sr_instance.alpha_upsampler = "interpolation"

        img = cv2.imdecode(
            np.fromfile(input_image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        h, w, _ = img.shape
        sr_img = sr_instance.universal_process_pipeline(img, tile_size=tile_size)
        scale = int(scale)
        target_h = None
        target_w = None
        if scale > model.scale and model.scale != 1:
            logger.debug("re process")
            # calc process times
            scale_log = math.log(scale, model.scale)
            total_times = math.ceil(scale_log)
            # calc target size
            if total_times != int(scale_log):
                target_h = h * scale
                target_w = w * scale

            for _ in range(total_times - 1):
                sr_img = sr_instance.universal_process_pipeline(
                    sr_img, tile_size=tile_size
                )
        elif scale < model.scale:
            logger.debug("down scale")
            target_h = h * scale
            target_w = w * scale

        if resize_to:
            logger.debug(f"resize to {resize_to}")
            if "x" in resize_to:
                param_w = int(resize_to.split("x")[0])
                target_w = param_w
                target_h = int(h * param_w / w)
            elif "/" in resize_to:
                ratio = int(resize_to.split("/")[0]) / int(resize_to.split("/")[1])
                target_w = int(w * ratio)
                target_h = int(h * ratio)

        if target_w:
            logger.debug(f"resize to {target_w}x{target_h}")
            img_out = cv2.resize(sr_img, (target_w, target_h))
        else:
            img_out = sr_img
        # save
        final_output_path = Path(output_path) / f"{input_image.stem}_{model.name}.png"
        if not Path(output_path).exists():
            Path(output_path).mkdir(parents=True)
        logger.info(f"save to {final_output_path}")
        cv2.imencode(".png", img_out)[1].tofile(final_output_path)
        return final_output_path
    except Exception as e:
        sr_instance = None
        logger.error(f"error: {e}")
        return None

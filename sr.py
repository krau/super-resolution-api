import datetime
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut
from loguru import logger

import common
from config import settings
from onnx_infer import OnnxSRInfer


@func_set_timeout(common.PROGRESS_TIMEOUT, allowOverride=True)
def _process_image(
    model: common.ModelInfo = common.models[common.MODEL_NAME_DEFAULT],
    tile_size: int = 64,  # 分块大小
    scale: int = 4,  # 放大倍数
    skip_alpha: bool = False,  # 是否跳过alpha通道
    resize_to: str = None,  # 调整大小 两种格式: 1. 1920x1080 2. 1/2
    input_image: Path = None,
    output_path: Path | str = "output",
    gpuid: int = 0,
    clean: bool = True,
) -> Path:
    logger.debug(f"processing image: {input_image}")
    start_time = datetime.datetime.now()
    try:
        provider_options = None
        if int(gpuid) >= 0:
            provider_options = [{"device_id": int(gpuid)}]
        sr_instance = OnnxSRInfer(
            model.path,
            model.scale,
            model.name,
            providers=[settings.get("provider", "CPUExecutionProvider")],
            provider_options=provider_options,
        )
        if skip_alpha:
            logger.debug("Skip Alpha Channel")
            sr_instance.alpha_upsampler = "interpolation"
        logger.debug(f"decoding image: {input_image}")
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
        cv2.imencode(".png", img_out)[1].tofile(final_output_path)
        return final_output_path
    except Exception as e:
        logger.error(f"process image error: {e}")
        return None
    finally:
        logger.debug(
            f"Time taken: {(datetime.datetime.now() - start_time).seconds} seconds to process {input_image}"
        )
        if clean and input_image.exists():
            input_image.unlink()


def listen_queue(
    stream_name: str = common.STREAM_NAME,
    default_timeout: int = common.PROGRESS_TIMEOUT,
):
    logger.info(f"Listening to stream: {stream_name}")
    last_id = "0"
    while True:
        messages = common.redis_client.xread({stream_name: last_id}, count=1, block=0)
        if not messages:
            continue
        message_id = messages[0][1][0][0]
        last_id = message_id
        message = messages[0][1][0][1]
        logger.info(f"Processing task: {message_id.decode('utf-8')}")
        data: dict[str, Path | int | bool | str | None] = pickle.loads(message[b"data"])
        input_image = data.get("input_image")
        tile_size = data.get("tile_size", 64)
        scale = data.get("scale", 4)
        skip_alpha = data.get("skip_alpha", False)
        resize_to = data.get("resize_to", None)
        time_out = data.get("timeout", default_timeout)
        model_name = data.get("model", common.MODEL_NAME_DEFAULT)
        common.redis_client.set(
            f"super_resolution_api_result_{message_id.decode('utf-8')}",
            pickle.dumps({"status": "processing"}),
            ex=86400,
        )
        processed_path: Path | None = None
        try:
            processed_path = _process_image(
                model=common.models[model_name],
                input_image=input_image,
                tile_size=tile_size,
                scale=scale,
                skip_alpha=skip_alpha,
                resize_to=resize_to,
                forceTimeout=time_out,
            )
        except FunctionTimedOut as e:
            logger.warning(e)
            processed_path = None
        if processed_path:
            common.redis_client.set(
                f"super_resolution_api_result_{message_id.decode('utf-8')}",
                pickle.dumps(
                    {
                        "status": "success",
                        "path": processed_path.as_posix(),
                        "size": processed_path.stat().st_size,
                    }
                ),
                ex=86400,
            )
            logger.success(f"Processed image: {processed_path}")
        else:
            common.redis_client.set(
                f"super_resolution_api_result_{message_id.decode('utf-8')}",
                pickle.dumps({"status": "failed"}),
                ex=86400,
            )
        common.redis_client.xdel(stream_name, message_id)
        for file in Path("output").iterdir():
            if datetime.datetime.now().timestamp() - file.stat().st_mtime > 86400:
                file.unlink()

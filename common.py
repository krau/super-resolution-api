import pathlib
from dataclasses import dataclass

import cv2
import numpy as np
import redis
from logger import logger

from config import settings

try:
    logger.info(
        f"Connecting to Redis: {settings.get('redis_url', 'redis://localhost:6379')}"
    )
    redis_client = redis.from_url(settings.get("redis_url", "redis://localhost:6379"))
    logger.info(f"Redis ping: {redis_client.ping()}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    exit(1)


@dataclass
class ModelInfo:
    name: str = ""
    path: str = ""
    scale: int = 4
    algo: str = ""


BASE_STREAM_NAME = (
    "super_resolution_api_queue"
    if not settings.get("worker_id")
    else f"super_resolution_api_queue_{settings.get('worker_id')}"
)
WORKER_KEY_PREFIX = "super_resolution_api_worker_"
DISTRIBUTED_STREAM_NAME = "super_resolution_api_distributed_queue"
RESULT_KEY_PREFIX = (
    "super_resolution_api_result_"
    if not settings.get("worker_id")
    else f"super_resolution_api_result_{settings.get('worker_id')}_"
)
PROGRESS_TIMEOUT = settings.get("timeout", 30)
MAX_ALLOWED_TIMEOUT = settings.get("max_timeout", 300)
MAX_THREAD = settings.get("max_thread", 8)
MODEL_NAME_DEFAULT = "x4_Anime_6B-Official"
MODEL_NAME_X4_JP_ILLUSTRATION_FIX1 = "x4_JP_Illustration-fix1"
MODEL_NAME_X4_JP_ILLUSTRATION_FIX2 = "x4_JP_Illustration-fix2"
MODEL_NAME_X4_JP_ILLUSTRATION_FIX1_D = "x4_JP_Illustration-fix1-d"
MODEL_NAME_X4_ANIME_6B_OFFICIAL = "x4_Anime_6B-Official"


model_Anime_Official = ModelInfo(
    MODEL_NAME_X4_ANIME_6B_OFFICIAL,
    "models/x4_Anime_6B-Official.onnx",
    4,
    "real-esrgan",
)

model_JP_Illustration_fix1 = ModelInfo(
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX1,
    "models/x4_jp_Illustration-fix1.onnx",
    4,
    "real-hatgan",
)


model_JP_Illustration_fix2 = ModelInfo(
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX2,
    "models/x4_jp_Illustration-fix2.onnx",
    4,
    "real-esrgan",
)

model_JP_Illustration_fix1_d = ModelInfo(
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX1_D,
    "models/x4_jp_Illustration-fix1-d.onnx",
    4,
    "real-esrgan",
)

models = {
    MODEL_NAME_X4_ANIME_6B_OFFICIAL: model_Anime_Official,
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX1: model_JP_Illustration_fix1,
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX2: model_JP_Illustration_fix2,
    MODEL_NAME_X4_JP_ILLUSTRATION_FIX1_D: model_JP_Illustration_fix1_d,
}


def get_image_size(image_path: pathlib.Path) -> tuple[int, int]:
    """
    return: (width, height)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise Exception(f"Failed to load image: {image_path}")
    return img.shape[1], img.shape[0]


@dataclass
class TileInfo:
    x: int
    y: int
    filpath: pathlib.Path


def split_image(
    img_path: pathlib.Path,
    save_dir: pathlib.Path,
    grid_size: tuple[int, int],
    overlap: int = 16,
) -> list[TileInfo]:
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        raise Exception(f"Failed to load image: {img_path}")

    height, width = img.shape[:2]
    rows, cols = grid_size

    base_h = height // rows
    base_w = width // cols

    tiles_info = []

    for row in range(rows):
        for col in range(cols):
            x1 = max(0, col * base_w - overlap)
            y1 = max(0, row * base_h - overlap)
            x2 = min(width, (col + 1) * base_w + overlap)
            y2 = min(height, (row + 1) * base_h + overlap)

            tile = img[y1:y2, x1:x2]
            tile_name = f"{img_path.stem}_tile_{row}_{col}.png"
            tile_path = save_path / tile_name
            cv2.imwrite(str(tile_path), tile)
            tiles_info.append(TileInfo(col, row, tile_path))

    return tiles_info


def merge_sr_tiles(
    tiles: list[TileInfo],
    output: pathlib.Path,
    original_size: tuple[int, int],
    scale: int,
    overlap: int = 16,
):
    """
    合并超分辨率后的图块

    tiles: 超分辨率后的图块信息列表, 需要根据 filepath 读取图块, 根据 x, y 位置信息进行拼接
    output: 合并后的图片保存路径
    original_size: 原始图片的尺寸
    overlap 为原始图片切割时设定的重叠像素数
    scale 为超分辨率倍数
    """
    # Calculate output dimensions
    logger.debug(
        f"正在合并 {len(tiles)} 张超分辨率图块, 原尺寸: {original_size}, 缩放倍数: {scale}"
    )

    width, height = original_size
    out_width = width * scale
    out_height = height * scale
    output_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # Calculate base tile sizes
    rows = max([t.y for t in tiles]) + 1
    cols = max([t.x for t in tiles]) + 1
    base_h = height // rows
    base_w = width // cols

    # Scale dimensions
    scaled_base_h = base_h * scale
    scaled_base_w = base_w * scale
    scaled_overlap = overlap * scale

    for tile_info in tiles:
        # Read tile
        tile = cv2.imread(str(tile_info.filpath))
        if tile is None:
            raise Exception(f"Failed to load tile: {tile_info.filpath}")

        # Calculate positions
        x1 = max(0, tile_info.x * scaled_base_w - scaled_overlap)
        y1 = max(0, tile_info.y * scaled_base_h - scaled_overlap)
        x2 = min(out_width, (tile_info.x + 1) * scaled_base_w + scaled_overlap)
        y2 = min(out_height, (tile_info.y + 1) * scaled_base_h + scaled_overlap)

        # Calculate blend mask for overlapping regions
        h, w = y2 - y1, x2 - x1
        blend_mask = np.ones((h, w, 1), dtype=np.float32)

        # Apply feathering at edges
        if tile_info.x > 0:  # Left edge
            blend_mask[:, :scaled_overlap] = np.linspace(0, 1, scaled_overlap).reshape(
                1, -1, 1
            )
        if tile_info.x < cols - 1:  # Right edge
            blend_mask[:, -scaled_overlap:] = np.linspace(1, 0, scaled_overlap).reshape(
                1, -1, 1
            )
        if tile_info.y > 0:  # Top edge
            blend_mask[:scaled_overlap, :] *= np.linspace(0, 1, scaled_overlap).reshape(
                -1, 1, 1
            )
        if tile_info.y < rows - 1:  # Bottom edge
            blend_mask[-scaled_overlap:, :] *= np.linspace(
                1, 0, scaled_overlap
            ).reshape(-1, 1, 1)

        # Blend tiles
        output_img[y1:y2, x1:x2] = (
            output_img[y1:y2, x1:x2] * (1 - blend_mask)
            + tile[: y2 - y1, : x2 - x1] * blend_mask
        ).astype(np.uint8)

    cv2.imwrite(output.as_posix(), output_img)


def calculate_grid(image_width, image_height, workers):
    if workers <= 0:
        raise ValueError("Worker count must be positive")

    best_rows, best_cols = 1, workers
    min_aspect_diff = float("inf")

    for rows in range(1, workers + 1):
        if workers % rows == 0:
            cols = workers // rows
            tile_width = image_width / cols
            tile_height = image_height / rows
            aspect_ratio = max(tile_width, tile_height) / min(tile_width, tile_height)
            aspect_diff = aspect_ratio - 1

            if aspect_diff < min_aspect_diff:
                best_rows, best_cols = rows, cols
                min_aspect_diff = aspect_diff
    logger.debug(f"calculate_grid: {best_rows}x{best_cols}")
    return best_rows, best_cols

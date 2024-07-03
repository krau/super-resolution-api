from dataclasses import dataclass

import redis
from loguru import logger

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


STREAM_NAME = "super_resolution_api_queue"
PROGRESS_TIMEOUT = settings.get("timeout", 30)
MAX_ALLOWED_TIMEOUT = settings.get("max_timeout", 300)
MODEL_NAME_DEFAULT = "x4_JP_Illustration-fix1-d"
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

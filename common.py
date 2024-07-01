import redis
from loguru import logger

from config import settings

try:
    redis_client = redis.from_url(settings.get("redis_url", "redis://localhost:6379"))
    logger.info(f"Redis ping: {redis_client.ping()}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    exit(1)

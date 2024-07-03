import redis
from loguru import logger

from config import settings

try:
    logger.info(f"Connecting to Redis: {settings.get('redis_url', 'redis://localhost:6379')}")
    redis_client = redis.from_url(settings.get("redis_url", "redis://localhost:6379"))
    logger.info(f"Redis ping: {redis_client.ping()}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    exit(1)

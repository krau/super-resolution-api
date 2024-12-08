import sys
from loguru import logger

import config

logger.remove()

logger.add(sys.stdout, level=config.settings.get("log_level", "INFO"))

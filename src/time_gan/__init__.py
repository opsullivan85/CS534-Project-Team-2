# You probably don't need to put anything else in this file.
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

from .time_gan import TimeGan

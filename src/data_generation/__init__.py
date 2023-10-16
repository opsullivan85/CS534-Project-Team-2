# You probably don't need to put anything else in this file.
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

from .boid import *

train_path = "data/boid_log.csv"
test_path = "data/boid_log_test.csv"

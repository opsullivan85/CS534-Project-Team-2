# You probably don't need to put anything else in this file.
# logger configuration should be done here
import logging

logging.basicConfig(
    filename="src.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

from .gan_anomaly_detection import *
from .naive_bayes import *
from .sota_method3 import *
from .sota_method4 import *

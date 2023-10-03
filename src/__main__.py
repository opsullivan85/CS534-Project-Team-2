# This file is the code that is run when you run `python -m src`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

from src import gan_anomaly_detection
from src import sota_method2
from src import sota_method3
from src import sota_method4

gan_anomaly_detection.hello_sota1()
sota_method2.hello_sota2()
sota_method3.hello_sota3()
sota_method4.hello_sota4()
logger.debug("HEE")

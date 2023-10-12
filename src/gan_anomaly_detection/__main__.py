# This file is the code that is run when you run `python -m src.sota_method1`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src import gan_anomaly_detection, run_mnist

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

# gan_anomaly_detection.hello_sota1()

run_mnist.run(100, 0.1, "fm", 1, 1)

# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src.data_generation import get_data

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")


get_data(num_good_boids=150, num_faulty_boids=30, num_iterations=100)

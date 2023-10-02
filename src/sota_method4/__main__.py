# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src import sota_method4

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

sota_method4.hello_sota4()

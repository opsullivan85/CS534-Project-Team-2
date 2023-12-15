# This file is the code that is run when you run `python -m src.sota_method3`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src.LSTM import lstm_method

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

lstm_method()

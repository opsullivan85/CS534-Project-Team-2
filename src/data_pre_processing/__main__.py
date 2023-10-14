# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src import data_pre_processing
import src
from pathlib import Path

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

# This gets the data in the format we want
X, y = data_pre_processing.load_data()

print(X.shape, y.shape)

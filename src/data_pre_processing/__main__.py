# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src import data_pre_processing
import src
from pathlib import Path

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

data_path = Path(src.__file__).parent.parent / "data" / "boid_log.csv"

# This gets the data in the format we want
X, y = data_pre_processing.get_rolling_data(data_path, window_size=50, step_size=1)

print(X.shape, y.shape)

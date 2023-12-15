# This file is the code that is run when you run `python -m src`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

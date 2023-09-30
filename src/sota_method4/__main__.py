# This file is the code that is run when you run `python -m sota_method1`.
import logging
print("HERREEEEE")
logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

def test():
    print("test")
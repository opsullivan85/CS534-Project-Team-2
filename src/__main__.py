# This file is the code that is run when you run `python -m src`.
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

from src import sota_method1
from src import sota_method2
from src import sota_method3
from src import sota_method4
logger.debug(f"Loaded modules with relative import")

[print(i) for i in sota_method4.__dict__]

print()
print(sys.path)

print()
sota_method4.test()

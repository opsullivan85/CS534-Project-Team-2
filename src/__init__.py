import logging

logging.basicConfig(
    filename="src.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

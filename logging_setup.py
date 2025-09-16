import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Environment / Config Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE   = "logs/app.log"

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(threadName)s:%(thread)d] [%(name)s] [%(pathname)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s"
)

def setup_logging():
    """Set up both console + rotating file loggers."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    root_logger.info("Logging setup complete.")

# Default Logger for Imports
setup_logging()
logger = logging.getLogger("app")
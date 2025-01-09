import logging
import os

# Configure logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join("logs", "system.log")

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler(),       # Log to console
    ],
)

def get_logger(name):
    return logging.getLogger(name)

import logging
import os
import sys
import tempfile


def setup_logging(log_level=logging.INFO, log_filename="result_companion.log"):
    """
    Configures logging to output to both stdout and a specified log file in the system's temporary directory.

    Parameters:
    - log_level: The logging level (e.g., logging.DEBUG, logging.INFO).
    - log_filename: The name of the log file.
    """
    # Determine the system's temporary directory
    temp_dir = tempfile.gettempdir()
    log_file_path = os.path.join(temp_dir, log_filename)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define the log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the location of the log file
    logger.debug(f"Logging to file: {log_file_path}")

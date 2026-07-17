import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "DeepCode", log_level: int = logging.INFO, log_dir: str = "logs") -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: logging.INFO)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent adding handlers multiple times if logger is already configured
    if logger.handlers:
        return logger
        
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name.lower()}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
            
    return logger

# Default logger instance
logger = setup_logger()

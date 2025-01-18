import logging, os
from datetime import datetime

class Logger:
    def __init__(self, log_folder, log_file, enable_console_output=False):
        """
        Initializes the logger with both file and optional console handlers.
        Args:
            log_folder (str): The folder where the log file will be stored.
            log_file (str): The base filename of the log file, which will be appended with a datetime stamp.
            enable_console_output (bool): Flag to enable or disable console logging.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Ensure the directory exists, create if it does not
        os.makedirs(log_folder, exist_ok=True)

        # Combine directory path and filename with datetime stamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_folder, f"{log_file}_{timestamp}.log")

        # Create file handler with datetime stamp in filename
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # Optionally create console handler
        if enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

    def write(self, message, level='info'):
        """
        Logs a message at the specified level.
        
        Args:
            message (str): The message to log.
            level (str): The logging level ('info', 'warning', 'error', 'critical').
        """
        {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(level, self.logger.info)(message)
import logging, os
from .utils import join_paths


class Logger:
    def __init__(self, config, enable_console_output=False):
        """
        Initializes the logger with a file handler.
        Args:
            log_folder (str): The folder where the log file will be stored.
            log_file (str): The base filename of the log file, which will be appended with a datetime stamp.
        """
        self.enable_console_output = enable_console_output
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Ensure the directory exists, create if it does not
        os.makedirs(config.log_folder, exist_ok=True)

        # Combine directory path and filename with datetime stamp
        self.log_file_path = join_paths(config.log_folder, config.log_file_name)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False


    def close(self):
        # Close all handlers associated with the logger
        print('closing logger handlers')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


    def write(self, message, level='info', output_to_console=True):
        """
        Logs a message at the specified level and optionally outputs to the console.
        
        Args:
            message (str): The message to log.
            level (str): The logging level ('info', 'warning', 'error', 'critical').
            output_to_console (bool): Flag to enable or disable console output for this specific message.
        """
        # Log the message at the specified level
        log_method = {
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical
        }.get(level, self.logger.info)
        
        log_method(message)
        
        # If requested, output the message to the console
        if self.enable_console_output and output_to_console:
            print(message)
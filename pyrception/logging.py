from pyrception import conf


class Logging:
    '''
    Auxiliary class containing methods for logging.
    '''

    def __init__(
        self,
        source: str,
    ):

        self.source = source

    def trace(self, message: str):
        """
        Print TRACE-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.trace(f"{self.source} | {message}")

    def debug(self, message: str):
        """
        Print DEBUG-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.debug(f"{self.source} | {message}")

    def info(self, message: str):
        """
        Print INFO-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.info(f"{self.source} | {message}")

    def success(self, message: str):
        """
        Print SUCCESS-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.success(f"{self.source} | {message}")

    def warning(self, message: str):
        """
        Print WARNING-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.warning(f"{self.source} | {message}")

    def error(self, message: str):
        """
        Print ERROR-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.error(f"{self.source} | {message}")

    def critical(self, message: str):
        """
        Print CRITICAL-level messages.

        Args:
            message (str):
                The string to print.
        """
        conf.logger.critical(f"{self.source} | {message}")

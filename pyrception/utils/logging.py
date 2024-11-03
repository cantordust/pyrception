# --------------------------------------
import typing as tp

# --------------------------------------
from pyrception import conf


class Logger:
    """
    Auxiliary class containing methods for logging.
    """

    def __init__(
        self,
        source: str,
        notifier: tp.Callable = None,
    ):
        """
        A simple class that takes care of logging and notifications.

        Args:
            source (str):
                The source of the message (usually a layer instance).

            notifier (tp.Callable):
                A progress notification function
        """

        self.source = source
        self.notifier = notifier

    def notify(
        self,
        message: str,
        value: int = 0,
    ):
        '''
        Send an optional notification via a callback.

        Args:
            message (str):
                Message.

            value (int, optional):
                Progress indicator. Defaults to 0.
        '''

        if self.notifier is not None:
            self.notifier(message, value)

    def trace(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print TRACE-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.trace(f"{self.source} | {message}")
        self.notify(message, value)

    def debug(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print DEBUG-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.debug(f"{self.source} | {message}")
        self.notify(message, value)

    def info(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print INFO-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.info(f"{self.source} | {message}")
        self.notify(message, value)

    def success(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print SUCCESS-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.success(f"{self.source} | {message}")
        self.notify(message, value)

    def warning(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print WARNING-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.warning(f"{self.source} | {message}")
        self.notify(message, value)

    def error(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print ERROR-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.error(f"{self.source} | {message}")
        self.notify(message, value)

    def critical(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Print CRITICAL-level messages.

        Args:
            message (str):
                The string to print.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """
        conf.logger.critical(f"{self.source} | {message}")
        self.notify(message, value)

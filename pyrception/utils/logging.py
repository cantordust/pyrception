# --------------------------------------
from typing import Callable

# --------------------------------------
from pyrception import conf
from pyrception import logger


class LoggingMixin:
    """
    Auxiliary class containing methods for logging.
    """

    def __init__(
        self,
        source: str,
        notifier: Callable = None,
    ):
        """
        A simple class that takes care of logging and notifications.

        Args:
            source (str):
                The source of the message (usually a layer instance).

            notifier (tp.Callable):
                A notification function (used for the GUI).
        """

        self.source = source
        if notifier is None:
            self.notifier = lambda m, v: None
        else:
            self.notifier = self.notify

    def notify(
        self,
        message: str,
        value: int = 0,
    ):
        """
        Send an optional notification via a callback.

        Args:
            message (str):
                Message.

            value (int, optional):
                Progress indicator. Defaults to 0.
        """

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
        logger.trace(f"{self.source} | {message}")
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
        logger.debug(f"{self.source} | {message}")
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
        logger.info(f"{self.source} | {message}")
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
        logger.success(f"{self.source} | {message}")
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
        logger.warning(f"{self.source} | {message}")
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
        logger.error(f"{self.source} | {message}")
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
        logger.critical(f"{self.source} | {message}")
        self.notify(message, value)

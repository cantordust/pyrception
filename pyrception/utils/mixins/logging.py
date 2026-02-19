from loguru import logger
from collections.abc import Callable
from pyrception import conf


class LoggingMixin:

    def __init__(
        self,
        source: str | None = None,
        notifier: Callable | None = None,
        *args,
        **kwargs,
    ):
        """
        A simple class that takes care of logging and notifications.

        Args:
            source:
                The source of the message (usually a layer instance).

            notifier:
                A notification function.
                Useful for communicating with a GUI (WIP).
        """
        super().__init__(*args, **kwargs)

        self.source = source or self.__class__.__name__
        self.notifier = notifier
        self.logger = logger.bind(source=f" | {self.source}")

        if isinstance(notifier, Callable):
            self.notifier = self.logger.add(notifier)

    def verbose(self, level: str = "INFO"):
        # WIP Not a great solution, to be redesigned.
        return self.logger.level(conf.log.level).no <= self.logger.level(level).no

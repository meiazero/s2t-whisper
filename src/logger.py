from datetime import datetime


class Logger:
    """Handles logging messages with timestamps for process tracking."""

    @staticmethod
    def log(message: str) -> None:
        """Logs a message with timestamp."""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} {message}")

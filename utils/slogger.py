import logging


class SLogger:
    def __init__(self):
        # Tạo logger
        self.slogger = logging.getLogger("slogger")
        self.slogger.setLevel(logging.DEBUG)

        # Thêm một handler mặc định (console)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Định dạng log
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Gắn handler vào logger
        self.slogger.addHandler(console_handler)

    def logger(self):
        return self.slogger

    def registerPublisher(self, publisher):
        """
        Thêm một publisher (handler) khác vào logger.
        Ví dụ: FileHandler, HTTPHandler, hoặc custom handler.
        """
        self.slogger.addHandler(publisher)


# Tạo instance của logger
SLOG = SLogger().logger()
import logging 

class Logger:
    
    logger = None
    
    @classmethod
    def add_console_handler(cls):
        cls.logger = cls.get()
        handler = logging.StreamHandler()
        cls.logger.addHandler(handler)
    
    @classmethod
    def create_file_logger(cls, filename):
        cls.logger = cls.get()
        handler = logging.FileHandler(filename)
        cls.logger.addHandler(handler)
    
    @classmethod
    def get(cls) -> logging.Logger:
        if cls.logger is None:
            cls.logger = logging.getLogger("1d_detect")
        return cls.logger
            
            
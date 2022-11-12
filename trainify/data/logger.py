import time
import logging
import sys
import colorlog
import os
from trainify.data.backend_logger_handler import BackendHandler

# 是否开启log日志
LOG_ENABLED = True
# 是否输出到控制台
LOG_TO_CONSOLE = True
# 是否输出到文件
LOG_TO_FILE = True
# 日志等级
LOG_LEVEL = logging.DEBUG
# 每条日志输出格式
LOG_FILE_FORMAT = '[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s'
LOG_CONSOLE_FORMAT = '%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(' \
                     'levelname)s] : %(message)s '
LOG_CONSOLE_COLOR = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


class Logger(object):
    def __init__(self, log_path, channel):
        self.loggers = {}
        self.enabled = LOG_ENABLED  # 是否开启日志
        self.is_console = LOG_TO_CONSOLE  # 是否输出到控制台
        self.is_file = LOG_TO_FILE  # 是否输出到文件
        self.path = log_path  # 日志文件路径
        self.level = LOG_LEVEL  # 日志级别
        # self.format = LOG_FORMAT  # 每条日志输出格式
        self.backup_count = 1000000  # 最多存放日志的数量
        self.channel = channel if channel is not None else False

    def create_logger(self, logger_name=None):
        if logger_name is None:
            logger_name = 'log_' + time.strftime("_%Y%m%d_%H%M%S", time.localtime())

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print(logger_name)
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        if self.enabled and self.is_console:
            console_handler = logging.StreamHandler(sys.stdout)
            # console_handler.setLevel(self.level)
            console_formatter = colorlog.ColoredFormatter(
                fmt=LOG_CONSOLE_FORMAT,
                datefmt='%Y-%m-%d  %H:%M:%S',
                log_colors=LOG_CONSOLE_COLOR
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        if self.enabled and self.is_file:
            file_handler = logging.FileHandler(self.path + '/' + logger_name + '.log')
            # file_handler.setLevel(self.level)
            file_handler.setFormatter(logging.Formatter(LOG_FILE_FORMAT))
            logger.addHandler(file_handler)

        if self.enabled and self.channel:
            back_handler = BackendHandler(self.channel)
            back_handler.setFormatter(logging.Formatter(LOG_FILE_FORMAT))
            logger.addHandler(back_handler)
        return logger

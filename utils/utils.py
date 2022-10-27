import logging
import os
from os.path import join

from transformers.trainer_utils import is_main_process


def get_logger(logger_name: str, exp_dir: str = None, rank=-1):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if is_main_process(rank) else logging.WARN)

    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Repeated calls to `addHandler` result in repeated log output
    if len(logger.handlers) == 0:
        formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        if exp_dir is None:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
        else:
            os.makedirs(exp_dir, exist_ok=True)
            handler = logging.FileHandler(join(exp_dir, "log.txt"), encoding="utf8")
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

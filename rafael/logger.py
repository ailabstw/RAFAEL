import logging
from logging.handlers import QueueListener
import multiprocessing as mp
from datetime import datetime


def setup_logger(log_path = None, levels = logging.DEBUG, log_queue: mp.Queue = None) -> logging.Logger:
    # turn off info log in package
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('plotnine').setLevel(logging.WARNING)

    logger = logging.getLogger()
    # Config.LOGGER_NAME
    logger.setLevel(levels)
    formatter = logging.Formatter(
        f"%(processName)s| %(levelname)1.1s %(asctime)s %(module)s:%(lineno)d| %(funcName)s| %(message)s",
        datefmt='%H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(levels)
    ch.setFormatter(formatter)
    if log_queue is not None:
        listener = QueueListener(log_queue, ch)
        listener.start()

    logger.addHandler(ch)

    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(levels)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if log_queue is not None:
        listener.start()

    logging.info(f"Start Date -> {datetime.now()}")
    return logger

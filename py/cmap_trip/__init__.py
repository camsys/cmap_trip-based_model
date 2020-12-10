# CMAP Trip-based model tools

__version__ = '20.12.1'

import logging
LOGGER_NAME = 'CMAP'
LOG_FORMAT = '[{elapsedTime}] {name:s}:{levelname:s}: {message:s}'

def format_elapsed_time(duration_milliseconds):
    hours, rem = divmod(duration_milliseconds/1000, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    else:
        return ("{:0>2}:{:05.2f}".format(int(minutes),seconds))


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        record.elapsedTime = format_elapsed_time(record.relativeCreated)
        return super(ElapsedTimeFormatter, self).format(record)


def log_to_stderr(level=30):
    """
    Turn on logging and add a handler which prints to stderr

    Parameters
    ----------
    level : int
        minimum level of the messages that will be logged
    """

    logger = logging.getLogger(LOGGER_NAME)

    # avoid creation of multiple stream handlers for logging to console
    for entry in logger.handlers:
        if isinstance(entry, logging.StreamHandler) and (entry.formatter._fmt == LOG_FORMAT):
            return logger

    formatter = ElapsedTimeFormatter(LOG_FORMAT, style='{')
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(level)

    return logger


from .filepaths import set_database_dir, set_skims_dir, set_cache_dir

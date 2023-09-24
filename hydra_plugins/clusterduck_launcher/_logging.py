from __future__ import annotations

import logging.config

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "clusterduck_basic": {"format": "[%(asctime)s][Clusterduck] - %(message)s"}
    },
    "handlers": {
        "clusterduck_out": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "clusterduck_basic",
            "stream": "ext://sys.stdout",
        },
        "clusterduck_err": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "clusterduck_basic",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "clusterduck": {
            "handlers": ["clusterduck_err", "clusterduck_out"],
            # use only these handlers and not any defined by Hydra logging for the root logger
            # TODO: why doesn't submitit logging config need this?
            "propagate": False,
        }
    },
}


def configure_log(
    verbose_config: bool = False,
) -> None:
    # TODO: add logging config to plugin config (preferably as a file)
    # refer to hydra logging and job_logging for how this is done

    CONFIG["loggers"]["clusterduck"]["level"] = "DEBUG" if verbose_config else "INFO"
    logging.config.dictConfig(CONFIG)


def get_logger() -> logging.Logger:
    return logging.getLogger("clusterduck")

from __future__ import annotations

import logging.config


def configure_log(
    verbose: bool,
) -> None:
    # TODO: add logging config to plugin config (preferably as a file)
    # refer to hydra logging and job_logging for how this is done

    CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "clusterduck_verbose": {
                "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            },
            "clusterduck_basic": {"format": "[%(asctime)s][clusterduck] %(message)s"},
        },
        "handlers": {
            "clusterduck_out": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "clusterduck_verbose" if verbose else "clusterduck_basic",
                "stream": "ext://sys.stdout",
            },
            "clusterduck_err": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "clusterduck_verbose" if verbose else "clusterduck_basic",
                "stream": "ext://sys.stderr",
            },
            # TODO: add a dedicated file output for clusterduck logging?
            # "clusterduck_file": {
            #     "class": "logging.FileHandler",
            #     "level": "DEBUG",
            #     "formatter": "clusterduck_verbose" if verbose else "clusterduck_basic",
            #     "filename": f"{hydra_sweep_dir}/.clusterduck/%j.log",
            # },
        },
        "loggers": {
            "clusterduck": {
                "handlers": ["clusterduck_err", "clusterduck_out"],
                "level": "DEBUG" if verbose else "INFO",
                # if verbose is on, log to both SLURM log and job log
                # if verbose is off, log only to SLURM log
                "propagate": verbose,
            }
        },
    }
    logging.config.dictConfig(CONFIG)


def get_logger(name: str) -> logging.Logger:
    names = name.split(".")
    if len(names) == 1:
        pretty_name = f"clusterduck.{name}"
    else:
        qualifiers = names[names.index("clusterduck_launcher") + 1 :]
        qualifiers = [qualifier.strip("_") for qualifier in qualifiers]
        pretty_name = ".".join(["clusterduck"] + qualifiers)
    return logging.getLogger(pretty_name)

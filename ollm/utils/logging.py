import json
from pathlib import Path

from absl import flags, logging


def setup_logging(
    dir_: Path | str,
    log_file_name: str = "logging",
    flags: flags.FlagValues | None = None,
):
    dir_ = Path(dir_)
    log_dir = dir_ / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Set up logging
    logging.get_absl_handler().use_absl_log_file(log_file_name, str(log_dir))
    logging.set_stderrthreshold("info")
    logging.set_verbosity("debug")
    print(f"Logging to {log_dir}")

    if flags is not None:
        log_flags(dir_ / f"{log_file_name}_flags.json", flags)
        logging.info("Flags: %s", json.dumps(flags.flag_values_dict(), indent=2))


def log_flags(file_name: Path | str, flags: flags.FlagValues):
    dict_ = flags.flag_values_dict()
    with open(file_name, "w") as f:
        json.dump(dict_, f, indent=2, default=str)

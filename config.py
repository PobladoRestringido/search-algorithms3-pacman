import os
from dotenv import load_dotenv

load_dotenv()


def _int(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Public config values
PACMAN_SEED = _int("PACMAN_SEED", None)

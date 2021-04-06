import os
import subprocess


def exe(cmd, args):
    """Execute a command."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        return (
            subprocess.check_output([cmd, *args], stderr=subprocess.PIPE)
            .decode("ascii")
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise
    return None

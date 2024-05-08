# hacked together using excerpts from dev.py
import os
import sys
import contextlib
from rich.console import Console
from rich.theme import Theme

PROJECT_MODULE = "scipy"
PROJECT_ROOT_FILES = ['scipy', 'LICENSE.txt', 'meson.build']

console_theme = Theme({
    "cmd": "italic gray50",
})

if sys.platform == 'win32':
    class EMOJI:
        cmd = ">"
else:
    class EMOJI:
        cmd = ":computer:"

def emit_cmdstr(cmd):
    """Print the command that's being run to stdout

    Note: cannot use this in the below tasks (yet), because as is these command
    strings are always echoed to the console, even if the command isn't run
    (but for example the `build` command is run).
    """
    console = Console(theme=console_theme)
    # The [cmd] square brackets controls the font styling, typically in italics
    # to differentiate it from other stdout content
    console.print(f"{EMOJI.cmd} [cmd] {cmd}")


@contextlib.contextmanager
def working_dir(new_dir):
    current_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(current_dir)


def __main__():
    try:
        import mypy.api
    except ImportError as e:
        raise RuntimeError(
            "Mypy not found. Please install it by running "
            "pip install -r mypy_requirements.txt from the repo root"
        ) from e

    dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    config = os.path.abspath(os.path.join(dir, "mypy.ini"))
    check_path = PROJECT_MODULE

    with working_dir(dir):
        # By default mypy won't color the output since it isn't being
        # invoked from a tty.
        os.environ['MYPY_FORCE_COLOR'] = '1'
        # Change to the site directory to make sure mypy doesn't pick
        # up any type stubs in the source tree.
        emit_cmdstr(f"mypy.api.run --config-file {config} {check_path}")
        report, errors, status = mypy.api.run([
            "--config-file",
            str(config),
            check_path,
        ])
    print(report, end='')
    print(errors, end='', file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(__main__())

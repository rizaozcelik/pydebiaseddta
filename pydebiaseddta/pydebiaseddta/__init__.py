"""
Docstring
"""
import pathlib

_PACKAGE_PATH = str(pathlib.Path(__file__).parent.resolve())


def __getattr__(name):
    if name == 'package_path':
        return _PACKAGE_PATH
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


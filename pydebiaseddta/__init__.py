import pathlib
import os

# For accurate randomization control.
os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)
os.environ["PYTHONHASHSEED"] = str(0)

_PACKAGE_PATH = str(pathlib.Path(__file__).parent.resolve())


def __getattr__(name):
    if name == 'package_path':
        return _PACKAGE_PATH
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


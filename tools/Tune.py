"""Compatibility entry point for Kernel Launcher's tuning script."""

import os
import runpy
import sys


kernel_launcher_python = os.environ.get("KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON")
if not kernel_launcher_python:
    raise RuntimeError("KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON is not configured")

sys.path.insert(0, kernel_launcher_python)

from kernel_launcher.reader import TuningProblem


_tune_options = TuningProblem._tune_options


def patched_tune_options(self, *args, **kwargs):
    extra_params, options = _tune_options(self, *args, **kwargs)
    # Recent Kernel Tuner versions inspect callable restrictions to build a
    # constraint solver. Kernel Launcher supplies an empty callback even when
    # the pragma source declares no restrictions, which becomes an invalid
    # zero-argument constraint. Omitting it preserves the intended semantics.
    if not self.space.restrictions:
        options.pop("restrictions", None)
    return extra_params, options


TuningProblem._tune_options = patched_tune_options
runpy.run_path(os.path.join(kernel_launcher_python, "tune.py"), run_name="__main__")

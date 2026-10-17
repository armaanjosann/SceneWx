"""Registers SceneWxPairedDataset with NAFNet's BasicSR.

Copy to <NAFNet>/basicsr/data/scenewx_dataset.py. The scenewx package is found through SCENEWX_REPO
(the folder containing scenewx/) or PYTHONPATH.
"""
import os
import sys

_REPO = os.environ.get('SCENEWX_REPO')
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scenewx.scenewx_dataset import SceneWxPairedDataset   # noqa: E402

__all__ = ['SceneWxPairedDataset']

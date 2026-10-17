"""Paths, all read from environment variables (see README).

Data paths have no default and raise an error naming the missing variable. SCENEWX_CALIBRATION,
SCENEWX_FRAMES and SCENEWX_OUTPUTS default to calibration/calibration_cells.json,
calibration/frames_train_frozen.txt and outputs/ in the repository.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HELP = {
    'SCENEWX_CITYSCAPES': 'the Cityscapes leftImg8bit folder (with train/ val/ test/ inside)',
    'SCENEWX_GTFINE': 'the Cityscapes gtFine folder',
    'SCENEWX_DEPTH_CACHE': 'a folder for the metric depth cache (python -m scenewx.depth --precompute train)',
    'SCENEWX_CSD_MASKS': 'the folder of CSD snow masks (*.tif)',
    'SCENEWX_ACDC': 'the ACDC root (contains rgb_anon/ and gt/)',
    'SCENEWX_FOGGY_ZURICH': 'the Foggy Zurich test root (contains RGB/ and gt_labelIds/)',
    'SCENEWX_FOGGY_DRIVING': 'the Foggy Driving root (contains leftImg8bit/ and gtFine/ or gtCoarse/)',
    'SCENEWX_GTRAIN': 'the GT-RAIN test folder (scene folders with -C- and -R- frames)',
    'SCENEWX_WEATHERBENCH': 'the WeatherBench root (<weather>/test/input and <weather>/test/target)',
    'NAFNET_ROOT': 'a checkout of https://github.com/megvii-research/NAFNet',
}


def get(name):
    """Value of a required path variable; raises with a hint if it is not set."""
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f'environment variable {name} is not set: point it at '
                           f'{_HELP.get(name, "the required path")}')
    return v


def cityscapes():
    return get('SCENEWX_CITYSCAPES')


def gtfine():
    return get('SCENEWX_GTFINE')


def depth_cache():
    return get('SCENEWX_DEPTH_CACHE')


def csd_masks():
    return get('SCENEWX_CSD_MASKS')


def calibration():
    return os.environ.get('SCENEWX_CALIBRATION') or os.path.join(REPO, 'calibration', 'calibration_cells.json')


def frames_file():
    return os.environ.get('SCENEWX_FRAMES') or os.path.join(REPO, 'calibration', 'frames_train_frozen.txt')


def outputs():
    return os.environ.get('SCENEWX_OUTPUTS') or os.path.join(REPO, 'outputs')

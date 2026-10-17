# SceneWx

Code for the honours thesis *Beyond Realism: Isolating the Properties of Synthetic Weather that
Transfer to Real Driving Scenes* (Armaan Singh Josan, The University of Western Australia, 2026).

SceneWx renders fog, rain and snow onto Cityscapes frames under conditions that each change one
property of the weather, all matched to the same severity. A restoration network is trained on each
condition and tested on real adverse-weather images.

No data, renders or trained weights are included (Cityscapes does not allow redistribution).

## Install

    pip install -r requirements.txt

Training also needs [NAFNet](https://github.com/megvii-research/NAFNet), which provides `basicsr`.
Depth Anything V2 and SegFormer weights download from Hugging Face on first use.

## Data

Download [Cityscapes](https://www.cityscapes-dataset.com/) and the
[CSD snow masks](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet). For
evaluation: [ACDC](https://acdc.vision.ee.ethz.ch/), Foggy Zurich, Foggy Driving,
[GT-RAIN](https://github.com/UCLA-VMG/GT-RAIN) and WeatherBench.

Paths are set with environment variables, listed in `scenewx/config.py`. The main ones:

    export SCENEWX_CITYSCAPES=/path/to/leftImg8bit
    export SCENEWX_GTFINE=/path/to/gtFine
    export SCENEWX_CSD_MASKS=/path/to/csd_masks
    export SCENEWX_DEPTH_CACHE=/path/to/depth_cache

## Usage

    python -m scenewx.depth --precompute train                 # metric depth, once per split
    python examples/render_example.py --all                    # render one frame in every condition
    python -m scenewx.calibrate_cells --sample 12              # optional: refit severity matching

Train with NAFNet using a config from `configs/` (setup in `configs/README.md`):

    python basicsr/train.py -opt $SCENEWX_REPO/configs/nafnet/nafnet_fog_full_moderate_c256_i60000_s10.yml

Evaluate with the scripts in `evaluation/` (restore-then-segment, paired PSNR, behaviour). The
thesis figures are in `figures/`.

## Conditions

| Name | Haze | Particles | Depth |
|---|---|---|---|
| `base` | uniform | fixed size | raw |
| `P1` | by distance | fixed size | raw |
| `P2` | uniform | by distance | raw |
| `full` | by distance | by distance | raw |
| `P4` | by distance | by distance | corrected with scene labels |
| `scrambled` | by distance | by distance | flipped (control) |
| `noise`, `identity` | controls | | |

Levels: `light`, `moderate`, `heavy`, `mixed`. Fog has no particles, so it skips `P2`. The frozen
severity settings used in the thesis are in `calibration/calibration_cells.json`.

## Notes

- Set `PYTHONHASHSEED` for identical training crops between runs.
- Corrected depth differs with and without `opencv-contrib-python`.

## Citation

    @thesis{josan2026beyond,
      author = {Josan, Armaan Singh},
      title  = {Beyond Realism: Isolating the Properties of Synthetic Weather that Transfer to Real Driving Scenes},
      school = {The University of Western Australia},
      type   = {Honours thesis},
      year   = {2026}
    }

## Licence

MIT (`LICENSE`). Third-party code, models and data keep their own terms; see `THIRD_PARTY.md`.

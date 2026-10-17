# Training option files

`nafnet/` and `restormer/` hold the BasicSR option files of the final study (35 NAFNet, 8 Restormer;
crop 256, 60k iterations, batch 4, one GPU). The thesis reports 43 NAFNet and 14 Restormer networks:
the remaining runs repeat some of these conditions with another seed, and such files differ only in
`seed`, `manual_seed` and `name`. Write them with

    python configs/make_configs.py --arch nafnet --weather fog --cells base full --seeds 11 12

`python configs/make_configs.py --thesis` regenerates the shipped files. The files contain no paths:
the dataset reads its data locations from the `SCENEWX_*` environment variables (see README.md), and
BasicSR writes checkpoints to `<NAFNet>/experiments/<name>/models/`.

## Registering SceneWx with NAFNet's BasicSR

NAFNet's `basicsr` imports every `basicsr/data/*_dataset.py` and every `basicsr/models/archs/*_arch.py`
at start-up and looks up `type:` names on those modules. To register SceneWx, copy two files:

    cp basicsr_plugins/scenewx_dataset.py  $NAFNET_ROOT/basicsr/data/scenewx_dataset.py
    cp basicsr_plugins/Restormer_arch.py   $NAFNET_ROOT/basicsr/models/archs/Restormer_arch.py   # Restormer runs only

The copied `scenewx_dataset.py` is a three-line shim that imports `SceneWxPairedDataset` from this
repository, which it finds through `SCENEWX_REPO` (the folder containing `scenewx/`) or `PYTHONPATH`.

Then, from the NAFNet folder:

    export SCENEWX_REPO=/path/to/SceneWx
    export SCENEWX_CITYSCAPES=... SCENEWX_GTFINE=... SCENEWX_DEPTH_CACHE=... SCENEWX_CSD_MASKS=...
    python basicsr/train.py -opt $SCENEWX_REPO/configs/nafnet/nafnet_fog_full_moderate_c256_i60000_s10.yml --launcher none

The configs have no `datasets.val` block; evaluation is done after training with the scripts in
`evaluation/`.

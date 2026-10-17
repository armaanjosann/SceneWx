"""Loading trained restorers and running them on images. Shared by every evaluation script.

NAFNet comes from a NAFNet checkout (NAFNET_ROOT); the Restormer used in the thesis is the
einops-free re-implementation in basicsr_plugins/Restormer_arch.py.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'basicsr_plugins'))
from scenewx import config                                                    # noqa: E402

# NAFNet width 32, as trained in the thesis
NAFNET_CFG = dict(img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
# Restormer "light", as trained in the thesis (configs/restormer)
RESTORMER_LIGHT = dict(dim=32, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2, heads=(1, 2, 4, 8))


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_restorer(ckpt, dev, width=None, arch='nafnet'):
    """A BasicSR checkpoint (net_g_*.pth) as an eval-mode network on `dev`."""
    if arch == 'restormer':
        from Restormer_arch import Restormer
        net = Restormer(**RESTORMER_LIGHT)
    elif arch == 'nafnet':
        root = config.get('NAFNET_ROOT')
        if root not in sys.path:
            sys.path.insert(0, root)
        from basicsr.models.archs.NAFNet_arch import NAFNet
        net = NAFNet(**(dict(NAFNET_CFG, width=width) if width else NAFNET_CFG))
    else:
        raise ValueError(f'unknown architecture {arch!r} (nafnet | restormer)')
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd['params'] if 'params' in sd else sd, strict=True)
    return net.eval().to(dev)


def restore(net, bgr, dev):
    """uint8 BGR in, uint8 BGR out. Pads to a multiple of 16 (reflect) and crops back."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.
    h, w = rgb.shape[:2]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(dev)
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        y = net(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        y = y[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return np.round(y[:, :, ::-1] * 255).astype(np.uint8)


def mean_abs(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-6)))

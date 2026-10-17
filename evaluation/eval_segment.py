#!/usr/bin/env python
"""Restore-then-segment evaluation on real weather: 19-class mIoU of a frozen SegFormer-B0 (Cityscapes).

The first row is the weather image segmented directly. --root defaults to SCENEWX_ACDC,
SCENEWX_FOGGY_ZURICH or SCENEWX_FOGGY_DRIVING depending on --layout.

  python evaluation/eval_segment.py --layout acdc --acdc-cond fog \
      --models full=experiments/nafnet_fog_full_moderate_c256_i60000_s10/models/net_g_60000.pth \
      --out outputs/eval/fog_full_acdc.json
"""
import argparse
import glob
import json
import os
import time
from os import path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Apple MPS: release cached GPU buffers earlier (memory policy only; outputs are unchanged).
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.8')
from restorers import load_restorer, default_device                           # noqa: E402
from scenewx import config                                                    # noqa: E402
from transformers import SegformerForSemanticSegmentation                     # noqa: E402

SEG_MODEL = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
N_CLASSES = 19
# Cityscapes labelId -> trainId (19 evaluation classes); everything else ignored.
LABELID_TO_TRAINID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                      23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
               'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle']
ROOT_VAR = {'acdc': 'SCENEWX_ACDC', 'zurich': 'SCENEWX_FOGGY_ZURICH', 'foggy_driving': 'SCENEWX_FOGGY_DRIVING'}


def to_trainid(label_ids):
    out = np.full(label_ids.shape, 255, np.uint8)
    for lid, tid in LABELID_TO_TRAINID.items():
        out[label_ids == lid] = tid
    return out


def segment(seg, img_rgb01, dev):
    """img_rgb01: HxWx3 float [0,1]. Returns HxW int trainId prediction."""
    h, w = img_rgb01.shape[:2]
    x = (img_rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        logits = seg(pixel_values=x).logits          # 1,19,h/4,w/4
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def confusion(pred, gt, cm):
    m = gt != 255
    if m.sum() == 0:
        return
    idx = gt[m].astype(np.int64) * N_CLASSES + pred[m].astype(np.int64)
    cm += np.bincount(idx, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def miou(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    with np.errstate(invalid='ignore'):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def gather_pairs(root, layout, acdc_cond='fog', acdc_split='val', limit=0):
    """(stem, image_path, label_path) triples for one real-benchmark layout."""
    pairs = []; imgs = []
    if layout == 'foggy_driving':
        imgs = sorted(glob.glob(osp.join(root, 'leftImg8bit', '**', '*_leftImg8bit.png'), recursive=True))
        for ip in imgs:
            stem = osp.basename(ip).replace('_leftImg8bit.png', '')
            gt = None
            for sub in ('gtFine', 'gtCoarse'):
                hits = glob.glob(osp.join(root, sub, '**', stem + f'_{sub}_labelIds.png'), recursive=True)
                if hits:
                    gt = hits[0]; break
            if gt:
                pairs.append((stem, ip, gt))
    elif layout == 'acdc':
        # ACDC: rgb_anon/<cond>/<split>/<seq>/<name>_rgb_anon.png and gt/<cond>/<split>/<seq>/<name>_gt_labelIds.png
        imgs = sorted(glob.glob(osp.join(root, 'rgb_anon', acdc_cond, acdc_split, '*', '*_rgb_anon.png')))
        for ip in imgs:
            gp = ip.replace(osp.join('rgb_anon', acdc_cond), osp.join('gt', acdc_cond)).replace('_rgb_anon.png', '_gt_labelIds.png')
            if osp.isfile(gp):
                pairs.append((osp.basename(ip)[:-13], ip, gp))
    elif layout == 'zurich':
        # Foggy Zurich: RGB/<seq>/<name>.png and gt_labelIds/<seq>/<name>.png (Cityscapes labelIds)
        imgs = sorted(glob.glob(osp.join(root, '**', 'RGB', '*', '*.png'), recursive=True))
        for ip in imgs:
            seq, name = osp.basename(osp.dirname(ip)), osp.basename(ip)
            gp = ip.replace(osp.join('RGB', seq, name), osp.join('gt_labelIds', seq, name))
            if osp.isfile(gp):
                pairs.append((f'{seq}/{name[:-4]}', ip, gp))
    else:
        raise ValueError(layout)
    if limit:
        pairs = pairs[:limit]
    return pairs, len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None, help='dataset root (default: from the layout\'s environment variable)')
    ap.add_argument('--layout', default='acdc', choices=['acdc', 'zurich', 'foggy_driving'])
    ap.add_argument('--acdc-cond', default='fog', choices=['fog', 'rain', 'snow'])
    ap.add_argument('--acdc-split', default='val', choices=['train', 'val'])
    ap.add_argument('--width', type=int, default=32, help='NAFNet width of the checkpoints')
    ap.add_argument('--arch', default='nafnet', choices=['nafnet', 'restormer'])
    ap.add_argument('--models', nargs='*', default=[], help='name=checkpoint')
    ap.add_argument('--device', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = a.device or default_device()
    root = a.root or config.get(ROOT_VAR[a.layout])

    pairs, n_imgs = gather_pairs(root, a.layout, a.acdc_cond, a.acdc_split, a.limit)
    print(f'{len(pairs)} annotated images of {n_imgs}', flush=True)

    seg = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).eval().to(dev)
    restorers = {n: load_restorer(c, dev, a.width, a.arch) for n, c in (s.split('=', 1) for s in a.models)}
    base = 'input (no restoration)'
    variants = [base] + list(restorers)
    cms = {v: np.zeros((N_CLASSES, N_CLASSES), np.int64) for v in variants}

    t0 = time.time()
    for i, (stem, ip, gp) in enumerate(pairs):
        rgb = cv2.imread(ip)[:, :, ::-1].astype(np.float32) / 255.0
        gt = to_trainid(cv2.imread(gp, cv2.IMREAD_UNCHANGED))
        if gt.shape != rgb.shape[:2]:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        confusion(segment(seg, rgb, dev), gt, cms[base])
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(dev)
        for name, net in restorers.items():
            with torch.no_grad():
                y = np.clip(net(x)[0].permute(1, 2, 0).cpu().numpy(), 0, 1)
            confusion(segment(seg, y, dev), gt, cms[name])
        if dev == 'mps':
            torch.mps.empty_cache()
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(pairs)}  ' + '  '.join(f'{v}={miou(cms[v])[0]*100:.1f}' for v in variants), flush=True)

    res = {'n_images': len(pairs), 'segmenter': SEG_MODEL, 'layout': a.layout, 'variants': {}}
    if a.layout == 'acdc':
        res['acdc_condition'] = a.acdc_cond
    print(f'\n=== {a.layout}, restore-then-segment mIoU ({len(pairs)} images, {time.time()-t0:.0f}s) ===')
    for v in variants:
        m, iou = miou(cms[v])
        res['variants'][v] = {'miou': m * 100, 'per_class': {CLASS_NAMES[i]: (None if np.isnan(x) else x * 100)
                                                            for i, x in enumerate(iou)}}
        print(f'  {v:30s} mIoU {m*100:6.2f}')
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

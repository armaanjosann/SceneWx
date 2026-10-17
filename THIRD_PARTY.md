# Third-party components

`LICENSE` (MIT) covers the code written for this project. The components below keep their own terms.
No third-party data or model weights are included in this repository.

## Code adapted or re-implemented

| Component | Used in | Licence |
|---|---|---|
| Li et al., Heavy Rain Image Restoration, CVPR 2019 ([code](https://github.com/liruoteng/HeavyRainRemoval)) | `scenewx/rain.py`: rain-streak noise statistics and streak lengths, re-implemented from the paper's generator | No licence file upstream |
| Restormer, Zamir et al., CVPR 2022 ([code](https://github.com/swz30/Restormer)) | `basicsr_plugins/Restormer_arch.py`: re-implementation of the architecture | MIT (notice below) |
| Sakaridis et al., IJCV 2018 and ECCV 2018 | `scenewx/depth.py` (label-guided depth filter), `scenewx/fog.py` (fog levels): methods only | n/a |

## Libraries

| Component | Licence |
|---|---|
| NAFNet ([code](https://github.com/megvii-research/NAFNet)), including its BasicSR fork | MIT and Apache-2.0 |
| Hugging Face transformers | Apache-2.0 |
| PyTorch, NumPy, OpenCV, SciPy, Pillow, Matplotlib | See each project |

## Model weights (downloaded at run time)

| Model | Used for | Licence |
|---|---|---|
| Depth Anything V2 Metric Outdoor Large | Metric depth | Non-commercial (CC BY-NC 4.0 per the project README) |
| SegFormer-B0, Cityscapes | Restore-then-segment evaluation | NVIDIA licence, non-commercial |

## Datasets (download from the providers)

| Dataset | Used for | Terms |
|---|---|---|
| Cityscapes | Clear frames and labels | Non-commercial; no redistribution, including modified images |
| CSD snow masks | Snow particles | See the CSD repository |
| ACDC, Foggy Zurich, Foggy Driving | Restore-then-segment evaluation | See each provider |
| GT-RAIN | Paired PSNR (rain) | Research use |
| WeatherBench | Paired PSNR (rain, snow) | See the provider |

`scenewx/calibrate.py` contains severity statistics measured on GT-RAIN and WeatherBench (summary
numbers, not data).

## Restormer licence

```
MIT License

Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

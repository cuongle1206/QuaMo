# QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture
Official implementation of QuaMo.

[![Paper](https://img.shields.io/badge/arXiv-2601.19580-red)](https://arxiv.org/abs/2601.19580) [![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2026-blue?logo=openreview)](https://openreview.net/forum?id=em0jPLYjaS) [![Live Demo](https://img.shields.io/badge/demo-online-green?style=flat-square)](https://cuongle1206.github.io/QuaMo/visualizations/index.html)

<div align="center">
<img src="figures/Teaser.png" width="1200" alt="logo"/>
</div>


## Installation
Clone the repository and create a fresh Conda environment with recommended dependencies as follow:

```bash
git clone https://github.com/cuongle1206/QuaMo.git
cd QuaMo
conda create -n quamo python=3.8
conda activate quamo
pip install -r requirements.txt
```

Download the project folder either from [Zenodo](https://zenodo.org/records/18402382?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJlZmY3YmI2LTBjNmUtNGM2Mi04ZWU5LTEzMzBlNTZlOWY1OSIsImRhdGEiOnt9LCJyYW5kb20iOiJlNTM5NGFlZWY3NGUxNDYyM2ZkM2I5YjlmN2RjOTBlNiJ9.IZkYx7TVkwqfVBhbEKn9iHgui-PfV_RYST_zsFaE2kvgPgK5jq84fdCT94kD-b431RmWCinUEvDs0OGv6yFeZA) or [OneDrive](https://liuonline-my.sharepoint.com/:u:/g/personal/cuole74_liu_se/IQBMsb7AFpPiR6DpFjW5s5C9AZFa58VhNhQf_LMFe2QrxJc?e=lRPFrI), and extract it to the repo. All pre-processed data and trained models are included.

## Experiments
Each network is trained with seed 0-5, you can choose any. For training, you could set the flag ```--train``` to each experiment.

To test/evaluate the models on the [Human3.6M](http://vision.imar.ro/human3.6m/description.php) database, you could change the input to [HMR2](https://shubham-goel.github.io/4dhumans/) or [TRACE](https://www.yusun.work/TRACE/TRACE.html):

```bash
python main_h36m.py --input hmr2 --seed 0
python main_h36m.py --input trace --seed 0
```

To test/evaluate the models on the [Fit3D](https://fit3d.imar.ro/) database, only [TRACE](https://www.yusun.work/TRACE/TRACE.html) is available:

```bash
python main_fit3d.py --seed 0
```

To test/evaluate the models on the [SportsPose](https://christianingwersen.github.io/SportsPose/) database, only [TRACE](https://www.yusun.work/TRACE/TRACE.html) is available:

```bash
python main_sport.py --seed 0
```

## Visualization
Some qualitative examples from QuaMo can be found here:
<a href="https://cuongle1206.github.io/QuaMo/visualizations/index.html" target="_blank">View Live Demo</a>

## Citation
If you find our project helpful, please cite the paper as
```bibtex
@inproceedings{le2024_osdcap,
  title     = {QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture},
  author    = {Le, Cuong and Melnyk, Pavlo and Waldmann, Urs and Wadenbäck, Mårten and Wandt, Bastian},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=em0jPLYjaS},
}
```

## Acknownledgement
This research is partially supported by the Wallenberg Artificial Intelligence, Autonomous Systems and Software Program (WASP), funded by Knut and Alice Wallenberg Foundation. The computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at C3SE, and by the Berzelius resource, provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre.
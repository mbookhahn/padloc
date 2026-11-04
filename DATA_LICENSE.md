# Data license

The code in this repository is MIT-licensed (`LICENSE`). The fixture package
(`padloc-fixtures-v1.0.1.tar.gz`, a release asset) and the figure `docs/padloc_method.gif` are
not code: they are adapted from two datasets released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 license and are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) in turn
(© 2026 The PadLoc authors). You may share and adapt them for non-commercial purposes if you
credit the sources below, indicate your changes and release adaptations under the same license.
The terms of the source datasets apply in addition.

## Bonn: LiLocBench

Source: N. Trekel, T. Guadagnino, T. Läbe, L. Wiesmann, P. Aguiar, J. Behley and C. Stachniss,
"Benchmark for evaluating long-term localization in indoor environments under substantial static
and dynamic scene changes", IROS 2025;
<https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/> (CC BY-NC-SA 4.0).

In the package: `fixtures/bonn/`, `fixtures/bonn_cam/`, `fixtures/bonn_struct_C/` and
`fixtures/odom/`, for the four sequences with public ground truth (`static_0`, `dynamics_0`,
`lt_changes_0`, `lt_changes_dynamics_0`). Our changes:

- `poses.txt`: the benchmark's ground-truth poses as planar (x, y, yaw), one per frame.
- `depth811.txt`, `depth90.txt`: the front 2-D LiDAR scans as per-frame range columns
  (811 columns over 270°; 90 pinhole columns over 106°).
- `bonn_cam/…/depth640.txt`: D455 depth reduced to per-column band medians (640 columns).
- `bonn_struct_C/…/depth640.txt`: monocular structural depth (UnLoc's per-column depth with a
  farthest-return readout; the paper's Bonn section).
- `map.png`: in `bonn/` and `bonn_cam/` the benchmark's office map; in `bonn_struct_C/` a
  cameras-only structural map rasterized from the semantically masked D455 depth of the
  benchmark's mapping run (map C of the supplement's reference-map ablation).
- `odom/wheel_*.tum`: the benchmark's wheel odometry, aligned to the frames;
  `odom/planar_vo*.tum`: planar monocular ORB-SLAM3 odometry that we computed from the
  benchmark's camera images.

## Gibson: F3Loc's Gibson Floorplan Localization Dataset

Source: C. Chen, R. Wang, C. Vogel and M. Pollefeys, "F3Loc: Fusion and filtering for floorplan
localization", CVPR 2024; dataset <https://doi.org/10.3929/ethz-b-000670118> (CC BY-NC-SA 4.0).
The dataset was collected in the Gibson Environment (F. Xia, A. R. Zamir, Z. He, A. Sax,
J. Malik and S. Savarese, "Gibson Env: Real-world perception for embodied agents", CVPR 2018)
from the Gibson Database of Spaces. The Gibson Database of Spaces agreement
(<https://svl.stanford.edu/gibson2/assets/GDS_agreement.pdf>, which includes Matterport's
academic license) restricts the data and information derived from it to non-commercial academic
research and asks for a reference to it wherever derived information is published; this section
is that reference.

In the package: `fixtures/gibson/…/gibson_t/<scene>/depth_learned_f3loc_{mono,comp}.txt`,
`fixtures/gibson_unloc_dump/*.npz` and `split.yaml`, for the nine `gibson_t` test scenes.
Our changes: per-frame depth columns predicted from the dataset's RGB images by F3Loc's
monocular and complementary (`comp_s`) networks and by UnLoc's network (Depth Anything V2
backbone; per-column depth μ and uncertainty b̂). The package contains none of the dataset's
images, floor plans, poses or ground-truth depth; `examples/install_gibson_inputs.py` installs
the floor plans and poses from your own download of the dataset.

`docs/padloc_method.gif` draws the floor plan and trajectory of the scene Superior together with
PadLoc's intermediate results on it.

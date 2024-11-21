# golf-pose-preprocessing

## mmpose installation

Versioning is very strict for mmpose.
Even if you follow the instructions on https://mmpose.readthedocs.io/en/latest/installation.html,
you may still encounter errors when installing the packages.
For my case, I used a clean conda environment with the following versions

* CUDA 11.8
* Python 3.8.20
* PyTorch 2.0.1
* mmengine 0.10.5
* mmcv 2.0.1
* mmpose 1.3.2
* mmdet 3.1.0

## Cloning the repository

Remember to also clone and update the submodule

```bash
git clone --recursive https://github.com/jbwenjoy/golf-pose-preprocessing.git
```

Or step by step

```bash
git clone https://github.com/jbwenjoy/golf-pose-preprocessing.git
cd golf-pose-preprocessing
git submodule init
git submodule update
```

After cloning the entire repository, run

```bash
pip install -r mmpose/requirements.txt
pip install -v -e mmpose
```

to install mmpose as a module for the current project. Or you may see errors like below

```txt
Traceback (most recent call last):
  File "mmpose/demo/body3d_pose_lifter_demo.py", line 17, in <module>
    from mmpose.apis import (_track_by_iou, _track_by_oks,
ModuleNotFoundError: No module named 'mmpose'
 demo.mp4
```

## Extract golf poses from videos

Put videos in `datafolder/videos`, go to the project root folder and run

```bash
python golf_pose_extraction.py
```

Output including visualization videos, json and pickle data will be saved in `datafolder/pose_extraction`.

## TODO

1. Fix confidence score bugs

2. Golf swing phase division

3. Labeling

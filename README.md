# Golf-Swing-Classification-and-Root-Cause-Feedback

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

Remember to also clone and update the submodules (mmpose and golfdb)

```bash
git clone --recursive https://github.com/ESE546-Team18/Golf-Swing-Classification-and-Root-Cause-Feedback.git
```

Or step by step (mmpose for example)

```bash
git clone https://github.com/ESE546-Team18/Golf-Swing-Classification-and-Root-Cause-Feedback.git
cd Golf-Swing-Classification-and-Root-Cause-Feedback
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

After you can run mmpose without any error, follow GolfDB's README to download the dataset and the pre-trained models. 
`swingnet_1800.pth.tar` shoule be placed in `golfdb/models/`.
`mobilenet_v2.pth.tar` should be placed in `golfdb/`.
Do not unzip these tar files.

## Get event frames from videos using GolfDB's SwingNet

```bash
python get_event_frames_swingnet.py
```

## Extract golf poses from videos using mmpose

* For event detection and 2D pose extraction, download the dataset (160x160 videos) here (Penn SEAS account required): https://drive.google.com/drive/folders/1CaQZyJLej_T2Z3MWrpB7nAlSEEJstGGx?usp=sharing. Remember to also read the README in the link. You should put the videos in `datafolder/0_square_videos/`. Then execute

  ```bash
  python golf_2d_pose_extraction.py
  ```

  Event frame detection result will be saved in `datafolder/video_events.json`. Pose extraction result (jpg files) will be saved in `datafolder/2_pose_extraction/`.

* For 3D pose extraction, put @qqbao's videos in `datafolder/0_square_videos/`, go to the project root folder and run

  ```bash
  python golf_3d_pose_extraction.py
  ```
  
  Output including visualization videos, json and pickle data will be saved in `datafolder/2_pose_extraction/`.

## Preprocess the data for ResNet

```bash
python resnet_dataset_prep.py
```

## Training and testing

Run Jupiter notebook `resnet_training.ipynb`.

To run Grad-CAM on SwingNet, run `python golfdb/grad_cam_demo.py -p [your_video_path]`

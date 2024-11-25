import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'mmpose'))

video_folder = os.path.join(os.getcwd(), 'datafolder', 'videos')
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]

skip_processed = True

for video_file in video_files:
    if skip_processed:
        # If the filename is already in the pose_extraction folder, 
        # skip it to avoid reprocessing the same video twice
        if os.path.exists(os.path.join('datafolder', 'pose_extraction', video_file)):
            print(f'Video {video_file} already processed. Skipping...')
            continue

    video_path = os.path.join(video_folder, video_file)
    command = f'python mmpose/demo/body3d_pose_lifter_demo.py ' \
              f'mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py ' \
              f'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/' \
              f'mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth ' \
              f'mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py ' \
              f'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/' \
              f'mmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth ' \
              f'mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py ' \
              f'https://download.openmmlab.com/mmpose/body3d/videopose/' \
              f'mmpose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth ' \
              f'--input {video_path} ' \
              f'--output-root datafolder/pose_extraction ' \
              f'--save-predictions'
    
    print(f'Start processing video {video_file}...')
    os.system(command)
    print(f' {video_file}')

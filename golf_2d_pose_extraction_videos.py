import os
import sys

# Add mmpose to system path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'mmpose'))

# Path to the video folder
video_folder = os.path.join(project_root, 'datafolder', 'square_videos')
output_folder = os.path.join(project_root, 'datafolder', 'video_2d_pose_extraction')

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all video files
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]

# Skip already processed videos
skip_processed = True

for video_file in video_files:
    if skip_processed:
        # If the filename already exists in the pose_extraction folder, skip to avoid reprocessing
        if os.path.exists(os.path.join(output_folder, video_file)):
            print(f'Video {video_file} already processed. Skipping...')
            continue

    video_path = os.path.join(video_folder, video_file)
    output_path = os.path.join(output_folder, video_file)
    
    command = f'python mmpose/demo/topdown_demo_with_mmdet.py ' \
            f'mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py ' \
            f'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/' \
            f'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth ' \
            f'mmpose/configs/body_2d_keypoint/rtmpose/body8/' \
            f'rtmpose-m_8xb256-420e_body8-256x192.py ' \
            f'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/' \
            f'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth ' \
            f'--input {video_path} ' \
            f'--output-root={output_path} ' \
            f'--save-predictions ' #\
        #   f'--show ' \
        #   f'--draw-heatmap '

    os.system(command)
    print(f'Processed video {video_file}')

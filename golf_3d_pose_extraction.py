import os
import sys
import logging
from argparse import ArgumentParser
import cv2
import mmcv
import numpy as np
import pickle

def setup_env():
    """Setup the environment"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    mmpose_path = os.path.join(project_root, 'mmpose')
    if mmpose_path not in sys.path:
        sys.path.insert(0, mmpose_path)
    os.environ['PYTHONPATH'] = mmpose_path + os.pathsep + os.environ.get('PYTHONPATH', '')

setup_env()

from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,  
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmdet.apis import inference_detector, init_detector
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base-folder', default='datafolder/videos', help='Base folder containing videos')
    parser.add_argument('--output-folder', default='datafolder/pose_extraction', help='Output folder')
    
    # Detection model
    parser.add_argument('--det-config', 
                       default='mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
                       help='Config file for detection')
    parser.add_argument('--det-checkpoint',
                       default='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
                       help='Checkpoint file for detection')
    
    # 2D pose estimation model
    parser.add_argument('--pose-config',
                       default='mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
                       help='Config file for pose estimation')
    parser.add_argument('--pose-checkpoint',
                       default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
                       help='Checkpoint file for pose estimation')

    # 3D pose lifting model
    parser.add_argument('--lift-config',
                       default='mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py',
                       help='Config for pose lifter')
    parser.add_argument('--lift-checkpoint',
                       default='https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',
                       help='Checkpoint for pose lifter')

    # Other parameters
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show visualization')
    parser.add_argument('--skip-processed', action='store_true', default=True, help='Skip processed videos')
    parser.add_argument('--save-predictions', action='store_true', default=True, help='Save predictions')
    
    args = parser.parse_args()
    return args

def process_video(video_path, args, detector, pose_estimator, pose_lifter):
    """Process a single video to extract 3D poses
    
    Args:
        video_path (str): Input video path
        args (namespace): Configuration parameters
        detector (mmdet.BaseDetector): Human detector
        pose_estimator (TopdownPoseEstimator): 2D pose estimation model  
        pose_lifter (PoseLifter): 3D pose lifting model
    """
    # Initialize video capture
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Initialize output video writer
    output_path = os.path.join(args.output_folder, os.path.basename(video_path))
    video_writer = None
    
    # Initialize tracking and result lists
    next_id = 0
    pose_est_results = []
    pose_est_results_list = []
    pred_instances_list = []
    frame_idx = 0
    
    # Get pose-lifting dataset configuration
    pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
    pose_lifter.cfg.visualizer.radius = 3
    pose_lifter.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)
    visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        frame_idx += 1
        pose_est_results_last = pose_est_results

        # 1. Human detection
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        
        # Filter bounding boxes
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,  # person class
                                      pred_instance.scores > 0.3)]  # confidence threshold

        # 2. 2D pose estimation
        pose_est_results = inference_topdown(pose_estimator, frame, bboxes)
        
        # 3. Tracking and converting keypoint definitions
        pose_est_results_converted = []
        for i, data_sample in enumerate(pose_est_results):
            # Calculate track_id
            track_id = _track_by_iou(data_sample, pose_est_results_last, 0.3)[0]
            if track_id == -1:
                track_id = next_id
                next_id += 1
            pose_est_results[i].set_field(track_id, 'track_id')
            
            # Convert keypoint format
            converted = convert_keypoint_definition(
                data_sample.pred_instances.keypoints,
                pose_estimator.dataset_meta['dataset_name'],
                pose_lifter.dataset_meta['dataset_name'])
            
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(data_sample.pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.pred_instances.set_field(converted, 'keypoints')
            pose_est_result_converted.set_field(track_id, 'track_id')
            pose_est_results_converted.append(pose_est_result_converted)
            
        pose_est_results_list.append(pose_est_results_converted)

        # 4. Extract pose sequence and perform 3D lifting
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=frame_idx,
            causal=pose_lift_dataset.get('causal', False),
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        pose_lift_results = inference_pose_lifter_model(
            pose_lifter,
            pose_seq_2d,
            image_size=frame.shape[:2],
            norm_pose_2d=True)

        # 5. Post-processing and saving results
        pred_3d_instances = merge_data_samples(pose_lift_results).pred_instances
        if args.save_predictions:
            pred_instances_list.append({
                'frame_id': frame_idx,
                'instances': pred_3d_instances
            })

        # 6. Visualization
        if args.show or args.output_folder:
            frame_vis = mmcv.bgr2rgb(frame)
            visualizer.add_datasample(
                'result',
                frame_vis,
                data_sample=merge_data_samples(pose_lift_results),
                det_data_sample=merge_data_samples(pose_est_results),
                draw_gt=False,
                show=args.show,
                wait_time=1,
                draw_bbox=True,
                kpt_thr=0.3)
            
            if args.output_folder:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                        (frame_vis.shape[1], frame_vis.shape[0]))
                video_writer.write(mmcv.rgb2bgr(frame_vis))

    # Clean up resources
    video.release()
    if video_writer:
        video_writer.release()

    # Save prediction results
    if args.save_predictions:
        pred_save_path = output_path.replace('.mp4', '_pred.pkl')
        with open(pred_save_path, 'wb') as f:
            pickle.dump({
                'meta_info': pose_lifter.dataset_meta,
                'predictions': pred_instances_list
            }, f)

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize models
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_estimator = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    pose_lifter = init_model(args.lift_config, args.lift_checkpoint, device=args.device)
    
    # Get list of videos
    video_files = [f for f in os.listdir(args.base_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]
    video_files.sort()
    
    # Process each video
    for video_file in video_files:
        out_path = os.path.join(args.output_folder, video_file)
        
        if args.skip_processed and os.path.exists(out_path):
            print(f'Video {video_file} already processed. Skipping...')
            continue
            
        video_path = os.path.join(args.base_folder, video_file)
        print(f'Processing {video_file}...')
        
        process_video(video_path, args, detector, pose_estimator, pose_lifter)
        
        print(f'Saved results to {out_path}')

if __name__ == '__main__':
    main()

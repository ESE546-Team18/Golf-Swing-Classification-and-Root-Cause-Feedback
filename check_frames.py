import os
import cv2

def count_frames_in_folder(folder_path):
    video_extensions = ('.mp4', '.mov')
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                
                try:
                    video = cv2.VideoCapture(video_path)
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"File: {file} has {total_frames} frames")
                    video.release()
                except Exception as e:
                    print(f"Error when processing {file}: {str(e)}")


if __name__ == "__main__":
    folder_path = "datafolder/square_videos/"
    # folder_path = "golfdb/data/videos_160/"
    count_frames_in_folder(folder_path)

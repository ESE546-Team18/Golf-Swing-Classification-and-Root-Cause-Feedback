"""
Preprocess our own raw video files in datafolder/videos/* just like GolfDB does
    1. Reshape the videos to 160x160 when loading them
    2. Split the videos into 8 events and 1 no-event class, using the same technique as GolfDB (MobileNetV2 + SwingNet)
    3. Output one json file containing the split data for all videos (frame indexes, event labels, etc.)
    4. Extract the corresponding frames for each event, save them in a separate folder for further processing
"""

import os
import cv2
import torch
import json
import numpy as np
from golfdb.model import EventDetector
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from golfdb.dataloader import ToTensor, Normalize


class RawGolfVideoDataset(Dataset):
    """Dataset for loading raw golf videos"""

    def __init__(self, video_dir, seq_length=64, transform=None):
        self.video_dir = video_dir
        self.seq_length = seq_length
        self.transform = transform
        video_extensions = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI")
        self.video_files = [
            f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)
        ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to 160x160 and convert BGR to RGB
            frame = cv2.resize(frame, (160, 160))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = np.array(frames)

        sample = {
            "images": frames,
            "labels": np.zeros(1),  # Dummy label
            "name": self.video_files[idx],
        }

        if self.transform:
            transformed = self.transform(sample)
            transformed["name"] = sample["name"]
            sample = transformed

        return sample


def predict_events(model, dataset, seq_length, device):
    """Predict swing events in videos using SwingNet"""
    model.eval()
    results = []

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total_videos = len(data_loader)

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            images = sample["images"]
            name = sample["name"][0]

            print(f"Processing [{i+1}/{total_videos}]: {name}")

            # Process video in chunks of seq_length
            probs_list = []
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length :, :, :, :]
                else:
                    image_batch = images[
                        :, batch * seq_length : (batch + 1) * seq_length, :, :, :
                    ]

                logits = model(image_batch.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
                batch += 1

            all_probs = np.concatenate(probs_list, axis=0)

            # Find event frames using argmax
            events = detect_golf_events(all_probs)

            results.append(
                {
                    "video_name": name,
                    "events": events.tolist(),
                }
            )

    print("Done!")
    return results


def detect_golf_events(all_probs):
    """
    Detect golf swing event sequences
    :param all_probs: (sequence_length, 9) probability matrix
        Example: [8, 8, ..., 8, 0, 8, ..., 8, 1, 8, ..., ..., 8, 7, 8, ...]
    :return: (8,) array of event frame indexes
    """
    preds = np.zeros(8)  # 8 valid events indexing 0 - 7, idx 8 is no-event
    for i in range(8):
        preds[i] = np.argsort(all_probs[:, i])[-1]

    return preds


def extract_event_frames(video_dir, events_json, output_dir, pad_frames=0):
    """
    Extract frames around each golf swing event from videos
    Args:
        video_dir: Directory containing the source videos
        events_json: Path to JSON file with event timestamps
        output_dir: Base directory to save extracted frames
        pad_frames: Number of frames to extract before/after event frame
    """
    # Load event data
    with open(events_json, 'r') as f:
        events_data = json.load(f)
    
    # Create output directories for each event type
    event_names = ['0.Address', '1.Toe-up', '2.Mid-backswing', '3.Top', 
                   '4.Mid-downswing', '5.Impact', '6.Mid-follow-through', '7.Finish']
    for event in event_names:
        os.makedirs(os.path.join(output_dir, event), exist_ok=True)
    
    # Process each video
    for video_data in events_data:
        video_name = video_data['video_name']
        video_path = os.path.join(video_dir, video_name)
        event_frames = video_data['events']
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames for each event
        for event_idx, frame_idx in enumerate(event_frames):
            frame_idx = int(frame_idx)
            event_name = event_names[event_idx]
            
            # Calculate frame range to extract
            start_frame = max(0, frame_idx - pad_frames)
            end_frame = min(total_frames, frame_idx + pad_frames + 1)
            
            # Extract frames
            for f in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if ret:
                    # Resize frame to 160x160
                    frame = cv2.resize(frame, (160, 160))
                    
                    # Save frame
                    out_filename = f"{video_name.split('.')[0]}_{f:04d}.jpg"
                    out_path = os.path.join(output_dir, event_name, out_filename)
                    cv2.imwrite(out_path, frame)
        
        cap.release()
        print(f"Processed {video_name}")


def predict_events_and_save():
    """Predict events in raw videos and save the indices of event frames as json"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )

    save_dict = torch.load("golfdb/models/swingnet_1800.pth.tar")
    model.load_state_dict(save_dict["model_state_dict"])
    model = model.to(device)

    transform = transforms.Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    dataset = RawGolfVideoDataset(
        video_dir="datafolder/0_square_videos", seq_length=64, transform=transform
    )

    results = predict_events(model, dataset, seq_length=64, device=device)

    with open("datafolder/video_events.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # predict_events_and_save()
    extract_event_frames(
        video_dir="datafolder/0_square_videos",
        events_json="datafolder/video_events.json", 
        output_dir="datafolder/1_original_event_frames",
        pad_frames=0
    )

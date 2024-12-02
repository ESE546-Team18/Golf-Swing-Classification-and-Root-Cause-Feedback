# Preprocess our own raw video files in datafolder/videos/* just like GolfDB does
# 1. Reshape the videos to 160x160 when loading them
# 2. Split the videos into 8 events and 1 no-event class, using the same technique as GolfDB (MobileNetV2 + SwingNet)
# 3. Output one json file containing the split data for all videos (frame indexes, event labels, etc.)

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


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )

    # Load pretrained weights
    save_dict = torch.load("golfdb/models/swingnet_1800.pth.tar")
    model.load_state_dict(save_dict["model_state_dict"])
    model = model.to(device)

    # Create dataset
    transform = transforms.Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    dataset = RawGolfVideoDataset(
        video_dir="datafolder/square_videos", seq_length=64, transform=transform
    )

    # Predict events
    results = predict_events(model, dataset, seq_length=64, device=device)

    # Save results
    with open("datafolder/video_events.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

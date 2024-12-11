import re
import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split


def extract_image_info(filename):
    """
    Extract the name before the underscore, the 4-digit frame number after the underscore, and the label from the image filename.

    :param filename: str, image filename (e.g., 'IMG_6182-00_0029.jpg')
    :return: tuple, (name, frame_number, label)
    """
    # match = re.match(r"(.+?)_(\d{4})\.jpg$", filename)
    match = re.match(r"(.+?)_(\d{4})\.jpg_vis_results\.jpg$", filename)

    if match:
        name = match.group(1)  # Part before the underscore
        frame_number = match.group(2)  # 4-digit frame number
        # If the name is purely numeric, set the label to 1 (golfdb good swings), otherwise -1 (our data bad swings)
        label = 1 if name.isdigit() else -1
        return name, frame_number, label
    else:
        raise ValueError(f"Incorrect filename format: {filename}")


def process_golf_swing_images(base_dir):
    """
    Traverse directories containing golf swing stages, extract the filename, frame number, and label for each image,
    hstack images with the same name (one per phase), and assign a label.

    :param base_dir: str, the base directory containing subdirectories for different golf swing stages
    :return: list, containing dictionaries with file_name, hstacked_image, and label
    """
    results = []

    # Iterate through each phase directory
    phase_folders = sorted(os.listdir(base_dir))  # Ensure consistent order for phases
    image_dict = {}

    for phase_folder in phase_folders:
        phase_path = os.path.join(base_dir, phase_folder)
        if not os.path.isdir(phase_path):
            continue

        print(f"Processing phase: {phase_folder}")

        for file in sorted(os.listdir(phase_path)):
            if file.endswith(".jpg"):
                try:
                    name, frame_number, label = extract_image_info(file)
                except ValueError as e:
                    print(e)
                    continue

                file_path = os.path.join(phase_path, file)
                img = cv2.imread(file_path)  # Read the image
                if img is None:
                    print(f"Unable to read image: {file_path}")
                    continue

                # Group images by name
                if name not in image_dict:
                    image_dict[name] = {"images": [None] * len(phase_folders), "label": label}
                # Store the image in the correct phase slot
                phase_idx = phase_folders.index(phase_folder)
                image_dict[name]["images"][phase_idx] = img

    # For each unique name, hstack all associated images
    for name, data in image_dict.items():
        # Ensure all phases have an image; if missing, replace with a black placeholder
        images = [
            img if img is not None else np.zeros_like(data["images"][0]) for img in data["images"]
        ]
        hstack_image = np.hstack(images)  # Horizontally stack images
        label = data["label"]
        results.append({"file_name": name, "hstack_image": hstack_image, "label": label})

    return results


def augment_images(images, good_sample_flip_prob=0.3):
    """
    Perform data augmentation:
    - Mirror-flip bad swings (-1 label) entirely.
    - Mirror-flip good swings (1 label) with a given probability.

    :param images: list of dictionaries containing 'file_name', 'hstack_image', and 'label'
    :param good_sample_flip_prob: float, probability of flipping good swings
    :return: list of augmented images
    """
    augmented_images = []

    for item in images:
        # Always add the original image
        augmented_images.append(item)

        if item["label"] == -1:
            # Add a horizontally flipped version for bad swings
            flipped_image = cv2.flip(item["hstack_image"], 1)  # Horizontal flip
            augmented_images.append({"file_name": item["file_name"] + "_flipped",
                                     "hstack_image": flipped_image, "label": -1})
        elif item["label"] == 1:
            # Flip good swings with a probability
            if random.random() < good_sample_flip_prob:
                flipped_image = cv2.flip(item["hstack_image"], 1)  # Horizontal flip
                augmented_images.append({"file_name": item["file_name"] + "_flipped",
                                         "hstack_image": flipped_image, "label": 1})

    return augmented_images


def resize_images(images, target_height, target_width):
    """
    Resize images to a given target height and width.

    :param images: list of dictionaries with 'hstack_image'
    :param target_height: int, target height of resized image
    :param target_width: int, target width of resized image
    :return: list of resized images
    """
    resized_images = []

    for item in images:
        resized_image = cv2.resize(item["hstack_image"], (target_width, target_height))
        resized_images.append({"file_name": item["file_name"],
                               "hstack_image": resized_image,
                               "label": item["label"]})
    return resized_images


def split_dataset(images, test_size=0.2):
    """
    Split the dataset into training and testing sets.

    :param images: list of dictionaries with 'hstack_image' and 'label'
    :param test_size: float, proportion of the dataset to include in the test split
    :return: training and testing datasets
    """
    train, test = train_test_split(images, test_size=test_size, random_state=42, stratify=[img["label"] for img in images])
    return train, test


def save_dataset(dataset, output_dir):
    for item in dataset:
        label_dir = os.path.join(output_dir, str(item['label']))
        os.makedirs(label_dir, exist_ok=True)
        output_path = os.path.join(label_dir, f"{item['file_name']}.jpg")
        cv2.imwrite(output_path, item["hstack_image"])


if __name__ == "__main__":
    base_directory = "datafolder/pose_extraction/without_bg"

    output = process_golf_swing_images(base_directory)

    # # If needed, save the hstacked images
    # output_dir = "test_output"
    # os.makedirs(output_dir, exist_ok=True)

    # for idx, item in enumerate(output):
    #     file_name = item["file_name"]
    #     hstack_image = item["hstack_image"]
    #     label = item["label"]

    #     # Save the hstacked image
    #     output_path = os.path.join(output_dir, f"{file_name}_label_{label}.jpg")
    #     cv2.imwrite(output_path, hstack_image)
    #     print(f"Saved: {output_path}")

    # Augment images: flip all bad swings and good swings with a probability
    augmented_output = augment_images(output, good_sample_flip_prob=0.3)

    # Resize images to 80x(80*8)
    target_height = 160
    target_width = 160 * 8  # 8 phases
    resized_output = resize_images(augmented_output, target_height, target_width)

    # Split dataset into training and testing sets
    train_data, test_data = split_dataset(resized_output, test_size=0.2)

    # If needed, save datasets for training and testing
    train_dir = "datafolder/frames_without_bg_train_dataset"
    test_dir = "datafolder/frames_without_bg_test_dataset"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    save_dataset(train_data, train_dir)
    save_dataset(test_data, test_dir)

    print(f"Training dataset saved to {train_dir}")
    print(f"Testing dataset saved to {test_dir}")

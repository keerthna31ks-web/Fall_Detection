
import cv2
import numpy as np
import torch
from torchvision import transforms


# Model preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def extract_frames(video_path, sequence_length=32):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Could not read video.")

    # Select 32 evenly spaced frames
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length,
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            continue

        # OpenCV BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize and convert to tensor
        frame_tensor = transform(frame)

        frames.append(frame_tensor)

    cap.release()

    if len(frames) != sequence_length:
        raise ValueError(
            f"Expected {sequence_length} frames, "
            f"but extracted {len(frames)}."
        )

    # Shape: [32, 3, 224, 224]
    video_tensor = torch.stack(frames)

    # Add batch dimension
    # Shape: [1, 32, 3, 224, 224]
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor

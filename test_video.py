
import torch
from model import load_model
from video_processor import extract_frames


# Model path
MODEL_PATH = "models/best_32frame.pth"

# Enter your video path here
VIDEO_PATH = r"D:\Fall_Detection\videos\normal.mp4"


# Load trained model
print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# Extract 32 frames from video
print("\nProcessing video...")

video_tensor = extract_frames(VIDEO_PATH)

print("Video processed successfully!")
print("Input shape:", video_tensor.shape)


# Predict
print("\nPredicting...")

with torch.no_grad():

    output = model(video_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


# Class names
classes = ["Normal", "Fall"]

prediction = classes[predicted_class]

confidence = probabilities[0][predicted_class].item() * 100


# Display result
print("\n========== RESULT ==========")

print("Prediction:", prediction)
print("Normal probability:", probabilities[0][0].item() * 100)
print("Fall probability:", probabilities[0][1].item() * 100)

print(f"Confidence: {confidence:.2f}%")

print("=========================")

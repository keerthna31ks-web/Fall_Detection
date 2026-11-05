
import torch
from model import load_model

MODEL_PATH = "models/best_32frame.pth"

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# Create a dummy sequence of 32 frames
sample = torch.randn(1, 32, 3, 224, 224)

print("Testing 32-frame sequence...")

with torch.no_grad():
    output = model(sample)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

classes = ["Normal", "Fall"]

print("\nOutput:", output)
print("\nProbabilities:", probabilities)
print("\nPredicted class:", classes[predicted_class])

print("\n32-frame test completed successfully!")

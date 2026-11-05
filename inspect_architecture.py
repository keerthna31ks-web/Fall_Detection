
import torch

MODEL_PATH = "models/best_32frame.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

state_dict = checkpoint["model_state_dict"]

print("========== CNN LAYER STRUCTURE ==========")

for key, value in state_dict.items():

    if key.startswith("cnn."):

        print(key, value.shape)

print("\n========== LSTM STRUCTURE ==========")

for key, value in state_dict.items():

    if key.startswith("lstm."):

        print(key, value.shape)

print("\n========== FC STRUCTURE ==========")

for key, value in state_dict.items():

    if key.startswith("fc."):

        print(key, value.shape)

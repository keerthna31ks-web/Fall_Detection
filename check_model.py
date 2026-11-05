
import torch

MODEL_PATH = "models/best_32frame.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

state_dict = checkpoint["model_state_dict"]

print("========== MODEL DIMENSIONS ==========")

print(
    "LSTM input size:",
    state_dict["lstm.weight_ih_l0"].shape[1]
)

print(
    "LSTM hidden size:",
    state_dict["lstm.weight_hh_l0"].shape[1]
)

print(
    "LSTM layers:",
    state_dict["lstm.weight_hh_l0"].shape[0] // 4
)

print(
    "FC input features:",
    state_dict["fc.weight"].shape[1]
)

print(
    "FC output classes:",
    state_dict["fc.weight"].shape[0]
)

print("\n========== CNN FINAL LAYER ==========")

print(
    "CNN final convolution shape:",
    state_dict["cnn.7.1.conv2.weight"].shape
)

print("\n========== TRAINING INFORMATION ==========")

print("Epoch:", checkpoint["epoch"])
print("Validation accuracy:", checkpoint["val_acc"])
print("Validation loss:", checkpoint["val_loss"])

print("\nModel inspection completed!")

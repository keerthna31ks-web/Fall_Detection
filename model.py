
import torch
import torch.nn as nn
from torchvision import models


class FallDetectionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ResNet18 CNN feature extractor
        resnet = models.resnet18(weights=None)

        # Remove original classification layer
        resnet.fc = nn.Identity()

        # Sequential CNN architecture
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # Final classification
        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        batch_size, seq_len, channels, height, width = x.shape

        # Combine batch and sequence dimensions
        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width
        )

        # Extract CNN features
        features = self.cnn(x)

        # Restore sequence dimension
        features = features.reshape(
            batch_size,
            seq_len,
            512
        )

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Last time step
        last_output = lstm_out[:, -1, :]

        # Classification
        output = self.fc(last_output)

        return output


def load_model(model_path):

    model = FallDetectionModel()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict)

    model.eval()

    return model

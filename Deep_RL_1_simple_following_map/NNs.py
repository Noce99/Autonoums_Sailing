import torch
import torch.nn as nn


class SimpleCnn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = self.calculate_cnn_output_size(input_size)
        print(f"cnn_output_size = {cnn_output_size}")
        self.fully_connected = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def calculate_cnn_output_size(self, input_size):
        output = self.cnn(torch.zeros((1, 1, input_size, input_size)))
        return output.shape[1]

    def forward(self, x):
        input_dimension = x.dim()
        if input_dimension == 3:
            x = x[:, None, :]
        elif input_dimension == 2:
            x = x[None, None, :]
        cnn_output = self.cnn(x)
        output = self.fully_connected(cnn_output)
        return output


if __name__ == "__main__":
    s_cnn = SimpleCnn(10)
    print(s_cnn(torch.zeros((10, 10))))


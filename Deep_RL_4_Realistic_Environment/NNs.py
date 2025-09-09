import math

import torch
import torch.nn as nn
from Sailing_Boats_Autopilot.utils import torch_from_direction_to_ones


class SimpleCnn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = self.calculate_cnn_output_size(input_size)
        output_cnn = 1
        self.fully_connected_cnn = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_cnn),
        )
        self.final_fully_connected = nn.Sequential(
            nn.Linear(output_cnn+16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def calculate_cnn_output_size(self, input_size):
        output = self.cnn(torch.zeros((1, 2, input_size, input_size)))
        return output.shape[1]

    def forward(self, x, numeric_inputs):
        input_dimension = x.dim()
        if not isinstance(numeric_inputs, list) and not isinstance(numeric_inputs, torch.Tensor):
            print(f"by Forward: [{numeric_inputs}] is not a list or a Tensor")
        numeric_input_dimension = numeric_inputs.dim()
        if numeric_input_dimension == 1:
            numeric_inputs = numeric_inputs[None, :]
        numeric_inputs = torch_from_direction_to_ones(numeric_inputs)
        if input_dimension == 3:
            x = x[None, :]
        cnn_output = self.cnn(x)
        # print(f"cnn_output = {cnn_output.shape}")
        output_cnn = self.fully_connected_cnn(cnn_output)
        # print(f"output_cnn = {output_cnn.shape}")
        # print(f"numeric_input = {numeric_input.shape}")
        # print(f"final_input = {torch.concatenate((output_cnn, numeric_input), dim=-1).shape}")
        output = self.final_fully_connected(torch.concatenate((output_cnn, numeric_inputs), dim=-1))
        # print(f"output = {output.shape}")
        return output


if __name__ == "__main__":
    s_cnn = SimpleCnn(10).cuda()
    numbers = torch.tensor([[0*math.pi/4 + 0.001, 5*math.pi/4 + 0.001], [4*math.pi/4 + 0.001, 5*math.pi/4 + 0.001],
                            [4*math.pi/4 + 0.001, 5*math.pi/4 + 0.001]]).cuda()
    print(s_cnn(torch.zeros((3, 2, 10, 10)).cuda(), numbers))


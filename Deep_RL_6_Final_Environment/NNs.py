import math

import torch
import torch.nn as nn
from Sailing_Boats_Autopilot.utils import torch_from_direction_to_ones
from matplotlib import pyplot as plt


class SimpleCnn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = self.calculate_cnn_output_size(input_size)
        cnn_fully_connected_output_size = 16
        self.fully_connected_cnn = nn.Sequential(
            nn.Linear(cnn_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cnn_fully_connected_output_size),
            nn.ReLU(),
        )
        self.final_fully_connected = nn.Sequential(
            nn.Linear(24+1+cnn_fully_connected_output_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=9, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.single_linear = nn.Sequential(
            nn.Linear(self.calculate_cnn_output_size(input_size)[1] + 7, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.ReLU()
        )
        """
        The 6 are:
            1: boat direction           [0: 1]
            2: target direction         [0; 1]
            3: wind direction           [0; 1]
            4: wind speed               [0; 1] almost supposing 40 as maximum speed
            5: boat speed               [0; 1] almost supposing 20 as maximum speed
            6: target angular distance  [0; 1] supposing starting angular distance as maximum
            7: passed steps             [0; 1] supposing MAX_EPISODE_STEPS as maximum
        """

    def calculate_cnn_output_size(self, input_size):
        output = self.cnn(torch.zeros((1, 2, input_size, input_size)))
        return output.shape

    def forward(self, state, state_image):
        """
        :param state: Torch Tensor of shape [Batch Size, 17]
        :param state_image: Torch Tensor of shape [Batch Size, 11, 11, 2]
        :return: Torch Tensor of shape [Batch Size, 3] witch give for each input the probability of doing one of the
                 three actions
        """

        dimension = state.dim()
        if dimension == 1:
            state = state[None, :]

        dimension = state_image.dim()
        if dimension == 3:
            state_image = state_image[None, :]

        """
        cnn_out = self.fully_connected_cnn(self.cnn(state_image))

        output = self.final_fully_connected(torch.concatenate([state, cnn_out], dim=1))
        """
        output_cnn = self.cnn(state_image)
        output = self.single_linear(torch.concatenate([state, output_cnn], dim=1))

        return output


if __name__ == "__main__":
    s_cnn = SimpleCnn(10)
    print(s_cnn.calculate_cnn_output_size(16))

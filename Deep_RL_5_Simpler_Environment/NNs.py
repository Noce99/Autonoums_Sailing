import math

import torch
import torch.nn as nn
from Sailing_Boats_Autopilot.utils import torch_from_direction_to_ones


class SimpleCnn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.final_fully_connected = nn.Sequential(
            nn.Linear(17, 32),
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

    def forward(self, numeric_inputs):
        """
        :param numeric_inputs: is a torch.Tensor or a list of shape [n, 2], with n >= 1 (also shape [2] is accepted)
        :return: the output with shape [n, 3] witch give for each input the probability of doing one of the three
                 actions
        """

        numeric_input_dimension = numeric_inputs.dim()
        if numeric_input_dimension == 1:
            numeric_inputs = numeric_inputs[None, :]

        output = self.final_fully_connected(numeric_inputs)

        # print(f"output = {output.shape}")
        return output


if __name__ == "__main__":
    s_cnn = SimpleCnn(10).cuda()
    print("Need to be updated!")
    numbers = torch.tensor([[0*math.pi/4 + 0.001, 5*math.pi/4 + 0.001], [4*math.pi/4 + 0.001, 5*math.pi/4 + 0.001],
                            [4*math.pi/4 + 0.001, 5*math.pi/4 + 0.001], [4*math.pi/4 + 0.001, 5*math.pi/4 + 0.001],
                            ]).cuda()

    # Also the following work:
    # numbers = torch.tensor([0*math.pi/4 + 0.001, 5*math.pi/4 + 0.001]).cuda()

    print(s_cnn(numbers))


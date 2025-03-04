from typing import List
import torch
import torch.nn as nn

class SimpleFeedForward(nn.Module):
    def __init__(self, n_input: int, hidden_widths: List[int], n_output: int, dropout_rate: float=0.2):
        super(SimpleFeedForward, self).__init__()

        layers = []
        prev_width = n_input

        for width in hidden_widths:
            layers.append(nn.Linear(prev_width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_width = width

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_width, n_output)

    def forward(self, x: torch.Tensor):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x








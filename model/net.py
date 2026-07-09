"""Model definition for the MNIST MLP mapped onto the systolic array.

Architecture: 784 -> 32 -> 16 -> 10, ReLU activations.
Chosen so each layer's weight matrix tiles cleanly into 4x4 (and 8x8) blocks:
    layer0: 784 x 32   -> 196 x 8 tiles of 4x4
    layer1:  32 x 16   ->   8 x 4 tiles of 4x4
    layer2:  16 x 10   ->   4 x 3 tiles of 4x4 (10 padded to 12 outputs)
"""

import torch
import torch.nn as nn

LAYER_DIMS = [(784, 32), (32, 16), (16, 10)]


class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.ModuleList(
            [nn.Linear(i, o) for i, o in LAYER_DIMS]
        )

    def forward(self, x):
        x = x.flatten(1)
        for i, layer in enumerate(self.fc):
            x = layer(x)
            if i < len(self.fc) - 1:
                x = torch.relu(x)
        return x

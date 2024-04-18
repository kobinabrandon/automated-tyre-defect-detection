"""
This module contains my attempts to build models that conform to the 
ResNet (residual network) architecture.
"""

from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, Dropout2d, Sequential, ReLU, MaxPool2d, Linear, Flatten


class ResidualBlock(Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()

        self.conv1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

        self.conv2 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

        self.downsample = downsample
        self.relu = ReLU()
        self.out_channels = out_channels

    
    def _forward(self, x:Tensor) -> Tensor:

        residual = x

        output = self.conv2(
            self.conv1(x)
        ) 

        if self.downsample:
            residual = self.downsample(x)

        output += residual

        return self.relu(output)


#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = torch.relu(self.bn2(self.conv2(y)))
        y = torch.relu(self.bn3(self.conv3(y)))
        y = y.view(y.size(0), -1)
        y = torch.relu(self.fc4(y))
        return y

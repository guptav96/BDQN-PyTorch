#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BDQNNet(nn.Module):
    def __init__(self, body):
        super(BDQNNet, self).__init__()
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        q = self.body(tensor(x))
        return dict(q=q)

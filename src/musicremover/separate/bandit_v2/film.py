# Copied from ZFTurbo/Music-Source-Separation-Training (MIT, see LICENSE in this
# folder), models/bandit_v2, commit 1f700c3. Changes: plain nn.Module instead of a
# Lightning module, librosa formulas inlined. Original model: Watcharasupat et al.,
# https://github.com/kwatcharasupat/bandit-v2

from torch import nn
import torch

class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, gamma, beta):
        return gamma * x + beta


class BTFBroadcastedFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.film = FiLM()
        
    def forward(self, x, gamma, beta):
        
        gamma = gamma[None, None, None, :]
        beta = beta[None, None, None, :]
        
        return self.film(x, gamma, beta)
    
    
    
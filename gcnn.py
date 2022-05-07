from torch import nn
import torch
from gelu import GELU
from SubLayerConnection import SublayerConnection
from Multihead_Combination import   MultiHeadedCombination


class GCNN(nn.Module):
    def __init__(self, dmodel):
        super(GCNN ,self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)

    def forward(self, state, left, inputad):
        if left is not None:
            state = torch.cat([left, state], dim=1)
        state = self.linear(state)
        degree = torch.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree = 1.0 / degree
        degree2 = degree * inputad
        state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.matmul(degree2, state)))
        state = self.linearSecond(state)
        if left is not None:
            state = state[:,left.size(1):,:]
        
        return state


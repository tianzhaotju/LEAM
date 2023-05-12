import torch.nn as nn
from CombinationLayer import CombinationLayer


class MultiHeadedCombination(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.combination = CombinationLayer()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, batch_size=-1):
        if batch_size == -1:
            batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x = self.combination(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.output_linear(x)
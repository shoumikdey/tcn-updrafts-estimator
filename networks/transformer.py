"""
Code that defines the Vanilla Transformer Encoder Pipeline.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import math
import torch
import torch.nn as nn
import copy

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, dropout=0.1):
        d_ff = d_model * 4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 35000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)
    

class Encoder_tx(nn.Module):
  def __init__(self, d_model, output_size):
    super(Encoder_tx, self).__init__()
    self.input_embedding = nn.Linear(d_model[-1] * d_model[-2], d_model[-1] * d_model[-2])
    self.attention = MultiHeadedAttention(d_model=d_model[-1] * d_model[-2], h=8)
    self.norm1 = nn.LayerNorm(d_model[-1] * d_model[-2])
    self.norm2 = nn.LayerNorm(d_model[-1] * d_model[-2])
    self.dropout1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(d_model[-1] * d_model[-2], output_size[-1])
    self.dropout2 = nn.Dropout(0.5)
    self.positional_encoding = PositionalEncoding(d_model=d_model[-1] * d_model[-2])
    self.position_wise_feed_forward = PositionwiseFeedForward(d_model[-1] * d_model[-2])
    

  def forward(self, x):
    #x = torch.flatten(x, start_dim = 1)
    #x = self.input_embedding(x)
    #x = torch.relu(x)
    #x = self.fc2(x)

    x = torch.flatten(x, start_dim=1)
    x_copy = x
    x = self.input_embedding(x)
    
    y = self.positional_encoding(x_copy)
    x = y + x
    x = torch.relu(x)

    x = self.attention(query=x, key=x, value=x, mask=None)

    x = self.dropout1(x)

    x = torch.squeeze(x, dim=1)
    x = self.norm1(x + x_copy)
    x_copy2 = x

    x = self.position_wise_feed_forward(x)
    x = x + self.norm2(x_copy2)

    x = self.fc2(x)
    x = self.dropout2(x)
    x = torch.relu(x)
    x = torch.mean(x, 0)
    return x
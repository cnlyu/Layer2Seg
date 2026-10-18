"""Layer2Seg and the comparison architectures used in the experiments."""

import math
import torch
from torch import nn


class LayerMLP(nn.Module):
    def __init__(self, input_size=16, output_size=8):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(input_size, 256), nn.Linear(256, 512)])
        self.shared_fc = nn.Linear(512, 512)
        self.fc_out1 = nn.Linear(512, output_size // 2)
        self.fc_out2 = nn.Linear(512, output_size // 2)

    def forward(self, values):
        for layer in self.fc:
            values = torch.relu(layer(values))
        values = torch.relu(self.shared_fc(values))
        return torch.cat((self.fc_out1(values), self.fc_out2(values)), dim=-1)

    def zero_parameters(self):
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.)


class SegmentRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(25, 50, 3, batch_first=True, nonlinearity="relu")
        self.fc1 = nn.Linear(50, 1)
        self.fc2 = nn.Linear(50, 1)

    def initialize(self):
        for name, parameter in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.constant_(parameter, .01)

    def forward(self, values):
        outputs, _ = self.rnn(values)
        last = outputs[:, -1, :]
        return torch.cat((self.fc1(last), self.fc2(last)), dim=-1)


class CumulativeMLP(LayerMLP):
    def __init__(self):
        super().__init__(28, 2)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        pe = torch.zeros(7, 128)
        position = torch.arange(7, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, 128, 2, dtype=torch.float32) * (-math.log(10000.) / 128))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values):
        return values + self.pe[:, :values.size(1), :]


class SegmentTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(25, 128)
        self.pos_encoder = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(128, 4, dim_feedforward=512, dropout=.1,
                                           batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 1)

    def initialize(self):
        for layer in [self.input_proj, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values):
        values = self.norm(self.encoder(self.pos_encoder(self.input_proj(values))))[:, -1, :]
        return torch.cat((self.fc1(values), self.fc2(values)), dim=-1)

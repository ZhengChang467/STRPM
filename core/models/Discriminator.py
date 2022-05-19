# -*- coding: utf-8 -*-
# @Author  : ZhengChang
# @Email   : changzheng18@mails.ucas.ac.cn
# @Software: PyCharm
import torch
import torch.nn as nn
import math


class Discriminator(nn.Module):
    def __init__(self, height, width, in_channels, hidden_channels):
        super(Discriminator, self).__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.ReLU()
        )
        self.n = int(math.log2(height))
        for i in range(self.n - 1):
            self.main.add_module(name='conv_{0}'.format(i + 1),
                                 module=nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                                                  kernel_size=3, stride=2, padding=1))
            self.main.add_module(name='gn_{0}'.format(i + 1),
                                 module=nn.GroupNorm(4, self.hidden_channels))
            self.main.add_module(name='relu_{0}'.format(i + 1),
                                 module=nn.ReLU())
        self.linear_in_channels = int(math.ceil(float(width) / (2 ** self.n)) * self.hidden_channels)
        self.linear = nn.Sequential(
            nn.Linear(self.linear_in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        output_tensor = []
        output_features = []
        for i in range(input_tensor.shape[1]):
            features = self.main(input_tensor[:, i, :])
            features = features.reshape([features.shape[0], -1])
            output_features.append(features)
            output = self.linear(features)
            output_tensor.append(output)
        output_tensor = torch.cat(output_tensor, dim=1)
        output_tensor = torch.mean(output_tensor, dim=1)
        output_features = torch.stack(output_features, dim=1)
        return output_tensor, output_features



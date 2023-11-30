# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: model
# @Time: 2023/11/30 15:50 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
import os
import numpy as np
import torch
from torchvision.datasets import mnist
# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# 导入nn和优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

class Model(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1)) # 批标准化层，对 n_hidden_1 维度的输入进行批标准化
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
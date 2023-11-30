# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: load_data
# @Time: 2023/11/30 15:00 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
import os
from torchvision.datasets import mnist
# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 下载数据
class Load_data():
    def __init__(self, root_path, data_path, train_batch_size, test_batch_size):
        self.root_path = root_path
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_loader = None
        self.test_loader = None

    def _load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        # 下载数据
        train_dataset = mnist.MNIST(os.path.join(self.root_path, self.data_path), train = True, transform = transform, download=True)
        test_dataset = mnist.MNIST(os.path.join(self.root_path, self.data_path), train = False, transform = transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

    def _plot_data(self):
        if self.train_loader is None:
            self._load_data()
        examples = enumerate(self.test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap = 'gray', interpolation='none')
            plt.title(f"Ground Truth: {example_targets[i]}")
            plt.xticks([]) # 以确保在子图中不显示刻度标签。
            plt.yticks([])
        plt.show()
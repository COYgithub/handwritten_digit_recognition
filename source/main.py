# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: main
# @Time: 2023/11/30 16:41 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
from source import load_data, world, model, Produce
import torch.optim as optim
from torch import nn

device = world.device
config = world.config
# 实例化网络
model = model.Model(28 * 28, 300, 100, 10)
model.to(device)

data_loader = load_data.Load_data(world.ROOT_PATH, world.DATA_PATH, config['train_batch_size'], config['test_batch_size'])
data_loader._load_data()
data_loader._plot_data()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = config['lr'], momentum = config['momentum'])
producer = Produce
test_loader = data_loader.test_loader
train_loader = data_loader.train_loader
train_loss, train_acc = producer.train_model(model, optimizer, criterion, train_loader, config, device)
for (loss, acc) in zip(train_loss, train_acc):
    print(f'training loss: {loss}, acc: {acc}')
test_loss, test_acc = producer.test_model(model, criterion, test_loader, device)
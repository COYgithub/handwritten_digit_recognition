# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: world
# @Time: 2023/11/30 15:27 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
# 配置全局参数
import os
import torch
import torch.nn as nn
import torch.optim as optim
import parse
args = parse.parse_args()

CODE_PATH = os.getcwd()
ROOT_PATH = os.path.dirname(CODE_PATH)
DATA_PATH = os.path.join(ROOT_PATH, 'dataSet')

FILE_PATH = os.path.join(CODE_PATH, 'checkpoints')

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True) # 设置为 True 表示如果目录已经存在，不会引发 FileExistsError 异常。如果设置为 False，当目录已存在时，会引发异常。

config = {}
config['train_batch_size'] = args.train_batch_size
config['test_batch_size'] = args.test_batch_size
config['lr'] = args.lr
config['epoches'] = args.epoches
config['momentum'] = 0.5

# 设置GPU
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu") # bool


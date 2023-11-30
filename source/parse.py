# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: parse
# @Time: 2023/11/30 15:19 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "handwritten digit recognition")

    parser.add_argument("--train_batch_size", type=int, default = 64)
    parser.add_argument("--test_batch_size", type=int, default = 128)
    parser.add_argument("--lr", type=int, default = 0.01)
    parser.add_argument("--epoches", type=int, default = 20, help = "epoches of training times")

    return parser.parse_args()
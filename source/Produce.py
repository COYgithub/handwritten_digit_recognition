# !/usr/bin/python 3.7
# -*- encoding: utf-8 -*-
# @ModuleName: handwritten_digit_recognition
# @fileName: main
# @Time: 2023/11/30 14:59 
# @Author: code_of_yang
# @blog_addr: https://blog.csdn.net/qq_45892431
import os.path

import torch
import world
def train_model(model, optimizer, criterion, train_loader, config, device):
    if os.path.exists(os.path.join(world.FILE_PATH, 'trained_model_Loss_and_Acc.pth')):
        # 加载已保存的模型参数
        training_progress = torch.load(os.path.join(world.FILE_PATH, 'trained_model_Loss_and_Acc.pth'))
        return training_progress['train_loss'], training_progress['train_acc']
    else:
        print("----------------Start training----------------")
        # 训练模型
        losses = []
        acces = []

        for epoch in range(config['epoches']):
            train_loss = 0
            train_acc = 0
            model.train()
            # 动态修改参数学习率
            if epoch % 5 == 0:
                optimizer.param_groups[0]['lr'] *= 0.1
            for img, label in train_loader:
                img = img.to(device)
                label = label.to(device)
                img = img.view(img.size(0), -1)
                # 前向传播
                out = model(img)
                loss = criterion(out, label)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 记录误差
                train_loss += loss.item()
                # 计算分类准确率
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                train_acc += acc
            losses.append(train_loss / len(train_loader))
            acces.append(train_acc / len(train_loader))
            # 保存模型
            # 假设你的模型实例是 model，并且你希望保存到文件 'your_trained_model.pth'
        torch.save(model.state_dict(), os.path.join(world.FILE_PATH, 'trained_model_pars.pth'))
        torch.save({'train_loss': losses, 'train_acc': acces}, os.path.join(world.FILE_PATH,
                                                                            'trained_model_Loss_and_Acc.pth'))

        return losses, acces


def test_model(model, criterion, test_loader, device):
    print("----------------开始测试----------------")
    try:
        if os.path.exists(os.path.join(world.FILE_PATH, 'trained_model_pars.pth')):
            model.load_state_dict(torch.load(os.path.join(world.FILE_PATH, 'trained_model_pars.pth')))

            eval_losses = []
            eval_accuracies = []  # 更改变量名以提高清晰度

            model.eval()
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                img = img.view(img.size(0), -1)

                # 前向传播
                out = model(img)
                loss = criterion(out, label)
                eval_losses.append(loss.item())

                # 计算准确率
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                accuracy = num_correct / img.shape[0]
                eval_accuracies.append(accuracy)

            # 计算平均损失和准确率
            avg_eval_loss = sum(eval_losses) / len(test_loader)
            avg_eval_accuracy = sum(eval_accuracies) / len(test_loader)

            print(f"测试损失: {avg_eval_loss}, 准确率: {avg_eval_accuracy}")

            return eval_losses, eval_accuracies

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        print("请先训练模型！")

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tran
from tensorboardX import SummaryWriter

from net.resnet import ResNet
from net.vgg import VggNet
from readdata.mydata import MyCIFA10


# 计算准确率
def compute_accurate(output, label):
    prediction = output.cpu().data.numpy()
    result = label.cpu().data.numpy()
    test = (np.argmax(prediction, 1) == result)
    test = np.float32(test)
    return np.mean(test)

# 运行类 主要书写网络的调用过程和数据集加载的过程
class Run(object):
    def __init__(self, net, device, lr, train):
        self.device = torch.device(device)
        self.lr = lr
        self.train = train
        # 数据预处理，转为tensor
        transform = tran.Compose([tran.ToTensor(), tran.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # 加载训练数据集
        trainset = MyCIFA10(train=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
        # 加载测试数据集
        testset = MyCIFA10(train=False, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        print("数据加载完成")
        # 定义Summary_Writer
        self.writer = SummaryWriter("./graph")  # 数据存放于/graph文件夹
        if net == 'resnet':
            self.ResNet_run()
        elif net == 'vgg':
            self.VggNet_run()
        self.writer.close()

    def VggNet_run(self):  # VggNet11训练
        if self.train is False:
            net = torch.load('vgg_net.pkl')
        else:
            net = VggNet().to(self.device)
            loss_fun = nn.CrossEntropyLoss().to(self.device)
            opt = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
            a = 1
            for epoch in range(5):  # 训练5轮
                running_loss = 0.0
                running_accurate = 0.0
                print("第" + str(epoch + 1) + "轮训练开始")
                for i, data in enumerate(self.trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    outputs = net(inputs)
                    loss = loss_fun(outputs, labels)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                    running_accurate += compute_accurate(outputs, labels)
                    # 每100个batch输出loss和accurate,并将loss和accurate用tensorboard画出
                    if i % 100 == 99:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                        print('[%d, %5d] accurate: %.3f' % (epoch + 1, i + 1, running_accurate / 100))
                        self.writer.add_scalar("VggNet_loss", running_loss, a)
                        self.writer.add_scalar("VggNet_accurate", running_accurate, a)
                        running_loss = 0.0
                        running_accurate = 0.0
                        a += 1
                print("第" + str(epoch + 1) + "轮训练完成")
            torch.save(net, "vgg_net.pkl")

        print("测试集测试结果如下")
        for i, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = net(inputs)
            t = compute_accurate(outputs, labels)
            print("第", i, "个batch准确率：", t)
            self.writer.add_scalar('VggNet_test_acc', t, i + 1)

    def ResNet_run(self):  # ResNet训练
        if self.train is False:
            net = torch.load('res_net.pkl')
        else:
            net = ResNet().to(self.device)
            loss_fun = nn.CrossEntropyLoss().to(self.device)
            opt = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
            a = 1
            for epoch in range(5):  # 训练5轮
                running_loss = 0.0
                running_accurate = 0.0
                print("第" + str(epoch + 1) + "轮训练开始")
                for i, data in enumerate(self.trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    outputs = net(inputs)
                    loss = loss_fun(outputs, labels)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                    running_accurate += compute_accurate(outputs, labels)
                    # 每100个batch输出loss和accurate,并将loss和accurate用tensorboard画出
                    if i % 100 == 99:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                        print('[%d, %5d] accurate: %.3f' % (epoch + 1, i + 1, running_accurate / 100))
                        self.writer.add_scalar("ResNet_loss", running_loss, a)
                        self.writer.add_scalar("ResNet_accurate", running_accurate, a)
                        running_loss = 0.0
                        running_accurate = 0.0
                        a += 1
                print("第" + str(epoch + 1) + "轮训练完成")
            torch.save(net, "res_net.pkl")

        print("测试集测试结果如下")
        for i, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = net(inputs)
            t = compute_accurate(outputs, labels)
            print("第", i, "个batch准确率：", t)
            self.writer.add_scalar('ResNet_test_acc', t, i + 1)

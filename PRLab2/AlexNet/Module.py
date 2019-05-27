import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from AlexNet.Dataset import DataSet
from AlexNet.Net import Net


# this class sets devices and loss_acc functions
# the function train and test are used in func run
# which is calls by main()
class NetModule(object):
    # initial the tensorboard , Net and datasets
    def __init__(self):
        self.__writer = SummaryWriter()
        self.__dataset = DataSet()
        self.__module = Net()
        self.__initial_func()
        self.__device = torch.device("cuda:0")
        self.__module.to(self.__device)
        print(self.__module)

    # initial loss function
    def __initial_func(self):
        self.__optimizer = SGD(self.__module.parameters(), lr=0.01, momentum=0.9)
        self.__lossfunc = CrossEntropyLoss()

    # initial acc function
    def __accuracy__compute(self, pred, label):
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        test_np = (np.argmax(pred, 1) == label)
        test_np = np.float32(test_np)
        return np.mean(test_np)

    # train
    def __train(self):
        print('Begin Training:')
        r = 0
        for epoch in range(10):
            acc_list = []
            for i, data in enumerate(self.__dataset.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                self.__optimizer.zero_grad()
                outputs = self.__module(inputs)
                loss = self.__lossfunc(outputs, labels).cuda()
                loss.backward()
                self.__optimizer.step()
                acc_list.append(self.__accuracy__compute(outputs, labels))
                if i % 200 == 0:
                    r += 1
                    acc = sum(acc_list) / len(acc_list)
                    # write data to tensorboardX
                    self.__writer.add_scalars('train', {'loss': loss.item(), 'acc': acc}, r)
                    acc_list.clear()
                    now = int(round(time.time() * 1000))
                    now02 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
                    print('[', now02, ']', i, ':', loss)
        torch.save(self.__module, "result1.pkl")
        print("Training finished!")

    # test
    def __test(self):
        print('Begin Test:')
        acc_list = []
        for i, data in enumerate(self.__dataset.testloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.__device), labels.to(self.__device)
            outputs = self.__module(inputs)
            acc_list.append(self.__accuracy__compute(outputs, labels))
        print('Test finished')
        print(sum(acc_list) / len(acc_list))

    # run train and test
    # close the writer
    def run(self):
        self.__train()
        self.__test()
        self.__writer.close()

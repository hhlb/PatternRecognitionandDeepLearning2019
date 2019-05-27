import torch
import torchvision
import numpy as np
from torch.utils.data import dataloader
import torch.nn
from mlp.mlp import MLP


# 运行序列 对象的初始化中会将所有必要的函数和参数进行初始化
class Sequence():
  def __init__(self):
    # 建立CPU模型
    self.model = MLP().cpu()
    # 初始化数据集
    self.__initial_dataset()
    # 初始化函数
    self.__initial_func()

  def train(self):
    self.__train()

  def test(self):
    self.__test()

  # 加载数据集 如果没有会重新下载
  def __initial_dataset(self):
    self.train_set = torchvision.datasets.MNIST("./dataset/mnist/train", train=True,
                                                transform=torchvision.transforms.ToTensor(), download=True)
    self.test_set = torchvision.datasets.MNIST("./dataset/mnist/test", train=False,
                                               transform=torchvision.transforms.ToTensor(), download=True)
    self.train_dataset = torch.utils.data.DataLoader(self.train_set, batch_size=100)
    self.test_dataset = torch.utils.data.DataLoader(self.test_set, batch_size=100)

  # 初始化函数 包括loss计算和优化计算
  def __initial_func(self):
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
    self.lossfunc = torch.nn.CrossEntropyLoss()

  # 准确率计算
  @staticmethod
  def __accuracy_compute(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

  # 训练
  def __train(self):
    for x in range(4):
      for i, data in enumerate(self.train_dataset):
        self.optimizer.zero_grad()
        (inputs, labels) = data
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels)
        outputs = self.model(inputs)
        loss = self.lossfunc(outputs, labels)
        loss.backward()
        self.optimizer.step()
        if i % 100 == 0:
          print(i, ":", self.__accuracy_compute(outputs, labels))

  # 测试
  def __test(self):
    accuracy_list = []
    for i, (inputs, labels) in enumerate(self.test_dataset):
      inputs = torch.autograd.Variable(inputs)
      labels = torch.autograd.Variable(labels)
      outputs = self.model(inputs)
      accuracy_list.append(self.__accuracy_compute(outputs, labels))
    print(sum(accuracy_list) / len(accuracy_list))

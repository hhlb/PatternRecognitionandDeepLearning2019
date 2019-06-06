import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from mydata.dataset import ReadData
from net.Net import SentimentAnalysisRNN
from net.SentimentAnalysisSettings import *


class SentimentAnalysis(object):
    def __init__(self):
        self.device = torch.device(device)
        self.module = SentimentAnalysisRNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.h_state = None
        d = ReadData()
        self.neglabels, self.negnum = d.readneg()
        self.poslabels, self.posnum = d.readpos()
        self.negtrainlabels = self.neglabels[0:4000]
        self.negtestlabels = self.neglabels[4001:len(self.neglabels) - 1]
        self.negtrainnum = self.negnum[0:4000]
        self.negtestnum = self.negnum[4001:len(self.negnum) - 1]
        self.postrainlabels = self.poslabels[0:4000]
        self.postestlabels = self.poslabels[4001:len(self.poslabels) - 1]
        self.postrainnum = self.posnum[0:4000]
        self.postestnum = self.posnum[4001:len(self.posnum) - 1]
        print("This is SentimentAnalysis")

    def train(self):
        print("Begin Train:")
        for i in range(4000):
            print(i)
            x_np = np.array(self.negtrainnum[i])
            y_n = [self.negtrainlabels[i]]
            y_np = np.array(y_n)
            x_np = x_np.astype(np.float32)
            y_np = y_np.astype(np.float32)
            x = torch.from_numpy(x_np[np.newaxis, :])
            y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
            x = Variable(x)
            y = Variable(y)
            prediction = self.module(x)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(loss.item())

            x_np = np.array(self.postrainnum[i])
            y_n = [self.postrainlabels[i]]
            y_np = np.array(y_n)
            x_np = x_np.astype(np.float32)
            y_np = y_np.astype(np.float32)
            x = torch.from_numpy(x_np[np.newaxis, :])
            y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
            x = Variable(x)
            y = Variable(y)
            prediction = self.module(x)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(loss.item())
        print("End Train.")
        print("Begin Test:")

        for i in range(1000):
            print(i)
            x_np = np.array(self.negtestnum[i])
            y_n = [self.negtestlabels[i]]
            y_np = np.array(y_n)
            x_np = x_np.astype(np.float32)
            # y_np = y_np.astype(np.float32)

            x = torch.from_numpy(x_np[np.newaxis, :])  # shape (batch, time_step, input_size)
            # y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
            x = Variable(x)
            # y = Variable(y)
            prediction = self.module(x)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
            pred = prediction.data.numpy()

        for i in range(1000):
            print(i)
            x_np = np.array(self.postestnum[i])
            y_n = [self.postestlabels[i]]
            y_np = np.array(y_n)
            x_np = x_np.astype(np.float32)
            # y_np = y_np.astype(np.float32)
            x = torch.from_numpy(x_np[np.newaxis, :])  # shape (batch, time_step, input_size)
            # y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
            x = Variable(x)
            # y = Variable(y)
            prediction = self.module(x)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
            pred = prediction.data.numpy()

        print("End Test.")

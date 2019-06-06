import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from net.Net import SinusoidalPredictionRNN
from net.SinusoidalPredictionSettings import *


class SinusoidalPrediction(object):
    def __init__(self):
        self.device = torch.device(device)
        self.module = SinusoidalPredictionRNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.h_state = None
        print("This is SinusoidalPrediction")

    def train(self):
        print('Begin Train:')
        s = []
        original_sin = []
        train_set = []
        test_part = []
        p = None
        plt.figure(figsize=(19.2, 10.8))
        for step in range(200):
            start, end = step * np.pi, (step + 1) * np.pi
            steps = np.linspace(start, end, 5, dtype=np.float32)
            s.extend(steps)
            x_np = np.sin(steps - np.pi)
            y_np = np.sin(steps)
            x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
            y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
            prediction, self.h_state = self.module(x, self.h_state)
            p = prediction
            self.h_state = Variable(self.h_state.data)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            original_sin.extend(y_np.flatten())
            train_set.extend(prediction.data.numpy().flatten())
            # plt.plot(steps, y_np.flatten(), label='sin')
            # plt.plot(steps, prediction.data.numpy().flatten(), label='prediction')
        plt.plot(s, original_sin, label='sin')
        plt.plot(s, train_set, label='train')
        print("End Train.")
        s.clear()
        self.h_state = None
        print("Begin Test:")
        for step in range(200, 250):
            start, end = step * np.pi, (step + 1) * np.pi
            steps = np.linspace(start, end, 5, dtype=np.float32)
            s.extend(steps)
            prediction, self.h_state = self.module(p, self.h_state)
            p = prediction
            self.h_state = Variable(self.h_state.data)
            test_part.extend(prediction.data.numpy().flatten())
        print("End Test.")
        plt.plot(s, test_part, label='prediction')
        plt.legend()
        plt.show()

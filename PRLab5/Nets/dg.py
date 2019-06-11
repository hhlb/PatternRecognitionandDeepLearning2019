import torch.nn as nn

from Nets.settings import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n1 = nn.Linear(g_input_size, g_hidden_size)
        self.n2 = nn.Linear(g_hidden_size, g_hidden_size)
        self.n3 = nn.Linear(g_hidden_size, g_return_size)
        self.selu = nn.ReLU()

    def forward(self, x):
        x = self.selu(self.n1(x))
        x = self.selu(self.n2(x))
        x = self.n3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n1 = nn.Linear(d_input_size, d_hidden_size)
        self.n2 = nn.Linear(d_hidden_size, d_hidden_size)
        self.n3 = nn.Linear(d_hidden_size, d_return_size)
        self.elu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.elu(self.n1(x))
        x = self.elu(self.n2(x))
        x = self.sig(self.n3(x))
        return x

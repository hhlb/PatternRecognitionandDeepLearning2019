import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.autograd import Variable

from Nets.dg import Discriminator, Generator
from Nets.settings import *
from mat.data import Data


class NetsTT(object):
    def __init__(self):
        print("Get Data...")
        self.data = Data().mat_data
        self.device = torch.device(device)
        self.art = Variable(torch.from_numpy(self.data).float())
        print("Data finished.")
        print("Init GAN Generator and Discriminator...")
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        print(self.G, self.D)
        print("Init finished.")
        print("Init optimizers and loss function")
        self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=d_lr)
        self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=g_lr)
        print("Init finished")

    def get_data(self):
        x = np.linspace(-1, 2, 100)
        y = np.linspace(-1, 2, 100)
        background = []
        for i in x:
            for j in y:
                a = []
                a.append(i)
                a.append(j)
                background.append(a)
        b = np.array(background)
        b_np = torch.from_numpy(b).float()
        b_data = Variable(b_np)
        return b_data

    # TODO
    def train(self):
        plt.ion()
        f, ax = plt.subplots(figsize=(8, 8))
        b_data = self.get_data()
        print('Start Generator Trainning:')
        for step in range(steps):
            generator_ideas = Variable(torch.randn(8192, g_input_size))
            generator = self.G(generator_ideas)
            generator_points = generator.detach().numpy()

            prob_art_first = self.D(self.art)
            prob_art_second = self.D(generator)

            discriminator_loss = - torch.mean(torch.log(prob_art_first) + torch.log(1 - prob_art_second))
            generator_loss = torch.mean(torch.log(1 - prob_art_second))

            self.d_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            self.g_optimizer.zero_grad()
            generator_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            if step % 10 == 0:
                b_output = self.D(b_data).data.numpy()
                b_output = np.reshape(b_output, (100, 100))
                X, Y = np.mgrid[-1:2:complex(0, 100), -1:2:complex(0, 100)]
                # b_output = pd.DataFrame(b_output, columns=['x', 'y'])
                original_data = pd.DataFrame(Data().mat_data, columns=['x', 'y'])
                output_data = pd.DataFrame(generator_points, columns=['x', 'y'])
                ax.set_xlim([-1, 2])
                ax.set_ylim([-1, 2])
                # ax.set_aspect("equal")
                pcm = ax.pcolor(X, Y, b_output, cmap='Greys_r')
                bar = f.colorbar(pcm, ax=ax)
                # ax = sns.kdeplot(b_output['x'], b_output['y'], cmap='Reds', shade=True, cbar=True)
                ax = sns.scatterplot(x='x', y='y', data=original_data, marker='X', color='b', s=30, legend='full')
                ax = sns.scatterplot(x='x', y='y', data=output_data, marker='+',color='r', s=30, legend='full')
                plt.pause(0.05)
                bar.remove()
        plt.ioff()
        f.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.autograd as autograd
import torch.nn.utils as utils
import torch.optim as optim
from torch.autograd import Variable

from Nets.dg import Discriminator, Generator, WGANDiscriminator
from Nets.settings import *
from mat.data import Data

s = 'tab20c'


class GanNetsTT(object):
    def __init__(self):
        print("This is GAN.")
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

    def gan_train(self):
        plt.ion()
        f, ax = plt.subplots(figsize=(8, 8))
        b_data = self.get_data()
        X, Y = np.mgrid[-1:2:complex(0, 100), -1:2:complex(0, 100)]
        print('Start Generator Trainning:')
        for step in range(steps):
            print('This is step:', str(step + 1))
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
            if step % 100 == 99:
                b_output = self.D(b_data).data.numpy()
                b_output = np.reshape(b_output, (100, 100))
                # b_output = pd.DataFrame(b_output, columns=['x', 'y'])
                original_data = pd.DataFrame(Data().mat_data, columns=['x', 'y'])
                output_data = pd.DataFrame(generator_points, columns=['x', 'y'])
                ax.set_xlim([-1, 2])
                ax.set_ylim([-1, 2])
                ax.set_title('GAN:This is step:' + str(step + 1), color='red', fontweight=800)
                # ax.set_aspect("equal")
                pcm = ax.pcolor(X, Y, b_output, cmap=s)
                bar = f.colorbar(pcm, ax=ax)
                # ax = sns.kdeplot(b_output['x'], b_output['y'], cmap='Reds', shade=True, cbar=True)
                ax = sns.scatterplot(x='x', y='y', data=original_data, marker='X', color='b', s=30)
                ax = sns.scatterplot(x='x', y='y', data=output_data, marker='P', color='r', alpha=0.5, s=30)
                f.savefig('./results/RMSprop/GAN/step' + str(step+1) + '.png')
                plt.pause(0.05)
                bar.remove()
        plt.ioff()
        f.show()


class WganNetsTT(object):
    def __init__(self):
        print('This is WGAN.')
        self.data = Data().mat_data
        self.device = torch.device(device)
        self.art = Variable(torch.from_numpy(self.data).float())
        print("Data finished.")
        print("Init WGAN Generator and Discriminator...")
        self.G = Generator().to(self.device)
        self.D = WGANDiscriminator().to(self.device)
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

    def wgan_train(self):
        plt.ion()
        f, ax = plt.subplots(figsize=(8, 8))
        b_data = self.get_data()
        X, Y = np.mgrid[-1:2:complex(0, 100), -1:2:complex(0, 100)]
        print('Start Generator Trainning:')
        for step in range(steps):
            print('This is step:', str(step + 1))
            generator_ideas = Variable(torch.randn(8192, g_input_size))
            generator = self.G(generator_ideas)
            generator_points = generator.detach().numpy()

            prob_art_first = self.D(self.art)
            prob_art_second = self.D(generator)

            discriminator_loss = - torch.mean(prob_art_first + 1 - prob_art_second)
            generator_loss = torch.mean(1 - prob_art_second)

            self.d_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            self.g_optimizer.zero_grad()
            generator_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            for i in self.D.parameters():
                i.data.clamp_(-c, c)
            if step % 100 == 99:
                b_output = self.D(b_data).data.numpy()
                b_output = np.reshape(b_output, (100, 100))
                # b_output = pd.DataFrame(b_output, columns=['x', 'y'])
                original_data = pd.DataFrame(Data().mat_data, columns=['x', 'y'])
                output_data = pd.DataFrame(generator_points, columns=['x', 'y'])
                ax.set_xlim([-1, 2])
                ax.set_ylim([-1, 2])
                ax.set_title('WGAN:This is step:' + str(step + 1), color='red', fontweight=800)
                # ax.set_aspect("equal")
                pcm = ax.pcolor(X, Y, b_output, cmap=s)
                bar = f.colorbar(pcm, ax=ax)
                # ax = sns.kdeplot(b_output['x'], b_output['y'], cmap='Reds', shade=True, cbar=True)
                ax = sns.scatterplot(x='x', y='y', data=original_data, marker='X', color='b', s=30)
                ax = sns.scatterplot(x='x', y='y', data=output_data, marker='P', color='r', alpha=0.5, s=30)
                f.savefig('./results/RMSprop/WGAN/step' + str(step + 1) + '.png')
                plt.pause(0.05)
                bar.remove()
        plt.ioff()
        f.show()


class WganGPNetsTT(object):
    def __init__(self):
        print('This is WGAN-GP.')
        self.data = Data().mat_data
        self.device = torch.device(device)
        self.art = Variable(torch.from_numpy(self.data).float())
        print("Data finished.")
        print("Init WGAN-GP Generator and Discriminator...")
        self.G = Generator().to(self.device)
        self.D = WGANDiscriminator().to(self.device)
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

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print real_data.size()
        alpha = torch.rand(8192, 1)
        alpha = alpha.expand(real_data.size())
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def wgangp_train(self):
        plt.ion()
        f, ax = plt.subplots(figsize=(8, 8))
        b_data = self.get_data()
        X, Y = np.mgrid[-1:2:complex(0, 100), -1:2:complex(0, 100)]
        print('Start Generator Trainning:')
        for step in range(steps):
            print('This is step:', str(step + 1))
            generator_ideas = Variable(torch.randn(8192, g_input_size))
            generator = self.G(generator_ideas)
            generator_points = generator.detach().numpy()

            prob_art_first = self.D(self.art)
            prob_art_second = self.D(generator)

            discriminator_loss = - torch.mean(prob_art_first + 1 - prob_art_second)
            generator_loss = torch.mean(1 - prob_art_second)

            gradient_penalty = self.calc_gradient_penalty(self.D, self.art.data,
                                                          autograd.Variable(prob_art_second.data))
            gradient_penalty.backward()
            self.d_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            self.g_optimizer.zero_grad()
            generator_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            utils.clip_grad_norm_(self.D.parameters(), c)
            if step % 100 == 99:
                b_output = self.D(b_data).data.numpy()
                b_output = np.reshape(b_output, (100, 100))
                # b_output = pd.DataFrame(b_output, columns=['x', 'y'])
                original_data = pd.DataFrame(Data().mat_data, columns=['x', 'y'])
                output_data = pd.DataFrame(generator_points, columns=['x', 'y'])
                ax.set_xlim([-1, 2])
                ax.set_ylim([-1, 2])
                ax.set_title('WGAN-GP:This is step:' + str(step + 1), color='red', fontweight=800)
                # ax.set_aspect("equal")
                pcm = ax.pcolor(X, Y, b_output, cmap=s)
                bar = f.colorbar(pcm, ax=ax)
                # ax = sns.kdeplot(b_output['x'], b_output['y'], cmap='Reds', shade=True, cbar=True)
                ax = sns.scatterplot(x='x', y='y', data=original_data, marker='X', color='b', s=30)
                ax = sns.scatterplot(x='x', y='y', data=output_data, marker='P', color='r', alpha=0.5, s=30)
                f.savefig('./results/RMSprop/WGANGP/step' + str(step + 1) + '.png')
                plt.pause(0.05)
                bar.remove()
        plt.ioff()
        f.show()

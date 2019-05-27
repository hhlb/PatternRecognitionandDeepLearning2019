from torch.nn import *


# Net class
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, 96, 11, 4, 0),
            ReLU(),
            MaxPool2d(3, 2)
        )
        self.conv2 = Sequential(
            Conv2d(96, 256, 5, 1, 2),
            ReLU(),
            MaxPool2d(3, 2)
        )
        self.conv3 = Sequential(
            Conv2d(256, 384, 3, 1, 1),
            ReLU()
        )
        self.conv4 = Sequential(
            Conv2d(384, 384, 3, 1, 1),
            ReLU()
        )
        self.conv5 = Sequential(
            Conv2d(384, 256, 3, 1, 1),
            ReLU(),
            MaxPool2d(3, 2)
        )
        self.dense = Sequential(
            Linear(9216, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 1000),
            Linear(1000, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

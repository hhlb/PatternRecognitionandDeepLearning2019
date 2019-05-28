import torch.nn as nn


# VggNet11网络
# 第一层，卷积核为3，步长为1，padding=1，输入RGB三通道，输出64通道，最大池化kernel_size=2
# 第二层，卷积核为3，步长为1，padding=1，输入为64通道，输出128通道，最大池化kernel_size=2
# 第三层，两层卷积，第一层卷积核为3，步长为1，padding=1，输入为128通道，输出256通道，
#	第一层卷积核为3，步长为1，padding=1，输入为256通道，输出256通道，最大池化kernel_size=2
# 第四层，两层卷积，第一层卷积核为3，步长为1，padding=1，输入为256通道，输出512通道，
#	第一层卷积核为3，步长为1，padding=1，输入为512通道，输出512通道，最大池化kernel_size=2
# 第五层，两层卷积，第一层卷积核为3，步长为1，padding=1，输入为512通道，输出512通道，
#	第一层卷积核为3，步长为1，padding=1，输入为512通道，输出512通道，最大池化kernel_size=2
# 第六层，512 × 10 的全连接层

class VggNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )  # (64,16,16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )  # (128,8,8)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )  # (256,4,4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )  # (512,2,2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )  # (512,1,1)

        self.full_conn = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)
        x = self.full_conn(x)

        return x

import torchvision
from torch.utils.data import DataLoader


# DataSet class include all datasets the Net needs.
class DataSet(object):
    # initial the transform and the dataset
    def __init__(self):
        self.__transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.__get_trainset()

    # download the dataset and transform
    def __get_trainset(self):
        self.trainset = torchvision.datasets.CIFAR10('./dataset', download=True, train=True,
                                                     transform=self.__transform)
        self.trainloader = DataLoader(self.trainset, batch_size=50, shuffle=True)
        self.testset = torchvision.datasets.CIFAR10('./dataset', download=True, train=False,
                                                    transform=self.__transform)
        self.testloader = DataLoader(self.testset, batch_size=50, shuffle=False)

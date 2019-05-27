import pickle

import numpy as np
from PIL import Image
from torch.utils import data

import readdata.path as datapath


class MyCIFA10(data.Dataset):
    def __init__(self, transform=None, target_transform=None, train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data = []
        self.targets = []

        if self.train:
            self.data_list = datapath.train_set
        else:
            self.data_list = datapath.test_set

        for file in self.data_list:
            filepath = datapath.datapath + file
            with open(filepath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self._load_meta()

    def _load_meta(self):
        filepath = datapath.datapath + datapath.meta_data
        with open(filepath, 'rb') as infile:
            d = pickle.load(infile, encoding='latin1')
            self.classes = d['label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

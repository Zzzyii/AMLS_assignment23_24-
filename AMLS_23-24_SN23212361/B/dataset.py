import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


INFO = "B/medmnist.json"


class MedMNIST(Dataset):

    flag = ...

    def __init__(self, root, split='train', transform=None, target_transform=None):

        with open(INFO, 'r') as f:
            self.info = json.load(f)[self.flag]

        self.root = root

        if not os.path.exists(os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):

        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__
        
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)





class taskB(MedMNIST):
    flag = "PathMNIST"


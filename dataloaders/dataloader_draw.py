import random
from grpc import ssl_server_certificate_configuration
import torch.utils.data as data_utils
import numpy as np
from torchvision import datasets, transforms
import torch

from dataloaders.colorMNIST import ColoredMNIST, ColoredFashionMNIST

class ColoredMnistBagsDraw(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)
        self.num_in_train = 60000
        self.num_in_test = 10000

        self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train == True:
            loader = data_utils.DataLoader(ColoredMNIST('/home/user/datasets',
                                        env='all_train',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            ])),
                                            batch_size=self.num_in_test,
                                            shuffle=True)        
        else:
            loader = data_utils.DataLoader(ColoredMNIST('/home/user/datasets',
                                        env='test',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            ])),
                                            batch_size=self.num_in_test,
                                            shuffle=True)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        if self.train:
          s = list(range(self.num_in_train))
        else:
          s = list(range(self.num_in_test))
        np.random.shuffle(s)

        s = list(range(10000))
        np.random.shuffle(s)

        for i in range(100):
            bag_length = 10
            temp_list = []
            for j in range(10):
                current_label = all_labels == j
                ins_list = all_imgs[current_label]
                
                # randomly select one instance belonging to class j
                perm = torch.randperm(ins_list.size(0))
                ins_current = ins_list[perm[0]]
                temp_list.append(ins_current)
            bag_current = torch.stack(temp_list)
            bags_list.append(bag_current)
            labels_list.append(torch.tensor(range(10)))
        return bags_list, labels_list

    def __len__(self):
        return len(self.test_labels_list)

    def __getitem__(self, index):
        bag = self.test_bags_list[index]
        label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        bag_idx = np.repeat(index, bag.shape[0])
        return bag, label, bag_idx


class ColoredFashionMnistBagsDraw_binary(data_utils.Dataset):
    def __init__(self, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)
        self.num_in_train = 40000
        self.num_in_test = 20000

        self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train == True:
            loader = data_utils.DataLoader(ColoredFashionMNIST('/home/user/datasets',
                                        env='all_train',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            ])),
                                            batch_size=self.num_in_test,
                                            shuffle=True)        
        else:
            loader = data_utils.DataLoader(ColoredFashionMNIST('/home/user/datasets',
                                        env='test',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            ])),
                                            batch_size=self.num_in_test,
                                            shuffle=True)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        if self.train:
          s = list(range(self.num_in_train))
        else:
          s = list(range(self.num_in_test))
        np.random.shuffle(s)

        s = list(range(10000))
        np.random.shuffle(s)

        for i in range(100):
            bag_length = 10
            temp_list = []
            for j in range(10):
                current_label = all_labels == j
                ins_list = all_imgs[current_label]
                
                # randomly select one instance belonging to class j
                perm = torch.randperm(ins_list.size(0))
                ins_current = ins_list[perm[0]]
                temp_list.append(ins_current)
            bag_current = torch.stack(temp_list)
            bags_list.append(bag_current)
            labels_list.append(torch.tensor(range(10)))
        return bags_list, labels_list

    def __len__(self):
        return len(self.test_labels_list)

    def __getitem__(self, index):
        bag = self.test_bags_list[index]
        label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        bag_idx = np.repeat(index, bag.shape[0])
        return bag, label, bag_idx



class FashionMnistBagsDraw(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)
        self.num_in_train = 60000
        self.num_in_test = 10000

        self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        loader = data_utils.DataLoader(datasets.FashionMNIST('/home/user/datasets',
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          # transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
                                        batch_size=self.num_in_test,
                                        shuffle=True)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        if self.train:
          s = list(range(self.num_in_train))
        else:
          s = list(range(self.num_in_test))
        np.random.shuffle(s)

        s = list(range(10000))
        np.random.shuffle(s)

        for i in range(100):
            bag_length = 10
            temp_list = []
            for j in range(10):
                current_label = all_labels == j
                ins_list = all_imgs[current_label]
                
                # randomly select one instance belonging to class j
                perm = torch.randperm(ins_list.size(0))
                ins_current = ins_list[perm[0]]
                temp_list.append(ins_current)
            bag_current = torch.stack(temp_list)
            bags_list.append(bag_current)
            labels_list.append(torch.tensor(range(10)))
        return bags_list, labels_list

    def __len__(self):
        return len(self.test_labels_list)

    def __getitem__(self, index):
        bag = self.test_bags_list[index]
        label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        bag_idx = np.repeat(index, bag.shape[0])
        return bag, label, bag_idx


class KMnistBagsDraw(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)
        self.num_in_train = 60000
        self.num_in_test = 10000

        self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        loader = data_utils.DataLoader(datasets.KMNIST('/home/user/datasets',
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                          ])),
                                        batch_size=self.num_in_test,
                                        shuffle=True)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        if self.train:
          s = list(range(self.num_in_train))
        else:
          s = list(range(self.num_in_test))
        np.random.shuffle(s)

        s = list(range(10000))
        np.random.shuffle(s)

        for i in range(100):
            bag_length = 10
            temp_list = []
            for j in range(10):
                current_label = all_labels == j
                ins_list = all_imgs[current_label]
                
                # randomly select one instance belonging to class j
                perm = torch.randperm(ins_list.size(0))
                ins_current = ins_list[perm[0]]
                temp_list.append(ins_current)
            bag_current = torch.stack(temp_list)
            bags_list.append(bag_current)
            labels_list.append(torch.tensor(range(10)))
        return bags_list, labels_list

    def __len__(self):
        return len(self.test_labels_list)

    def __getitem__(self, index):
        bag = self.test_bags_list[index]
        label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        bag_idx = np.repeat(index, bag.shape[0])
        return bag, label, bag_idx


class MnistBagsDraw(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=0, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)
        self.num_in_train = 60000
        self.num_in_test = 10000

        self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        loader = data_utils.DataLoader(datasets.MNIST('/home/user/datasets',
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          # transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
                                        batch_size=self.num_in_test,
                                        shuffle=True)
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        if self.train:
          s = list(range(self.num_in_train))
        else:
          s = list(range(self.num_in_test))
        np.random.shuffle(s)

        s = list(range(10000))
        np.random.shuffle(s)

        for i in range(100):
            bag_length = 10
            temp_list = []
            for j in range(10):
                current_label = all_labels == j
                ins_list = all_imgs[current_label]
                
                # randomly select one instance belonging to class j
                perm = torch.randperm(ins_list.size(0))
                ins_current = ins_list[perm[0]]
                temp_list.append(ins_current)
            bag_current = torch.stack(temp_list)
            bags_list.append(bag_current)
            labels_list.append(torch.tensor(range(10)))
        return bags_list, labels_list

    def __len__(self):
        return len(self.test_labels_list)

    def __getitem__(self, index):
        bag = self.test_bags_list[index]
        label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        bag_idx = np.repeat(index, bag.shape[0])
        return bag, label, bag_idx

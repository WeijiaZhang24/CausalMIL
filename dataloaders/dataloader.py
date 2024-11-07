import random
import torch.utils.data as data_utils
import numpy as np
from torchvision import datasets, transforms
import torch
from dataloaders.colorMNIST import ColoredMNIST, ColoredFashionMNIST, ColoredMNIST
# from colorMNIST import ColoredMNIST, ColoredFashionMNIST, ColourBiasedMNIST


class KMnistBags3(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.KMNIST('/home/user/datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                            #   transforms.Normalize((0.5,), (0.5,))
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.KMNIST('/home/user/datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                            #   transforms.Normalize((0.5,), (0.5,))
                                                          ])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)            

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        positive_ins = all_labels == self.target_number
        negative_ins = all_labels != self.target_number


        positive_idx = []
        negative_idx = []
        for i, p_ins in enumerate(positive_ins):
            if p_ins:
                positive_idx.append(i)

        for i, n_ins in enumerate(negative_ins):
            if n_ins:
                negative_idx.append(i)


        bags_list = []
        labels_list = []

        # positive
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            positive_num = abs(bag_length - self.mean_bag_length)
            if positive_num == 0:
                positive_num = self.mean_bag_length//5

            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length - positive_num)
            try:
                t = random.sample(positive_idx, positive_num)
            except ValueError:
                print('No more positive instances 1')
                break
            positive_idx = list(set(positive_idx) - set(t))
            for i in t:
                indices.append(i)

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        # negative
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1
            try:
                indices = random.sample(negative_idx, bag_length)
            except ValueError:
                print('No more negative instances 1')
                break
            negative_idx = list(set(negative_idx) - set(t))
            
            indices = torch.LongTensor(indices)
            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number
            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])

        return bag, label, bag_idx


class FashionMnistBags3(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.FashionMNIST('/home/user/datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.FashionMNIST('/home/user/datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)            

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        positive_ins = all_labels == self.target_number
        negative_ins = all_labels != self.target_number


        positive_idx = []
        negative_idx = []
        for i, p_ins in enumerate(positive_ins):
            if p_ins:
                positive_idx.append(i)

        for i, n_ins in enumerate(negative_ins):
            if n_ins:
                negative_idx.append(i)


        bags_list = []
        labels_list = []

        # positive
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            positive_num = abs(bag_length - self.mean_bag_length)
            if positive_num == 0:
                positive_num = self.mean_bag_length//5

            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length - positive_num)
            try:
                t = random.sample(positive_idx, positive_num)
            except ValueError:
                print('No more positive instances 1')
                break
            try:
                positive_idx = list(set(positive_idx) - set(t))
            except ValueError:
                print('No more positive instances 2')
                break
            for i in t:
                indices.append(i)

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        # negative
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))

            if bag_length < 1:
                bag_length = 1

            try:
                indices = random.sample(negative_idx, bag_length)
            except ValueError:
                print('No more negative instances 1')
                break
            negative_idx = list(set(negative_idx) - set(t))

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)


        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])

        return bag, label, bag_idx


class MnistBags3(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('/home/user/datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              #transforms.Normalize((0.1307,), (0.3081,))
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('/home/user/datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              #transforms.Normalize((0.1307,), (0.3081,))
                                                          ])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)            

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        positive_ins = all_labels == self.target_number
        negative_ins = all_labels != self.target_number


        positive_idx = []
        negative_idx = []
        for i, p_ins in enumerate(positive_ins):
            if p_ins:
                positive_idx.append(i)

        for i, n_ins in enumerate(negative_ins):
            if n_ins:
                negative_idx.append(i)


        bags_list = []
        labels_list = []

        # positive
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            positive_num = abs(bag_length - self.mean_bag_length)
            if positive_num == 0:
                positive_num = self.mean_bag_length//5

            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length - positive_num)
            
            try:
                t = random.sample(positive_idx, positive_num)
            except ValueError:
                print('No more positive instances 1')
                break
            try:
                positive_idx = list(set(positive_idx) - set(t))
            except ValueError:
                print('No more positive instances 2')
                break

            for i in t:
                indices.append(i)


            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        # negative
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))

            if bag_length < 1:
                bag_length = 1

            try:
                indices = random.sample(negative_idx, bag_length)
            except ValueError:
                print('No more negative instances 1')
                break
            negative_idx = list(set(negative_idx) - set(t))

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)


        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])

        return bag, label, bag_idx


# Binary classification, 0-4 positive, 5-9 negative
# In training, positive are mostly red
# In test, positive are mostly green
class ColoredMnistBags3_binary(data_utils.Dataset):
    def __init__(self,  mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState()

        self.num_in_train = 40000
        self.num_in_test = 20000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(ColoredMNIST(
                                                          env='all_train',
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=True)
        else:
            loader = data_utils.DataLoader(ColoredMNIST(
                                                          env='test',
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_test,
                                           shuffle=True)            

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        positive_ins = all_labels < 5
        negative_ins = all_labels > 4

        # Flip label with 25% probability
        idx1 = np.random.choice(len(positive_ins), int(0.125*len(positive_ins)), replace=False)
        positive_ins[idx1] = ~positive_ins[idx1]
        idx2 = np.random.choice(len(negative_ins), int(0.125*len(negative_ins)), replace=False)
        negative_ins[idx2] = ~negative_ins[idx2]

        positive_idx = []
        negative_idx = []
        for i, p_ins in enumerate(positive_ins):
            if p_ins:
                positive_idx.append(i)

        for i, n_ins in enumerate(negative_ins):
            if n_ins:
                negative_idx.append(i)


        bags_list = []
        labels_list = []

        # positive
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            positive_num = abs(bag_length - self.mean_bag_length)
            if positive_num == 0:
                positive_num = self.mean_bag_length//1
            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length - positive_num)
            try:
                t = random.sample(positive_idx, positive_num)
            except ValueError:
                print('No more positive instances 1')
                break
            positive_idx = list(set(positive_idx) - set(t))

            for i in t:
                indices.append(i)


            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag < 5

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        # negative
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))

            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length)

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag < 5

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)


        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])

        return bag, label, bag_idx


class ColoredFashionMnistBags3_binary(data_utils.Dataset):
    def __init__(self, mean_bag_length=10, var_bag_length=2, num_bag=250, train=True):
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState()

        self.num_in_train = 40000
        self.num_in_test = 20000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(ColoredFashionMNIST(
                                                          env='all_train',
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=True)
        else:
            loader = data_utils.DataLoader(ColoredFashionMNIST(
                                                          env='test',
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                          ])),
                                           batch_size=self.num_in_test,
                                           shuffle=True)            

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        positive_ins = all_labels < 5
        negative_ins = all_labels > 4

        # Flip label with 25% probability
        idx1 = np.random.choice(len(positive_ins), int(0.125*len(positive_ins)), replace=False)
        positive_ins[idx1] = ~positive_ins[idx1]
        idx2 = np.random.choice(len(negative_ins), int(0.125*len(negative_ins)), replace=False)
        negative_ins[idx2] = ~negative_ins[idx2]

        positive_idx = []
        negative_idx = []
        for i, p_ins in enumerate(positive_ins):
            if p_ins:
                positive_idx.append(i)

        for i, n_ins in enumerate(negative_ins):
            if n_ins:
                negative_idx.append(i)


        bags_list = []
        labels_list = []

        # positive
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            positive_num = abs(bag_length - self.mean_bag_length)
            if positive_num == 0:
                # positive_num = self.mean_bag_length//1 # for Colored binarized tasks only 
                positive_num = self.mean_bag_length//1 # for Colored binarized tasks only 
            if bag_length < 1:
                bag_length = 1

            indices = random.sample(negative_idx, bag_length - positive_num)
            # t = random.sample(positive_idx, positive_num)
            try:
                t = random.sample(positive_idx, positive_num)
            except ValueError:
                print('No more positive instances 1')
                break
            positive_idx = list(set(positive_idx) - set(t))
   
            
            for i in t:
                indices.append(i)

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag < 5

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        # negative
        for i in range(self.num_bag // 2):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))

            if bag_length < 1:
                bag_length = 1
            try:
                indices = random.sample(negative_idx, bag_length)
            except ValueError:
                print('No more negative instances 1')
                break
            negative_idx = list(set(negative_idx) - set(t))

            indices = torch.LongTensor(indices)

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag < 5 # labels in negative bags are all > 5, so this returns all False vector :)

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)


    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            bag_idx = np.repeat(index, bag.shape[0])

        return bag, label, bag_idx


if __name__ == '__main__':
    dataset = ColoredFashionMnistBags3_binary(
                                          mean_bag_length=10,
                                          var_bag_length=0,
                                          num_bag=40000//10,
                                          train=True)


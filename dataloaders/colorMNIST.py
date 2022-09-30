# https://github.com/reiinakano/invariant-risk-minimization

import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.datasets import MNIST

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr

# MIL version, each digit/item is colored, but the label is its original label
# Binarization is performed in the MIL bag construction step.
class ColoredMNIST(datasets.VisionDataset):
  def __init__(self, root='/home/user/datasets/', env='train1', transform=None, target_transform=None, target_num = None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')


  def __getitem__(self, index):
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = "/home/user/datasets/ColoredMNIST/"
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return
    print('Preparing Colored MNIST')
    train_mnist = datasets.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      # binary_label = 0 if label != self.target_num else 1
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
          train1_set.append((Image.fromarray(colored_arr), label))
      elif idx < 40000:
          train2_set.append((Image.fromarray(colored_arr), label))
      else:
          test_set.append((Image.fromarray(colored_arr), label))

    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

class ColoredFashionMNIST(datasets.VisionDataset):
  """
  Colored FashionMNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='/home/user/datasets/', env='train1', transform=None, target_transform=None, target_num=None):
    super(ColoredFashionMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredFashionMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredFashionMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredFashionMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_fashionmnist_dir = "/home/user/datasets/ColoredFashionMNIST/"
    if os.path.exists(os.path.join(colored_fashionmnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_fashionmnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_fashionmnist_dir, 'test.pt')):
      print('Colored FashionMNIST dataset already exists')
      return

    print('Preparing Colored FashionMNIST')
    train_mnist = datasets.FashionMNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
          train1_set.append((Image.fromarray(colored_arr), label))
      elif idx < 40000:
          train2_set.append((Image.fromarray(colored_arr), label))
      else:
          test_set.append((Image.fromarray(colored_arr), label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    # dataset_utils.makedir_exist_ok(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_fashionmnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_fashionmnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_fashionmnist_dir, 'test.pt'))





class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.
    You have two paramters to control the level of bias.
    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.
        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).
        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.
    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9):
        super(ColourBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels)

    def _binary_to_colour(self, data, colour):
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_biased_mnist_dataloader(root, batch_size, data_label_correlation,
                                n_confusing_labels=9, train=True, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation,
                                n_confusing_labels=n_confusing_labels)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

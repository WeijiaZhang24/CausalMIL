# %% [markdown]
# ## Utils

# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.networks_colon import cmil
import torch.nn as nn
import numpy as np
import glob

def generate_batch(path, target='epithelial'):
    bags = []
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.bmp')
        num_ins = len(img_path)

        

        instance_label = [int(target in temp) for temp in img_path]
        label = int(each_path.split('/')[-2])
        
        # if label == 1:
        #     curr_label = np.ones(num_ins,dtype=np.uint8)
        # else:
        #     curr_label = np.zeros(num_ins, dtype=np.uint8)
        for each_img in img_path:
            img_data = np.asarray( imageio.imread(each_img), dtype = np.uint8)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, instance_label, name_img))

    return bags

def mi_collate_img(batch):
    # collate_fn for pytorch DataLoader
    bag = [item[0] for item in batch]
    bag = torch.tensor(np.concatenate(bag, axis = 0))
    
    bag_idx = [item[1] for item in batch]
    bag_idx = torch.tensor(np.concatenate(bag_idx, axis = 0))
    
    bag_label = [item[2] for item in batch]
    bag_label = torch.tensor(bag_label)

    instance_label = [item[3] for item in batch]
    instance_label = torch.tensor(np.concatenate(instance_label, axis = 0))
    return bag, bag_idx, bag_label, instance_label


class mi_imagedata(Dataset):
    def __init__(self, data, cuda, transformations = None, batch_size=32, shuffle=True):
        self.device = torch.device('cuda') 
        self.cuda = cuda
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transforms = transformations
        self.bags = [bag[0] for bag in data]
        self.bag_label =  [max(bag[1]) for bag in data]
        self.instance_label =  [bag[1] for bag in data]

    def __len__(self):
        return len(self.bag_label)

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bag = self.bags[idx]
        if self.transforms is not None:
            temp = [self.transforms(item) for item in bag]
            bag = torch.stack(temp)
        bag_label = self.bag_label[idx]
        bag_idx = np.repeat(idx, bag.shape[0])
        instance_label = self.instance_label[idx]

        return bag, bag_idx, bag_label, instance_label
    
def load_dataset(dataset_path, n_folds, seed=0):
    # load datapath from path
    pos_path = glob.glob(dataset_path+'1//img*')
    neg_path = glob.glob(dataset_path+'0//img*')

    pos_num = len(pos_path)
    neg_num = len(neg_path)

    all_path = pos_path + neg_path

    #num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets  

def map_bag_embeddings(zx_q, zy_q, bag_idx, list_g):
    bag_latent_embeddings = torch.empty(zx_q.shape[0], zy_q.shape[1])
    for _, g in enumerate(list_g):
        group_label = g
        samples_group = bag_idx.eq(group_label).nonzero().squeeze()
        if samples_group.numel() >1 :
            for index in samples_group:
                # print("index: ", index)
                bag_latent_embeddings[index] = zy_q[list_g.index(group_label)]
        else:
            bag_latent_embeddings[samples_group] = zy_q[list_g.index(group_label)]
    return bag_latent_embeddings

def reorder_y(bag_label, bag_idx, list_g):
    def unique_keeporder(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]
    bag_idx = bag_idx.tolist()
    index = unique_keeporder(bag_idx)
    y_reordered = torch.empty(bag_label.shape)
    for i in range(len(list_g)):
        y_reordered[i] = bag_label[index.index(list_g[i])]
    return y_reordered

def get_bag_labels(bag_idx):
    list_bags_labels = []
    bags = (bag_idx).unique()

    for _, g in enumerate(bags):
        bag_label = g.item()
        list_bags_labels.append(bag_label)

    return list_bags_labels

def kaiming_uniform_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
    std = gain/ math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std 
    with torch.no_grad():
        return tensor.uniform_( -bound, bound )
    
def weights_init(layer):
    r"""Apparently in Chainer Lecun normal initialisation was the default one
    """
    if isinstance(layer, nn.Linear):
        layer.bias.data.zero_()
        kaiming_uniform_(layer.weight)
        # torch.nn.init.kaiming_uniform_(layer.bias)
        # torch.nn.init.kaiming_uniform_(layer.weight)

# %%
def get_loss(model, bags, bag_index, bag_label):
    bags = bags.to(torch.device('cuda'))     
    with torch.no_grad():
        elbo, auxiliary_y, reconstruction_proba, KL_zx = \
            model.loss_function(bags, bag_index, bag_label, 1000)
    return elbo, auxiliary_y, reconstruction_proba, KL_zx
 
def get_accuracy(model, bags, bag_idx, bag_label, instance_label):
    with torch.no_grad():
        pred_instance = model.classifier_ins(bags, bag_idx.cpu())
    instance_auc = roc_auc_score(instance_label.cpu(), pred_instance.cpu())
    instance_aucpr = average_precision_score(instance_label.cpu(), pred_instance.cpu())
    
    return  instance_auc,instance_aucpr

# %%
def training_procedure(FLAGS, input_dim, dataset, target):
  device = torch.device('cuda') 

  train_bags = dataset['train']
  test_bags = dataset['test']

  train_bags = train_bags + test_bags
  # convert bag to batch
  train_set = generate_batch(train_bags, target)
  test_set = generate_batch(test_bags, target)


  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=(-90, 90)),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize((123.68/255, 116.779/255, 103.939/255), (0.5,0.5, 0.5)),

    ])
  transform_test= transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
      transforms.Normalize((123.68/255, 116.779/255, 103.939/255), (0.5,0.5, 0.5)),
      ])

  train_data = mi_imagedata(train_set,  FLAGS.cuda, transformations = transform)
  dataloader = DataLoader(train_data, batch_size = FLAGS.batch_size, shuffle=True, num_workers = 4,  collate_fn=mi_collate_img)
  
  model = cmil(FLAGS).to(device)
  model.apply(weights_init)
  model.train()
  auto_encoder_optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.initial_learning_rate, weight_decay=FLAGS.weight_decay)
  best_loss = 100000000.

  for epoch in tqdm(range(0, FLAGS.end_epoch)):
      elbo_epoch = 0
      recon_epoch = 0
      y_epoch = 0
      KL_ins_epoch = 0
      for (i, batch) in enumerate(dataloader):
          bag, bag_idx, bag_label, instance_label = batch
          auto_encoder_optimizer.zero_grad()            
          elbo, class_y_loss, reconstruction_proba, KL_instance = \
              model.loss_function(bag.float().to(device), bag_idx.to(device), bag_label.to(device), epoch)
          elbo.backward()
          auto_encoder_optimizer.step()  
            
          elbo_epoch  += elbo
          recon_epoch += reconstruction_proba
          y_epoch += class_y_loss
          KL_ins_epoch += KL_instance
      elbo_epoch = elbo_epoch / (dataloader.__len__()/batch_size)
      recon_epoch = recon_epoch / (dataloader.__len__()/batch_size)
      y_epoch = y_epoch / (dataloader.__len__()/batch_size)
      KL_ins_epoch = KL_ins_epoch / (dataloader.__len__()/batch_size)
      
      
      # 
      if elbo_epoch < best_loss:
        best_loss = elbo_epoch
        torch.save(model.state_dict(), '/home/weijia/Code/weights/colon_weights.pt')
  
    #   if ((epoch + 1) % 1 ==0):
    #       print('Epoch #' + str(epoch+1) + '..............................................')
    #       print("Train AUC  {:.3f}, Train AUC-PR {:3f}".format (epoch_train_auc, epoch_train_aucpr))
    #       print("Val AUC  {:.3f}, Val AUC-PR {:3f}".format (epoch_val_auc, epoch_val_aucpr))
    #       print("Test ACC: {:.4f}, Test AUC-PR: {:.4f}".format(test_accuracy, test_aucpr))
  
  model.load_state_dict(torch.load('/home/weijia/Code/weights/colon_weights.pt'))
  test_data = mi_imagedata(test_set, FLAGS.cuda, transformations = transform_test)
  testloader = DataLoader(test_data, batch_size = test_data.__len__(), shuffle=False, num_workers = 0,  collate_fn=mi_collate_img)    
  test_bag, test_bag_idx,test_bag_label, test_instance_label = next(iter(testloader))
  test_auc, test_aucpr  = get_accuracy(model, test_bag.float().to(device), test_bag_idx, test_bag_label, test_instance_label)

  test_auc, test_aucpr = get_accuracy(model, test_bag.float().to(device), test_bag_idx, test_bag_label, test_instance_label)
  print("Test AUC: {:.4f}, Test AUC-PR: {:.4f}".format(test_auc, test_aucpr))
 
  return test_aucpr,model

# %%
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection  import ParameterGrid
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import KFold
import imageio

param_grid = {'instance_dim': [128],  'aux_loss_multiplier_y': [10.], 'kl_divergence_coef':[1]}
grid = ParameterGrid(param_grid)    
for params in grid:
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes on which the data set trained")
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3, help="starting learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument('--instance_dim', type=int, default=params['instance_dim'], help="dimension of instance factor latent space")
    parser.add_argument('--reconstruction_coef', type=float, default=1., help="coefficient for reconstruction term")
    parser.add_argument('--kl_divergence_coef', type=float, default=params['kl_divergence_coef'], help="coefficient for instance KL-Divergence loss term")
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=params['aux_loss_multiplier_y'])
    parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
    parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")
    parser.add_argument('--batch_size', type=int, default=128, help="flag to indicate the final epoch of training")
    parser.add_argument('-w', '--warmup', type=int, default=0, metavar='N', help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    FLAGS = parser.parse_args(args=[])

    batch_size = 32

    data_path = '/home/weijia/datasets/ColonCancer/'
    input_dim = (27,27,3)

    n_folds = 5
    dataset = load_dataset(dataset_path=data_path, n_folds=n_folds,seed = 12345678)

    test_aucpr = []

    for ifold in range(5):
        fold_aucpr,model = training_procedure(FLAGS, input_dim, dataset[ifold], target = 'epithelial')
        print('Test AUCPR is: {:.4f}'.format(fold_aucpr))




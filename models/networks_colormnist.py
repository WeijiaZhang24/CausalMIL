''' network with set transformer as bag prior'''
import sys 
sys.path.append("..") 

import torch
import torch.nn as nn
import torch.distributions as dist
from util import reorder_y
import torch.distributions as dist
import numpy as np

def map_bag_embeddings(zx_q, zy_q, bag_idx, list_g):
    bag_latent_embeddings = torch.empty(zx_q.shape[0], zy_q.shape[1], requires_grad= True).to(torch.device('cuda'))
    for _, g in enumerate(list_g):
        group_label = g
        samples_group = bag_idx.eq(group_label).nonzero().squeeze()
        if samples_group.numel() >1 :
            for index in samples_group:
                bag_latent_embeddings[index] = zy_q[list_g.index(group_label)]
        else:
            bag_latent_embeddings[samples_group] = zy_q[list_g.index(group_label)]
    return bag_latent_embeddings

def deep_set_prior(encoded_u, bag_idx):
    bag_encoded = []
    list_bags_labels = []

    bags = (bag_idx).unique()
    for _, g in enumerate(bags):
        bag_label = g.item()
        samples_bag = bag_idx.eq(bag_label).nonzero().squeeze()
        group_encoded =  encoded_u[samples_bag,:]
        sum_encoded = torch.mean(group_encoded, dim = 0, keepdim=True)
        bag_encoded.append(sum_encoded)
        list_bags_labels.append(bag_label)

    bag_encoded = torch.cat(bag_encoded, dim=0).to(torch.device('cuda'))
    return bag_encoded, list_bags_labels

class encoder_x(nn.Module):
    # qzx z_I ~ x
    # Take an instance x and z_B as input, encode the instance level latent z_I
    def __init__(self, latent_dim):
        super(encoder_x, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,32,3,2,1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,2,1),
            nn.ReLU(),
            nn.Conv2d(32,32,3,2,1),
            nn.ReLU()
        )
        self.instance_mu = nn.Linear(in_features=32 * 4 * 4, out_features=latent_dim, bias=True)
        self.instance_logvar = nn.Sequential(nn.Linear(in_features=32 * 4 * 4, out_features=latent_dim, bias=True))
        
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        H = H.flatten(start_dim = 1)
        
        # instance_latent_space_mu = self.instance_mu(H)
        # instance_latent_space_logvar = self.instance_logvar(H)
        return H

    
class decoder_x(nn.Module):
    # p(x| z_I, z_B)
    def __init__(self, latent_dim):
        super(decoder_x, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(latent_dim, 32*4*4), nn.ReLU())
        self.de1 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, 2, 1, 0),nn.ReLU())
        self.de2 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, 2, 1,1), nn.ReLU()) 
        self.de3 = nn.ConvTranspose2d(32, 3, 3, 2, 1, 1)   

    def forward(self, instance_latent_space):
        # x = torch.cat((instance_latent_space,bag_latent_space), dim = -1)
        hidden1 = self.fc1(instance_latent_space)
        hidden2 = hidden1.view(-1, 32, 4, 4)
        hidden3 = self.de1(hidden2)
        loc_img = self.de2(hidden3)
        loc_img = self.de3(loc_img)
        return torch.sigmoid(loc_img)


class encoder_xnou(nn.Module):
    def __init__(self, hidden_dims, latent_dim):
        super(encoder_xnou, self).__init__()

        self.latent_dim = latent_dim
        self.mean = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, inputs):
        return self.mean(inputs), self.logvar(inputs)



class auxiliary_y_fixed(nn.Module):
    def __init__(self, instance_latent_dim, num_classes = 2):
        super(auxiliary_y_fixed, self).__init__()
        self.fc_ins = nn.Linear(instance_latent_dim, 1)
    def forward(self, z_ins, bag_idx, bag_instances, instance_mu, instance_std):
        loc_ins= self.fc_ins(z_ins)
        bags = (bag_idx).unique()
        M = torch.zeros((bags.shape[0], 1)).to(torch.device('cuda')) # A Matrix that stores the prediction for the "max" instance of each bag
        max_z_ins = torch.zeros((bags.shape[0], z_ins.shape[1])).to(torch.device('cuda')) # the ins_code of "max" instance
        max_instances = torch.zeros((bags.shape[0], bag_instances.shape[1])).to(torch.device('cuda')) # the "max" instance of each bag, used for reconstruction
        max_instances_mu = torch.zeros((bags.shape[0], instance_mu.shape[1])).to(torch.device('cuda')) # the "max" instance of each bag, used for reconstruction
        max_instances_std = torch.zeros((bags.shape[0], instance_std.shape[1])).to(torch.device('cuda')) # the "max" instance of each bag, used for reconstruction

        for iter_id, bag in enumerate(bags):
            bag_id = bag.item()
            instances_bag = bag_idx.eq(bag_id).nonzero().squeeze()
            if instances_bag.numel()>0:
                if instances_bag.numel()>1:
                    M_bag, index = torch.max(loc_ins[instances_bag],dim=0)
                    M[iter_id, :] = M_bag
                    max_z_ins[iter_id, :] = z_ins[index]
                    max_instances[iter_id, :] = bag_instances[index]
                    max_instances_mu[iter_id, :] = instance_mu[index]
                    max_instances_std[iter_id, :] = instance_std[index]

                else:
                    M[iter_id, :] = loc_ins[instances_bag]
                    max_z_ins[iter_id, :] = z_ins[instances_bag,:]
                    max_instances[iter_id, :] = bag_instances[instances_bag,:]
                    max_instances_mu[iter_id, :] = instance_mu[index]
                    max_instances_std[iter_id, :] = instance_std[index]        
        # prediction for the max instances, original max instances, latent for max_instances, prediction for all instances
        return M, max_instances, max_z_ins, loc_ins, max_instances_mu, max_instances_std


class cmil_mnist(nn.Module):
    def __init__(self, args):
        super(cmil_mnist, self).__init__()
        self.cuda = args.cuda
        self.instance_latent_dim = args.instance_dim
        self.num_classes = args.num_classes
        self.warmup = args.warmup
    
        self.reconstruction_coef = args.reconstruction_coef
        self.kl_divergence_coef = args.kl_divergence_coef
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.decoder_x= decoder_x(latent_dim=self.instance_latent_dim)
        self.encoder_x = encoder_x(latent_dim=self.instance_latent_dim)
        self.encoder_xnou = encoder_xnou(latent_dim=self.instance_latent_dim, hidden_dims=None)
        self.auxiliary_y_fixed = auxiliary_y_fixed(self.instance_latent_dim, self.num_classes)
        
        self.deep_set_mean = nn.Linear(512,1)
        self.deep_set_logvar = nn.Linear(512, 1)
        self.c = 2 * np.pi * torch.ones(1).to(torch.device('cuda'))
    def forward(self, bag, bag_idx, bag_label):
        # Encode bag prior with deep set 
        intermediate_output = self.encoder_x(bag)
        instance_mu, instance_logvar = self.encoder_xnou(intermediate_output) # encoder params
        instance_std = instance_logvar.mul(0.5).exp_()
        qzx = dist.Normal(instance_mu, instance_std)
        zx_q = qzx.rsample()  # [# of instances, instance_latent_dim] 

        bag_encoded, list_bags_labels = deep_set_prior(intermediate_output, bag_idx)
        prior_mu = self.deep_set_mean(bag_encoded)
        prior_logvar = self.deep_set_logvar(bag_encoded)
        prior_std = prior_logvar.mul(0.5).exp_()
        KL_loss =  0.5 * (prior_mu.pow(2) + prior_std.pow(2) - 2*torch.log(prior_std) - 1).sum(dim=-1).mean()
        # KL_loss = KL_loss +  0.5 * (instance_mu.pow(2) + instance_std.pow(2) - 2*torch.log(instance_std) - 1).mean()

        x_target = bag.flatten(start_dim = 1).contiguous()
        y_hat, x_max, ins_hat, _, instance_mu_hat, instance_std_hat = self.auxiliary_y_fixed(zx_q, bag_idx, x_target, instance_mu, instance_std)
        loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        reordered_y = reorder_y(bag_label, bag_idx, list_bags_labels)
        auxiliary_loss_y= loss(y_hat.squeeze(), reordered_y)
        
        # only calculates the recon_loss for the maximum one per bag
        x_recon = self.decoder_x(ins_hat).flatten(start_dim = 1)
        loss = nn.MSELoss(reduction = 'mean') 
        reconstruction_loss = loss(x_recon,x_max)        

        # lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        lpdf_qz_xu = -0.5 * ( 2*instance_std_hat.log() + (ins_hat - instance_mu_hat).pow(2).div(instance_std_hat.pow(2)))
        lpdf_pzu = -0.5 * (  2*prior_std.log() + ( ins_hat- prior_mu).pow(2).div(prior_std.pow(2)))
        KL_loss = KL_loss + (lpdf_qz_xu.sum(dim=-1) - lpdf_pzu.sum(dim=-1)).mean()

        return reconstruction_loss, KL_loss, auxiliary_loss_y

    def loss_function(self, bag, bag_idx, bag_label, epoch):
        # supervised
        if self.warmup > 0:
            kl_divergence_coef = min([self.kl_divergence_coef, (epoch * 1.) / self.warmup])
        else:
            kl_divergence_coef = self.kl_divergence_coef
        reconstruction_loss, KL_loss, auxiliary_y \
            = self.forward(bag, bag_idx, bag_label)
        
        elbo = (  reconstruction_loss + kl_divergence_coef * KL_loss + self.aux_loss_multiplier_y * auxiliary_y )
        return elbo, auxiliary_y, reconstruction_loss, KL_loss
        
  
    def get_encoding(self, bag, bag_idx):
        with torch.no_grad():
            intermediate_x = self.encoder_x(bag)            
            xu_mu, _ = self.encoder_xnou(intermediate_x)
        return xu_mu
    
    def classifier_ins(self, bag, bag_idx):
        with torch.no_grad():
            intermediate_x = self.encoder_x(bag)            
            x_mu, _ = self.encoder_xnou(intermediate_x)

            x_target = bag.flatten(start_dim = 1)
            _, _, _, pred_ins, _,  _ = self.auxiliary_y_fixed(x_mu, bag_idx, x_target, x_mu, x_mu)
        return torch.sigmoid(pred_ins)


    def reconstruct(self, ins_code):
        with torch.no_grad():
            img =self.decoder_x(ins_code)
        return img

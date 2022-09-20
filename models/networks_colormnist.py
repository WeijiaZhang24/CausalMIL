import torch
import torch.nn as nn
import torch.distributions as dist
from util import get_bag_labels, reorder_y

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
        
        instance_latent_space_mu = self.instance_mu(H)
        instance_latent_space_logvar = self.instance_logvar(H)
        return instance_latent_space_mu, instance_latent_space_logvar

    
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


class auxiliary_y_fixed(nn.Module):
    def __init__(self, instance_latent_dim, num_classes = 2):
        super(auxiliary_y_fixed, self).__init__()
        self.fc_ins = nn.Linear(instance_latent_dim, 1)
    def forward(self, z_ins, bag_idx, bag_instances):
        loc_ins= self.fc_ins(z_ins)
        bags = (bag_idx).unique()
        M = torch.zeros((bags.shape[0], 1)).to(torch.device('cuda')) # A Matrix that stores the prediction for the "max" instance of each bag
        max_z_ins = torch.zeros((bags.shape[0], z_ins.shape[1])).to(torch.device('cuda')) # the ins_code of "max" instance
        max_instances = torch.zeros((bags.shape[0], bag_instances.shape[1])).to(torch.device('cuda')) # the "max" instance of each bag, used for reconstruction

        for iter_id, bag in enumerate(bags):
            bag_id = bag.item()
            instances_bag = bag_idx.eq(bag_id).nonzero().squeeze()
            if instances_bag.numel()>0:
                if instances_bag.numel()>1:
                    M_bag, index = torch.max(loc_ins[instances_bag],dim=0)
                    M[iter_id, :] = M_bag
                    max_z_ins[iter_id, :] = z_ins[index]
                    max_instances[iter_id, :] = bag_instances[index]
                else:
                    M[iter_id, :] = loc_ins[instances_bag]
                    max_z_ins[iter_id, :] = z_ins[instances_bag,:]
                    max_instances[iter_id, :] = bag_instances[instances_bag,:]

        return M, max_instances, max_z_ins, loc_ins


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

        self.auxiliary_y_fixed = auxiliary_y_fixed(self.instance_latent_dim, self.num_classes)
    def forward(self, bag, bag_idx, bag_label):
        # Encode
        # encode instance latents
        instance_mu, instance_logvar = self.encoder_x(bag)
        instance_std = instance_logvar.mul(0.5).exp_()
        qzx = dist.Normal(instance_mu, instance_std)
        zx_q = qzx.rsample()  # [# of instances, instance_latent_dim]
        
        list_g = get_bag_labels(bag_idx)

        # kl-divergence error for bag latent space should be KL( q(z_y|x) || p(z_y|y) )
        #reorder by the same order as bag_latent_embeddings
        reordered_y = reorder_y(bag_label, bag_idx, list_g).to(torch.device('cuda'))

        # kl-divergence error for instance latent space
        KL_zx =  0.5 * (instance_mu.pow(2) + instance_std.pow(2) - 2*torch.log(instance_std) - 1).mean()
        
        x_target = bag.flatten(start_dim = 1).contiguous()
        y_hat, x_max, ins_hat, _ = self.auxiliary_y_fixed(zx_q, bag_idx, x_target)
        loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        auxiliary_loss_y= loss(y_hat.squeeze(), reordered_y)
        
        # only calculates the recon_loss for the maximum one per bag
        x_recon = self.decoder_x(ins_hat).flatten(start_dim = 1)
        loss = nn.MSELoss(reduction = 'mean') 
        reconstruction_loss = loss(x_recon,x_max)        
        
        return reconstruction_loss, KL_zx, auxiliary_loss_y

    def loss_function(self, bag, bag_idx, bag_label, epoch):
        # supervised
        if self.warmup > 0:
            kl_divergence_coef = min([self.kl_divergence_coef, (epoch * 1.) / self.warmup])
        else:
            kl_divergence_coef = self.kl_divergence_coef
        reconstruction_loss, KL_zx, auxiliary_y \
            = self.forward(bag, bag_idx, bag_label)
        
        elbo = (  reconstruction_loss + kl_divergence_coef * KL_zx + self.aux_loss_multiplier_y * auxiliary_y )
        return elbo, auxiliary_y, reconstruction_loss, KL_zx
        
    def get_encoding(self, bag, bag_idx, threshold=0.5, L=10):
        with torch.no_grad():
            ins_loc, ins_scale = self.encoder_x.forward(bag)
            
        return ins_loc
    
    def classifier_ins(self, bag, bag_idx):
        with torch.no_grad():
            ins_loc, _ = self.encoder_x.forward(bag)            
            # zy_q_loc, zy_q_scale = self.encoder_y.forward(bag)
            
            x_target = bag.flatten(start_dim = 1)
            _, _, _, pred_ins = self.auxiliary_y_fixed(ins_loc, bag_idx, x_target)
        # return pred_ins
        return torch.sigmoid(pred_ins)


    def reconstruct(self, ins_code):
        with torch.no_grad():
            img =self.decoder_x(ins_code)
        return img

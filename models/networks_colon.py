import torch
import torch.nn as nn
import torch.distributions as dist
from util import get_bag_labels, reorder_y


class decoder_x(nn.Module):
    # p(x| z_I, z_B)
    def __init__(self, instance_latent_dim):
        super(decoder_x, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(instance_latent_dim, 48*5*5, bias=True), nn.ReLU())
        self.up1 = nn.Upsample(10)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(48, 32, kernel_size=3, bias=True), 
                                  nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size = 4, bias=True), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, stride=1))

    def forward(self, instance_latent_space):
        # x = torch.cat((instance_latent_space,bag_latent_space), dim = -1)
        hidden1 = self.fc1(instance_latent_space)
        hidden2 = hidden1.view(-1, 48, 5, 5)
        hidden2 = self.up1(hidden2)
        hidden3 = self.de1(hidden2)
        hidden3 = self.up2(hidden3)
        loc_img = self.de2(hidden3)
        loc_img = self.de3(loc_img)
        return loc_img  

class encoder_x(nn.Module):
    # qzx z_I ~ x
    # Take an instance x and z_B as input, encode the instance level latent z_I
    def __init__(self, instance_latent_dim):
        super(encoder_x, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.instance_mu = nn.Linear(in_features=48 * 5 * 5, out_features=instance_latent_dim, bias=True)
        self.instance_logvar = nn.Sequential(nn.Linear(in_features=48 * 5 * 5, 
                                                       out_features=instance_latent_dim, bias=True))
        
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.flatten(start_dim = 1)
        
        instance_latent_space_mu = self.instance_mu(H)
        instance_latent_space_logvar = self.instance_logvar(H)
        return instance_latent_space_mu, instance_latent_space_logvar


class auxiliary_y_fixed(nn.Module):
    def __init__(self, instance_latent_dim, num_classes = 2):
        super(auxiliary_y_fixed, self).__init__()
        self.fc_ins = nn.Linear(instance_latent_dim, 1)

    def forward(self, z_ins, bag_idx, bag_instances):
        loc_ins= self.fc_ins(z_ins)
        bags = (bag_idx).unique()
        M = torch.zeros((bags.shape[0], 1)).to(torch.device('cuda')) # A Matrix that stores the max prediction of each bag
        max_z_ins = torch.zeros((bags.shape[0], z_ins.shape[1])).to(torch.device('cuda')) # the ins_code of "max" instance
        max_instances = torch.zeros((bags.shape[0], 2187)).to(torch.device('cuda')) # the "max" original instance of each bag, used for reconstruction

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

class cmil(nn.Module):
    def __init__(self, args):
        super(cmil, self).__init__()
        self.cuda = args.cuda
        self.instance_latent_dim = args.instance_dim
        self.num_classes = args.num_classes
        self.warmup = args.warmup
        
        self.reconstruction_coef = args.reconstruction_coef
        self.kl_divergence_coef = args.kl_divergence_coef
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.decoder_x= decoder_x(self.instance_latent_dim)
        self.encoder_x = encoder_x(self.instance_latent_dim)
        self.auxiliary_y_fixed = auxiliary_y_fixed(self.instance_latent_dim)
    def forward(self, bag, bag_idx, bag_label):
        # Encode
        # encode instance latents
        instance_mu, instance_logvar = self.encoder_x(bag)
        instance_var = instance_logvar.exp_()
        qzx = dist.Normal(instance_mu, instance_logvar)
        zx_q = qzx.rsample()  # [# of instances, instance_latent_dim]
        
        #reorder by the same order as bag_latent_embeddings
        list_g = get_bag_labels(bag_idx)
        reordered_y = reorder_y(bag_label, bag_idx, list_g).to(torch.device('cuda'))
        
        # kl-divergence error for instance latent space
        KL_zx = - 0.5 * (instance_mu.pow(2) + instance_var - torch.log(instance_var) - 1).mean()

        # probablistic reconstruct samples
        x_target = bag.flatten(start_dim = 1)
        y_hat, x_max, ins_hat, _ = self.auxiliary_y_fixed(zx_q, bag_idx, x_target)
        loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        auxiliary_loss_y= loss(y_hat.squeeze(), reordered_y.to(torch.device('cuda')))
        

        x_recon = self.decoder_x(ins_hat).flatten(start_dim = 1)
        loss = nn.MSELoss(reduction = 'mean') 
        reconstruction_loss = loss(x_recon,x_max)        
        
        return reconstruction_loss, KL_zx, auxiliary_loss_y

    def loss_function(self, bag, bag_idx, bag_label, epoch):
        # supervised
        if self.warmup > 0:
            # kl_divergence_coef = min([self.kl_divergence_coef, (epoch * 1.) / self.warmup])
            # kl_divergence_coef2 = min([self.kl_divergence_coef2, (epoch * 1.) / self.warmup])
            kl_divergence_coef = self.kl_divergence_coef
            if epoch > self.warmup:
                aux_loss_multiplier_y =  self.aux_loss_multiplier_y
            else:
                aux_loss_multiplier_y = 0
        else:
            kl_divergence_coef = self.kl_divergence_coef
            aux_loss_multiplier_y = self.aux_loss_multiplier_y
        reconstruction_proba, KL_zx, auxiliary_y \
            = self.forward(bag, bag_idx, bag_label)
        
        reconstruction_proba = reconstruction_proba 
        KL_zx = KL_zx
        auxiliary_y = auxiliary_y
        elbo = (  reconstruction_proba - kl_divergence_coef * KL_zx + aux_loss_multiplier_y * auxiliary_y )
        return elbo, auxiliary_y, reconstruction_proba, KL_zx
        
    def classifier_ins(self, bag, bag_idx):
        with torch.no_grad():
            ins_loc, _ = self.encoder_x.forward(bag)            
            
            x_target = bag.flatten(start_dim = 1)
            _, _, _, pred_ins = self.auxiliary_y_fixed(ins_loc, bag_idx, x_target)
        return pred_ins


    def get_encoding(self, bag, bag_idx, threshold=0.5, L=10):
        """
        classify the bag label of the instances
        """
        with torch.no_grad():
            ins_loc, ins_scale = self.encoder_x.forward(bag)
            
        return ins_loc
        
    def reconstruct(self, ins_code):
        """
        reconstruction from latent representations
        """
        with torch.no_grad():
            img =self.decoder_x(ins_code)
        return img        
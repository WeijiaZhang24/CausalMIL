import torch
import torch.nn as nn
import torch.distributions as dist
from util import get_bag_labels, reorder_y

class encoder_x(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(encoder_x, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(self.in_channels, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )

        self.mean = nn.Linear(200, latent_dim)
        self.logvar = nn.Linear(200, latent_dim)

    def forward(self, inputs):
        # # Batch size
        # batch_size = inputs.size(0)
        # Encoded feature map
        hidden = self.encoder(inputs)
        # Reshape
        hidden = torch.flatten(hidden, start_dim=1)
        # Calculate mean and (log)variance
        mean, logvar = self.mean(hidden), self.logvar(hidden)

        return mean, logvar

    
class decoder_x(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(decoder_x, self).__init__()

        self.in_channels = in_channels
        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, self.in_channels, 4, 2, 1)  # 1x28x28
        )

    def forward(self, input):
        h = self.decoder(input.view(input.size(0), input.size(1), 1, 1))
        # return torch.sigmoid(h.view(-1, 28 * 28)) 
        return torch.sigmoid(h)


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
        self.decoder_x= decoder_x(latent_dim=self.instance_latent_dim, hidden_dims=None, in_channels=args.in_channels)
        self.encoder_x = encoder_x(latent_dim=self.instance_latent_dim, hidden_dims=None, in_channels=args.in_channels)

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

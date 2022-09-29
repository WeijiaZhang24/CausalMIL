import torch.nn as nn
import torch

def map_bag_embeddings(zx_q, zy_q, bag_idx, list_g):
    bag_latent_embeddings = torch.empty(zx_q.shape[0], zy_q.shape[1], device = torch.device('cuda'))
    for _, g in enumerate(list_g):
        group_label = g
        samples_group = bag_idx.eq(group_label).nonzero().squeeze()
        if samples_group.numel() >1 :
            for index in samples_group:
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
    y_reordered = torch.empty(bag_label.shape).to(torch.device('cuda'))
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

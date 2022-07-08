"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import math


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()

        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}

class KNNConLoss(nn.Module):
    def __init__(self, temperature=0.05, topK=10):
        super(KNNConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.topK=topK
        print(f"\n Initializing KNNConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        all = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        all = all.masked_select(mask).view(2 * batch_size, -1)
        #print(all.masked_select(mask).shape)
        pos_2, _ = all.topk(self.topK, dim=-1, largest=True)
        #print(pos_2.shape)
        pos = pos.unsqueeze(1)
        pos_sample = torch.cat([pos, pos_2], dim=-1)
        neg_sample, _ = all.topk(batch_size*2-2-self.topK, dim=-1, largest=False)
        #print(pos_sample.shape, neg_sample.shape)

        pos_sample = pos_sample.contiguous().view([-1, 1])
        neg_sample = neg_sample.repeat([1, self.topK+1])
        neg_sample = neg_sample.view([-1, batch_size*2-2-self.topK])
        #print(pos_sample.shape, neg_sample.shape)

        neg_mean = torch.mean(neg_sample)
        pos_n = torch.mean(pos_sample)
        Ng = neg_sample.sum(dim=-1)
        loss_pos = (- torch.log(pos_sample / (Ng + pos_sample))).mean()
        #print(loss_pos)

        #exit()

        return {"loss": loss_pos, "pos_mean": pos_n.detach().cpu().numpy(), "neg_mean": neg_mean.detach().cpu().numpy(),
                "pos": pos_sample.detach().cpu().numpy(), "neg": neg_sample.detach().cpu().numpy()}


class MaskInstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(MaskInstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_temprature_samples(self, batch_size, c_i, c_j):
        c = torch.cat((c_i, c_j), dim=0)
        t_mask = torch.matmul(c, c.T)
        t_mask[t_mask < self.temperature] = self.temperature
        #t_mask[t_mask >= self.temperature] = 1
        for i in range(batch_size):
            t_mask[i, batch_size + i] = self.temperature
            t_mask[batch_size + i, i] = self.temperature
        #logger.debug(t_mask)

        return t_mask

    def positive_selection(self, batch_size, c_i, c_j):
        c = torch.cat((c_i, c_j), dim=0)
        t_mask = torch.matmul(c, c.T)

    def forward(self, z_i, z_j, c_i, c_j):
        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)


        temperature_mask = self.mask_temprature_samples(self.batch_size, c_i, c_j)
        temperature_mask = temperature_mask.detach()
        #sim = torch.matmul(z, z.T) / self.temperature
        sim = torch.matmul(z, z.T) / temperature_mask
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        #print(positive_samples)
        #print(negative_samples)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss





class self_KNNLoss(nn.Module):
    def __init__(self, batch_size, temperature, device, topK):
        super(self_KNNLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.threshold = 0.5
        self.topK = topK
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def pair_cosine_similarity(self, x, x_adv, eps=1e-8):
        n = x.norm(p=2, dim=1, keepdim=True)
        n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
        # print(x.shape)
        # print(x_adv.shape)
        # print(n.shape)
        # print(n_adv.shape)
        # print((n * n.t()).shape)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
                    x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

    def forward(self, z_i, z_j, c_i, c_j):
        batch_size = z_i.shape[0]

        mask = torch.matmul(c_i, c_i.T)
        mask[mask > self.threshold] = 1
        mask[mask <= self.threshold] = 0
        for i in range(batch_size):
            mask[i, i] = 1

        index = torch.zeros(batch_size, batch_size)

        x, x_adv, x_c_i = self.pair_cosine_similarity(z_i, z_j)
        x_c_j = x_c_i.T

        x = torch.exp(x / self.temperature)
        x_adv = torch.exp(x_adv / self.temperature)
        x_c_i = torch.exp(x_c_i / self.temperature)
        x_c_j = torch.exp(x_c_j / self.temperature)

        _, top_x = x.topk(self.topK + 1 , dim=1)

        for i in range(top_x.shape[0]):
            for j in range(self.topK + 1):
                index[i, top_x[i, j]] = 1

        for i in range(batch_size):
            index[i, i] = 0

        index_x_copy = index.clone()
        index[index_x_copy == 0] = 0
        index[index_x_copy == 1] = 1

        index_x_c = index.clone()
        for i in range(batch_size):
            index_x_c[i, i] = 1

        x = x[index == 1].reshape([batch_size, self.topK])
        x_adv = x_adv[index == 1].reshape([batch_size, self.topK])
        x_c_i = x_c_i[index_x_c == 1].reshape([batch_size, self.topK+1])
        x_c_j = x_c_j[index_x_c == 1].reshape([batch_size, self.topK+1])

        mask_x = mask[index == 1].reshape([batch_size, self.topK])
        mask_x_reverse = (~(mask_x.bool())).long()

        mask_x_adv = mask[index == 1].reshape([batch_size, self.topK])
        mask_x_adv_reverse = (~(mask_x_adv.bool())).long()

        mask_x_c_i = mask[index_x_c == 1].reshape([batch_size, self.topK+1])
        mask_x_c_i_reverse = (~(mask_x_c_i.bool())).long()

        mask_x_c_j = mask[index_x_c == 1].reshape([batch_size, self.topK+1])
        mask_x_c_j_reverse = (~(mask_x_c_j.bool())).long()

        x1 = torch.cat((x, x_c_i), dim=1)
        x2 = torch.cat((x_adv, x_c_j), dim=1)

        mask_1 = torch.cat((mask_x, mask_x_c_i), dim=1)
        mask_2 = torch.cat((mask_x_adv, mask_x_c_j), dim=1)

        mask_x1_reverse = torch.cat((mask_x_reverse, mask_x_c_i_reverse), dim=1)
        mask_x2_reverse = torch.cat((mask_x_adv_reverse, mask_x_c_j_reverse), dim=1)

        dis_x1 = (x1 * mask_1) / (x1.sum(1).unsqueeze(1)) + mask_x1_reverse
        dis_x2 = (x2 * mask_2) / (x2.sum(1).unsqueeze(1)) + mask_x2_reverse

        loss = (torch.log(dis_x1).sum(1) + torch.log(dis_x2).sum(1)) / (mask_x.sum(1)*2 + 1)

        return -loss.mean()




class KNN_InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device, topK):
        super(KNN_InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.threshold = 0.5
        self.topK = topK
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, z_i, z_j, c_i, c_j):
        self.batch_size = len(z_i)
        label_mask_i = torch.matmul(c_i, c_i.T)
        label_mask_j = torch.matmul(c_j, c_j.T)
        label_mask = (label_mask_i + label_mask_j) / 2   # 可调
        #print(label_mask[label_mask > self.threshold])

        for i in range(self.batch_size):
            label_mask[i, i] = 1
        label_mask[label_mask > self.threshold] = 1
        label_mask[label_mask <= self.threshold] = 0
        pos_mask_index = label_mask.bool()
        neg_mask_index = ~ pos_mask_index

        cos_sim = torch.matmul(z_i, z_j.T)
        cos_sim_i = torch.matmul(z_i, z_i.T)
        cos_sim_j = torch.matmul(z_j, z_j.T)

        print(cos_sim.shape)
        print(pos_mask_index.shape, neg_mask_index.shape)
        print("------------------------------------")

        feature_value = cos_sim.masked_select(pos_mask_index)
        feature_value_i = cos_sim_i.masked_select(pos_mask_index)
        feature_value_j = cos_sim_j.masked_select(pos_mask_index)
        print(feature_value.shape, feature_value_i.shape, feature_value_j.shape)

        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        print(pos_sample.shape)
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)
        print(pos_sample.shape)
        print("------------------------------------")

        feature_value = cos_sim.masked_select(neg_mask_index)
        feature_value_i = cos_sim_i.masked_select(neg_mask_index)
        feature_value_j = cos_sim_j.masked_select(neg_mask_index)
        print(feature_value.shape, feature_value_i.shape, feature_value_j.shape)

        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        print(neg_sample.shape)
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)
        print(neg_sample.shape)
        print("####################################")

        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        #pos_number_bool = torch.full_like(pos_number, 0).cuda()
        #print("pos_number:", pos_number, pos_number.shape)

        #pos_number[pos_number >= self.topK] = 1
        #pos_number[pos_number < self.topK] = 0
        #pos_number = pos_number.bool()
        #print("pos_number_bool:", pos_number_bool, pos_number_bool.shape)

        #mask_id = torch.full_like(cos_sim, 0).cuda()
        #for i in range(self.batch_size):
        #    mask_id[i, i] = 1
        #print(pos_sample[mask_id==1])
        #exit()
        #pos_sample_selected_1 = pos_sample[mask_id==1]
        #pos_sample_selected_1 = pos_sample_selected_1.unsqueeze(1)

        #for i in range(self.batch_size):
        #    pos_sample[i, i] = -np.inf
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        #pos_sample_selected = torch.cat([pos_sample_selected_1, pos_sample_selected_2], dim=-1)
        print(pos_sample.shape)
        #pos_sample_1 = pos_sample[pos_number]
        #pos_sample_2 = pos_sample[~ pos_number]
        #pos_sample_2, _ = pos_sample_2.topk(1, dim=-1)
        #print(pos_sample_1.shape, pos_sample_2.shape)
        #exit()
        pos_sample = pos_sample.contiguous().view([-1, 1])
        #pos_sample_1 = pos_sample_1.contiguous().view([-1, 1])
        #pos_sample_2 = pos_sample_2.contiguous().view([-1, 1])
        print(pos_sample.shape)

        print("---------------------------------------------")
        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)

        #neg_sample_1 = neg_sample[pos_number]
        #neg_sample_2 = neg_sample[~ pos_number]
        #print(neg_sample_1.shape, neg_sample_2.shape)

        neg_sample = neg_sample.repeat([1, pos_min])
        neg_sample = neg_sample.view([-1, neg_min])
        #neg_sample_1 = neg_sample_1.repeat([1, self.topK])
        #neg_sample_1 = neg_sample_1.view([-1, neg_min])
        #neg_sample_2 = neg_sample_2.repeat([1, 1])
        #neg_sample_2 = neg_sample_2.view([-1, neg_min])
        print(neg_sample.shape)
        print("---------------------------------------------")

        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        print(logits_con.shape)
        logits_con /= self.temperature
        #logits_con_1 = torch.cat([pos_sample_1, neg_sample_1], dim=-1)
        #print(logits_con_1.shape)
        #logits_con_1 /= self.temperature
        #logits_con_2 = torch.cat([pos_sample_2, neg_sample_2], dim=-1)
        #print(logits_con_2.shape)
        #logits_con_2 /= self.temperature
        print("---------------------------------------------")

        loss = 0
        labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss()(logits_con, labels_con)

        '''
        if logits_con_1.shape[0]!=0:
            labels_con_1 = torch.zeros(logits_con_1.shape[0], dtype=torch.long).cuda()
            loss_1 = nn.CrossEntropyLoss()(logits_con_1, labels_con_1)
            loss+=loss_1

        if logits_con_2.shape[0]!=0:
            labels_con_2 = torch.zeros(logits_con_2.shape[0], dtype=torch.long).cuda()
            loss_2 = nn.CrossEntropyLoss()(logits_con_2, labels_con_2)
            loss += loss_2
        '''


        return loss




class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        #print(positive_samples)
        #print(negative_samples)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


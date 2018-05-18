"""
model class
KernelPooling: the kernel pooling layer
KNRM: base class of KNRM, can choose to:
    learn distance metric
    learn entity attention
"""
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np


class knrm(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, opt, weights):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(knrm, self).__init__()

        self.word_emb = nn.Embedding(len(weights), opt.d_word_vec, padding_idx = 0)
        self.word_emb.weight.data.copy_(torch.from_numpy(weights))
        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, opt.n_bins)
        self.dense = nn.Linear(opt.n_bins, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):

        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum


    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed = self.word_emb(inputs_q)
        d_embed = self.word_emb(inputs_d)
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = torch.squeeze(F.tanh(self.dense(log_pooling_sum)), 1)
        return output

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

    def __init__(self, opt, embedding_init):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(knrm, self).__init__()
        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()

        self.d_word_vec = opt.d_word_vec
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.wrd_emb = nn.Embedding(len(embedding_init), opt.d_word_vec)
        self.wrd_emb.weight.data.copy_(torch.from_numpy(embedding_init))
        self.dense_f = nn.Linear(opt.n_bins * 9, 1, 1)
        self.tanh = nn.Tanh()

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, opt.d_word_vec)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, opt.d_word_vec)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, opt.d_word_vec)),
            nn.ReLU()
        )


    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):

        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * atten_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum



    def forward(self, inputs_qwt, inputs_dwt, inputs_qwm, inputs_dwm):
        qw_embed = self.wrd_emb(inputs_qwt)
        dw_embed = self.wrd_emb(inputs_dwt)
        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi (dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec))) + 0.000000001
        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
        mask_qw = inputs_qwm.view(inputs_qwt.size()[0], inputs_qwt.size()[1], 1)
        mask_dw = inputs_dwm.view(inputs_dwt.size()[0], 1, inputs_dwt.size()[1], 1)
        mask_qwu = mask_qw[:, :inputs_qwt.size()[1] - (1 - 1), :]
        mask_qwb = mask_qw[:, :inputs_qwt.size()[1] - (2 - 1), :]
        mask_qwt = mask_qw[:, :inputs_qwt.size()[1] - (3 - 1), :]
        mask_dwu = mask_dw[:, :, :inputs_dwt.size()[1] - (1 - 1), :]
        mask_dwb = mask_dw[:, :, :inputs_dwt.size()[1] - (2 - 1), :]
        mask_dwt = mask_dw[:, :, :inputs_dwt.size()[1] - (3 - 1), :]
        log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm, mask_qwu, mask_dwu)
        log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm, mask_qwu, mask_dwt)
        log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm, mask_qwu, mask_dwb)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm, mask_qwb, mask_dwu)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm, mask_qwt, mask_dwu)

        log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm, mask_qwb, mask_dwb)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm, mask_qwb, mask_dwt)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm, mask_qwt, mask_dwb)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm, mask_qwt, mask_dwt)
        log_pooling_sum = torch.cat([ log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
            log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        output = torch.squeeze(F.tanh(self.dense_f(log_pooling_sum)), 1)
        return output

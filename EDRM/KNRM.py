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
        self.batch_size = opt.batch_size
        self.ent_q_size= 5
        self.ent_d_size = 10
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.wrd_emb = nn.Embedding(opt.wrd_vocab_size, opt.d_word_vec)
        self.wrd_emb.weight.data.copy_(torch.from_numpy(embedding_init))
        self.ent_emb = nn.Embedding(opt.ent_vocab_size, 128)
        self.car_emb = nn.Embedding(opt.car_vocab_size, 128)
        self.relu = nn.ReLU()
        self.dense_f = nn.Linear(opt.n_bins * 16, 1, 1)
        self.dense_qf = nn.Linear(128 * 2, 128, 128)
        self.dense_bow = nn.Linear(opt.d_word_vec, 128)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.conv_des = nn.Sequential(
            nn.Conv1d(1, 128, opt.d_word_vec * opt.window_size, stride=opt.d_word_vec),
            nn.ReLU(),
            nn.MaxPool1d(20 - opt.window_size + 1),

        )
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

    def forward(self, inputs_qw, inputs_qe, inputs_dw, inputs_de):
        inputs_qwt, inputs_qwm = inputs_qw
        inputs_qei, inputs_qet, inputs_qew, inputs_qem = inputs_qe
        inputs_dwt, inputs_dwm = inputs_dw
        inputs_dei, inputs_det, inputs_dew, inputs_dem = inputs_de
        self.batch_size = inputs_qwt.size()[0]
        qw_embed = self.wrd_emb(inputs_qwt)
        qe_embed = self.wrd_emb(inputs_qet)
        dw_embed = self.wrd_emb(inputs_dwt)
        de_embed = self.wrd_emb(inputs_det)
        qi_embed = self.ent_emb(inputs_qei)
        di_embed = self.ent_emb(inputs_dei)
        qt_embed = self.car_emb(inputs_qew)
        dt_embed = self.car_emb(inputs_dew)
        qw_bow = torch.sum(self.dense_bow(qw_embed), 1, keepdim=True)
        atten_qew = torch.sum(qw_bow * qt_embed.view(self.batch_size, -1, 128), 2).view(self.batch_size, self.ent_q_size, -1, 1)
        atten_qew = self.softmax(atten_qew)
        qew = torch.sum(atten_qew * qt_embed.view(self.batch_size, self.ent_q_size, -1, 128), 2)
        dw_bow = torch.sum(self.dense_bow(dw_embed), 1, keepdim=True)
        atten_dew = torch.sum(dw_bow * dt_embed.view(self.batch_size, -1, 128), 2).view(self.batch_size, self.ent_d_size, -1, 1)
        atten_dew = self.softmax(atten_dew)
        dew = torch.sum(atten_dew * dt_embed.view(self.batch_size, self.ent_d_size, -1, 128), 2)
        qe_embed_conv = self.conv_des(qe_embed.view(self.batch_size * self.ent_q_size, 1, -1))
        de_embed_conv = self.conv_des(de_embed.view(self.batch_size * self.ent_d_size, 1, -1))
        qe = qe_embed_conv.view(self.batch_size, -1, 128)
        de = de_embed_conv.view(self.batch_size, -1, 128)
        qi = qi_embed.view(self.batch_size, -1, 128)
        di = di_embed.view(self.batch_size, -1, 128)
        qe = qe + qew
        de = de + dew
        qs_embed = qi + qe
        ds_embed = di + de
        qeu_embed_norm = F.normalize(qs_embed, p=2, dim=2, eps=1e-10)
        deu_embed_norm = torch.transpose(F.normalize(ds_embed, p=2, dim=2, eps=1e-10), 1, 2)
        mask_qw = inputs_qwm.view(inputs_qwt.size()[0], inputs_qwt.size()[1], 1)
        mask_qeu = inputs_qem.view(qe.size()[0], qe.size()[1], 1)
        mask_dw = inputs_dwm.view(inputs_dwt.size()[0], 1, inputs_dwt.size()[1], 1)
        mask_deu = inputs_dem.view(de.size()[0], 1, de.size()[1], 1)
        qwu_embed = torch.transpose(
            torch.squeeze(self.conv_uni(qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1,
            2)
        qwb_embed = torch.transpose(
            torch.squeeze(self.conv_bi(qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1,
            2)
        qwt_embed = torch.transpose(
            torch.squeeze(self.conv_tri(qw_embed.view(inputs_qwt.size()[0], 1, -1, self.d_word_vec))), 1,
            2)
        dwu_embed = torch.squeeze(
            self.conv_uni(dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec)))
        dwb_embed = torch.squeeze(
            self.conv_bi(dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec)))
        dwt_embed = torch.squeeze(
            self.conv_tri(dw_embed.view(inputs_dwt.size()[0], 1, -1, self.d_word_vec)))
        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
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
        log_pooling_sum_ewuu = self.get_intersect_matrix(qeu_embed_norm, dwu_embed_norm, mask_qeu, mask_dwu)
        log_pooling_sum_ewub = self.get_intersect_matrix(qeu_embed_norm, dwb_embed_norm, mask_qeu, mask_dwb)
        log_pooling_sum_ewut = self.get_intersect_matrix(qeu_embed_norm, dwt_embed_norm, mask_qeu, mask_dwt)
        log_pooling_sum_weuu = self.get_intersect_matrix(qwu_embed_norm, deu_embed_norm, mask_qwu, mask_deu)
        log_pooling_sum_webu = self.get_intersect_matrix(qwb_embed_norm, deu_embed_norm, mask_qwb, mask_deu)
        log_pooling_sum_wetu = self.get_intersect_matrix(qwt_embed_norm, deu_embed_norm, mask_qwt, mask_deu)
        log_pooling_sum_eeuu = self.get_intersect_matrix(qeu_embed_norm, deu_embed_norm, mask_qeu, mask_deu)


        log_pooling_sum = torch.cat([log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub,
                                     log_pooling_sum_wwbu, log_pooling_sum_wwtu, log_pooling_sum_wwbb,
                                     log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt,
                                     log_pooling_sum_ewuu, log_pooling_sum_ewub, log_pooling_sum_ewut,
                                     log_pooling_sum_weuu, log_pooling_sum_webu, log_pooling_sum_wetu, log_pooling_sum_eeuu
                                     ], 1)
        output = torch.squeeze(F.tanh(self.dense_f(log_pooling_sum)), 1)
        return output

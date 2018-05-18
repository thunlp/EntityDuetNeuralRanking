''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from collections import defaultdict
from torch.autograd import Variable

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, file_name=None, ent_des=None, ent_wrd=None, ent2idx=None, ent_des_dict = None, ent_wrd_dict = None, cuda=True, batch_size=64, test=False):
        self._ent_des = ent_des
        self._ent_wrd = ent_wrd
        self._ent2idx = ent2idx
        self._ent_wrd_dict = ent_wrd_dict
        self._ent_des_dict = ent_des_dict
        self._cuda= cuda
        self._batch_size = batch_size
        self._test = test
        self._file_name = file_name


    def cover_text2int(self, sentence):
        tokens = sentence.strip().split(",")
        return [int(token) for token in tokens]


    def cover_text2inte(self, sentence, ent2idx, ent_des_dict, ent_wrd_dict, max_entity):
        tokens = sentence.strip().split(",")
        ent_tokens = [token for token in tokens if token in ent_des_dict or token in ent2idx]
        if len(ent_tokens) != 0:
            ids_list = list()
            voc_list = list()
            con_list = list()
            wrd_list = list()
            token_counter = defaultdict(int)
            for token in ent_tokens:
                key = (ent2idx.get(token, 0), ent_des_dict.get(token, 0), ent_wrd_dict.get(token, 0))
                if len(token_counter) < max_entity:
                    token_counter[key] += 1
                elif token in token_counter:
                    token_counter[key] += 1
            for key, value in token_counter.items():
                voc_list.append(key[0])
                ids_list.append(key[1])
                wrd_list.append(key[2])
                con_list.append(value)
            return (voc_list, ids_list, wrd_list, con_list)
        return ([0], [0], [0], [0])


    def read_instances_from_file(self, inst_file):
        ''' Convert file into word seq lists '''
        qw = []
        qe = []
        dw_pos = []
        de_pos = []
        dw_neg = []
        de_neg = []
        with open(inst_file) as f:
            for line in f:
                tokens = line.decode("utf8").strip().split("\t")
                if len(tokens) != 7:
                    error_sent_count += 1
                    continue
                score = float(tokens[3])
                if score == 0 or abs(score) < 0.1:
                    continue
                qw.append(self.cover_text2int(tokens[0]))
                qe.append(self.cover_text2inte(tokens[4], self._ent2idx, self._ent_des_dict, self._ent_wrd_dict, 5))
                if score > 0:
                    dw_pos.append(self.cover_text2int(tokens[1]))
                    dw_neg.append(self.cover_text2int(tokens[2]))
                    de_pos.append(self.cover_text2inte(tokens[5], self._ent2idx, self._ent_des_dict, self._ent_wrd_dict, 10))
                    de_neg.append(self.cover_text2inte(tokens[6], self._ent2idx, self._ent_des_dict, self._ent_wrd_dict, 10))
                elif score < 0:
                    dw_pos.append(self.cover_text2int(tokens[2]))
                    dw_neg.append(self.cover_text2int(tokens[1]))
                    de_pos.append(self.cover_text2inte(tokens[6], self._ent2idx, self._ent_des_dict, self._ent_wrd_dict, 10))
                    de_neg.append(self.cover_text2inte(tokens[5], self._ent2idx, self._ent_des_dict, self._ent_wrd_dict, 10))
                if len(qw) >= self._batch_size:
                    yield qw, dw_pos, dw_neg, qe, de_pos, de_neg
                    qw = []
                    qe = []
                    dw_pos = []
                    de_pos = []
                    dw_neg = []
                    de_neg = []



    def pad_to_longest(self, insts, max_len):
        ''' Pad the instance to the max seq length in batch '''
        inst_data = np.array([
            inst[:max_len] + [0] * (max_len - len(inst[:max_len]))
            for inst in insts])
        mask = np.zeros((inst_data.shape[0], inst_data.shape[1]))
        for b in range(len(inst_data)):
            for i in range(len(inst_data[b])):
                if inst_data[b, i] > 0:
                    mask[b, i] = 1
        mask_tensor = Variable(
                torch.FloatTensor(mask), requires_grad = False)
        inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self._test)
        if self._cuda:
            mask_tensor = mask_tensor.cuda()
            inst_data_tensor = inst_data_tensor.cuda()
        return (inst_data_tensor, mask_tensor)


    def pad_to_longest_entity(self, insts, max_entity):
        ''' Pad the instance to the max seq length in batch '''
        insts_tok = list()
        insts_ids = list()
        insts_msk = list()
        insts_wrd = list()
        for inst in insts:
            ids_list, tok_list, wrd_list, con_list = inst
            assert len(tok_list) == len(con_list) == len(ids_list) == len(wrd_list)
            insts_ids.extend(ids_list[:max_entity] + [0] * (max(max_entity - len(ids_list), 0)))
            insts_tok.extend(tok_list[:max_entity] + [0] * (max(max_entity - len(tok_list), 0)))
            insts_msk.extend(con_list[:max_entity] + [0] * (max(max_entity - len(con_list), 0)))
            insts_wrd.extend(wrd_list[:max_entity] + [0] * (max(max_entity - len(wrd_list), 0)))
        des_data = [self._ent_des[inst][:20] + [0] * (max(20 - len(self._ent_des[inst]), 0)) for inst in insts_tok]
        wrd_data = [self._ent_wrd[inst][:15] + [0] * (max(15 - len(self._ent_wrd[inst]), 0)) for inst in insts_wrd]
        insts_ids_tensor = Variable(
            torch.LongTensor(insts_ids), volatile=self._test)
        insts_tok_tensor = Variable(
            torch.LongTensor(des_data), volatile=self._test)
        insts_wrd_tensor = Variable(
            torch.LongTensor(wrd_data), volatile=self._test)
        insts_msk_tensor = Variable(
            torch.FloatTensor(insts_msk), requires_grad=False)
        if self._cuda:
            insts_ids_tensor = insts_ids_tensor.cuda()
            insts_tok_tensor = insts_tok_tensor.cuda()
            insts_wrd_tensor = insts_wrd_tensor.cuda()
            insts_msk_tensor = insts_msk_tensor.cuda()
        return (insts_ids_tensor, insts_tok_tensor, insts_wrd_tensor, insts_msk_tensor)



    def generate_pairs(self):
        for qw, dw_pos, dw_neg, qe, de_pos, de_neg in self.read_instances_from_file(self._file_name):
            inst_qw = self.pad_to_longest(qw, 10)
            inst_dw_pos = self.pad_to_longest(dw_pos, 50)
            inst_dw_neg = self.pad_to_longest(dw_neg, 50)
            inst_qe = self.pad_to_longest_entity(qe, 5)
            inst_de_pos = self.pad_to_longest_entity(de_pos, 10)
            inst_de_neg = self.pad_to_longest_entity(de_neg, 10)
            yield inst_qw, inst_qe, inst_dw_pos, inst_de_pos, inst_dw_neg, inst_de_neg



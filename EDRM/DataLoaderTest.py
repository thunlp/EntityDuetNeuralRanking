''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable


class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, ent_des=None, ent_wrd=None, inputs_qw=None, inputs_qe = None,
            inputs_dw=None, inputs_de=None, inputs_i = None, inputs_n = None,
            cuda=True, batch_size=64, test=True):

        assert inputs_dw
        assert inputs_de
        assert inputs_qw
        assert inputs_qe
        assert inputs_i
        assert inputs_n
        assert len(inputs_qw) >= batch_size
        assert len(inputs_qw) == len(inputs_dw) == len(inputs_qe) == len(inputs_de) == len(inputs_i) == len(inputs_n)
        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(inputs_qw) * 1.0 / batch_size))

        self._batch_size = batch_size
        self._ent_des = ent_des
        self._ent_wrd = ent_wrd


        self._inputs_qw = inputs_qw
        self._inputs_qe = inputs_qe
        self._inputs_dw = inputs_dw
        self._inputs_de = inputs_de
        self._inputs_i = inputs_i
        self._inputs_n = inputs_n

        self._iter_count = 0




    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts, max_len):
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
                    torch.LongTensor(inst_data), volatile=self.test)
            if self.cuda:
                mask_tensor = mask_tensor.cuda()
                inst_data_tensor = inst_data_tensor.cuda()
            return (inst_data_tensor, mask_tensor)


        def pad_to_longest_entity(insts, max_entity):
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
            des_data =[self._ent_des[inst][:20] + [0] * (max(20 - len(self._ent_des[inst]), 0)) for inst in insts_tok]
            wrd_data =[self._ent_wrd[inst][:10] + [0] * (max(10 - len(self._ent_wrd[inst]), 0)) for inst in insts_wrd]
            insts_ids_tensor = Variable(
                    torch.LongTensor(insts_ids), volatile=self.test)
            insts_tok_tensor = Variable(
                    torch.LongTensor(des_data), volatile=self.test)
            insts_wrd_tensor = Variable(
                    torch.LongTensor(wrd_data), volatile=self.test)
            insts_msk_tensor = Variable(
                    torch.FloatTensor(insts_msk), requires_grad = False)
            if self.cuda:
                insts_ids_tensor = insts_ids_tensor.cuda()
                insts_tok_tensor = insts_tok_tensor.cuda()
                insts_wrd_tensor = insts_wrd_tensor.cuda()
                insts_msk_tensor = insts_msk_tensor.cuda()
            return (insts_ids_tensor, insts_tok_tensor, insts_wrd_tensor, insts_msk_tensor)



        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            inst_qw = pad_to_longest(self._inputs_qw[start_idx:end_idx], 10)
            inst_dw = pad_to_longest(self._inputs_dw[start_idx:end_idx], 50)
            inst_qe = pad_to_longest_entity(self._inputs_qe[start_idx:end_idx], 5)
            inst_de = pad_to_longest_entity(self._inputs_de[start_idx:end_idx], 10)

            return inst_qw, inst_qe, inst_dw, inst_de, self._inputs_i[start_idx:end_idx], self._inputs_n[start_idx:end_idx]

        else:

            self._iter_count = 0
            raise StopIteration()

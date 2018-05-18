''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable

class Dataloader(object):
    ''' For data iteration '''

    def __init__(self, data, opt, shuffle):
        assert data
        assert len(data) >= opt.batch_size
        self._data = data
        self._data_size = len(data)
        self._cuda = opt.cuda
        self._batch_size = opt.batch_size
        self._iter_count = 0
        self._need_shuffle = shuffle
        self._n_batch = int(np.ceil(self._data_size * 1.0 / self._batch_size))
        if self._need_shuffle:
            np.random.shuffle(self._data)
        self.query, self.answer_pos, self.answer_neg = zip(*self._data)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts, max_len=0):
            ''' Pad the instance to the max seq length in batch '''
            if max_len == 0:
                max_len = max(len(inst) for inst in insts)
            inst_data = np.array([
                inst[:max_len] + [0] * (max_len - len(inst[:max_len]))
                for inst in insts])
            mask = np.zeros((inst_data.shape[0], inst_data.shape[1]))
            for b in range(len(inst_data)):
                for i in range(len(inst_data[b])):
                    if inst_data[b, i] > 0:
                        mask[b, i] = 1
            mask_tensor = Variable(
                torch.FloatTensor(mask), requires_grad=False)
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=False)
            if self._cuda:
                mask_tensor = mask_tensor.cuda()
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor, mask_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            inst_q, mask_q = pad_to_longest(self.query[start_idx:end_idx])
            inst_d_pos, mask_d_pos = pad_to_longest(self.answer_pos[start_idx:end_idx])
            inst_d_neg, mask_d_neg = pad_to_longest(self.answer_neg[start_idx:end_idx])
            return inst_q, inst_d_pos, inst_d_neg, mask_q, mask_d_pos, mask_d_neg

        else:
            if self._need_shuffle:
                np.random.shuffle(self._data)
                self.query, self.answer_pos, self.answer_neg = zip(*self._data)
            self._iter_count = 0
            raise StopIteration()


class DataloaderTest(object):
    ''' For data iteration '''

    def __init__(self, data, opt):
        assert data
        assert len(data) >= opt.batch_size
        self._data = data
        self._data_size = len(data)
        self._cuda = opt.cuda
        self._batch_size = opt.batch_size
        self._iter_count = 0
        self._n_batch = int(np.ceil(self._data_size * 1.0 / self._batch_size))
        self.query, self.answer, self.label, self.id = zip(*self._data)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts, max_len=0):
            ''' Pad the instance to the max seq length in batch '''
            if max_len == 0:
                max_len = max(len(inst) for inst in insts)
            inst_data = np.array([
                inst[:max_len] + [0] * (max_len - len(inst[:max_len]))
                for inst in insts])
            mask = np.zeros((inst_data.shape[0], inst_data.shape[1]))
            for b in range(len(inst_data)):
                for i in range(len(inst_data[b])):
                    if inst_data[b, i] > 0:
                        mask[b, i] = 1
            mask_tensor = Variable(
                torch.FloatTensor(mask), requires_grad=False)
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=True)
            if self._cuda:
                mask_tensor = mask_tensor.cuda()
                inst_data_tensor = inst_data_tensor.cuda()
            return inst_data_tensor, mask_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            inst_q, mask_q = pad_to_longest(self.query[start_idx:end_idx], 10)
            inst_d, mask_d = pad_to_longest(self.answer[start_idx:end_idx], 40)
            return inst_q, inst_d, mask_q, mask_d, self.label[start_idx:end_idx], self.id[start_idx:end_idx]

        else:
            self._iter_count = 0
            raise StopIteration()
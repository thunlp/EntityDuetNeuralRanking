import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import KNRM
import subprocess
import math
import numpy as np
from torch.autograd import Variable
from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest
import sys

def eval_epoch(model, opt, validation_data, crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0.0
    counter = 0.0
    for batch in validation_data.generate_pairs():

        # prepare data
        inputs_qw, inputs_qe, inputs_dw_pos, inputs_de_pos, inputs_dw_neg, inputs_de_neg = batch

        # forward
        outputs_pos = model(inputs_qw, inputs_qe, inputs_dw_pos, inputs_de_pos)
        outputs_neg = model(inputs_qw, inputs_qe, inputs_dw_neg, inputs_de_neg)
        label = torch.ones(outputs_pos.size())
        if opt.cuda:
            label = label.cuda()
        batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))

        total_loss += batch_loss
        counter += 1
    return total_loss.data[0]/counter




def test_epoch(model, test_data, file_name):
    model.eval()
    out = open(file_name, "w")
    test_dict = dict()
    for batch in test_data:
        # prepare data
        qw, qe, dw, de, i, n = batch

        # forward
        outputs = model(qw, qe, dw, de)
        output = outputs.data.tolist()
        tuples = zip(i, n, output)
        for item in tuples:
            if item[0] not in test_dict:
                test_dict[item[0]] = []
            test_dict[item[0]].append((item[1], item[2]))
    for qid, value in test_dict.items():
        res = sorted(value, key=lambda x: x[1], reverse=True)
        for step, item in enumerate(res):
            out.write(str(qid) + " Q0 " + str(item[0]) + " " + str(step + 1) + " " + str(item[1]) + " knrm\n")
    out.close()

def train(model, opt, crit, optimizer, training_data, valid_data, test_data):
    ''' Start training '''
    step = 0
    model.train()
    min_valid_loss = float('inf')
    for epoch_i in range(opt.epoch):
        total_loss = 0.0
        for batch in training_data.generate_pairs():
            # prepare data
            inputs_qw, inputs_qe, inputs_dw_pos, inputs_de_pos, inputs_dw_neg, inputs_de_neg = batch

            # forward
            optimizer.zero_grad()
            outputs_pos = model(inputs_qw, inputs_qe, inputs_dw_pos, inputs_de_pos)
            outputs_neg = model(inputs_qw, inputs_qe, inputs_dw_neg, inputs_de_neg)
            label = torch.ones(outputs_pos.size())
            if opt.cuda:
                label = label.cuda()
            batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))

            # backward
            batch_loss.backward()

            # update parameters
            optimizer.step()
            step += 1
            total_loss += batch_loss.data[0]
            if step % opt.eval_step == 0:
                valid_loss = eval_epoch(model, opt, valid_data, crit)
                print(' Epoch %d step %d Training_loss %f Validation_loss %f' %(epoch_i, step, total_loss/opt.eval_step, valid_loss) )
                file_name = "step_{}.trec".format(step)
                test_epoch(model, test_data, file_name)
                for test_file in ["test_same.qrel", "test_diff.qrel"]:
                    for k in [1, 3, 10]:
                        p = subprocess.Popen(
                            '/data/disk2/private/liuzhenghao/neuralIR/test_data/gdeval.pl -c -k {0} /data/disk2/private/liuzhenghao/neuralIR/test_data/{1} {2}'.format(
                                k, test_file, file_name),
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        for line in p.stdout.readlines():
                            print test_file, k, line
                        retval = p.wait()
                        sys.stdout.flush()
                total_loss = 0
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'settings': opt,
                        'epoch': epoch_i}
                    if opt.save_mode == 'all':
                        model_name = opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                        print('    - [Info] The checkpoint file has been saved.')
                    elif opt.save_mode == 'best':
                        model_name = opt.save_model + '.chkpt'
                        if valid_loss < min_valid_loss:
                            min_valid_loss = valid_loss
                            torch.save(checkpoint, model_name)
                            print('    - [Info] The checkpoint file has been updated.')



def get_ent_embedding(path, vocab_size):
    emb = np.random.uniform(low=-1, high=1, size=(vocab_size, 300))
    nlines = 0
    with open(path) as f:
        for line in f:
            nlines += 1
            if nlines == 1:
                continue
            items = line.split()
            tid = int(items[0])
            if tid > vocab_size:
                print tid
                continue
            vec = np.array([float(t) for t in items[1:]])
            emb[tid, :] = vec
    return emb



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)
    parser.add_argument('-train', required=True)
    parser.add_argument('-valid', required=True)
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-test_data', required=True)
    parser.add_argument('-filter_size', type=int, default=300)
    parser.add_argument('-window_size', type=int, default=5)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=40)
    parser.add_argument('-eval_step', type=int, default=5000)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-mu', type=list, default=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9])
    parser.add_argument('-sigma', type=list, default=[1e-3] + [0.1] * 10)
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    assert len(opt.mu) == len(opt.sigma)


    data = torch.load(opt.data)
    data_test = torch.load(opt.test_data)
    opt.wrd_vocab_size = len(data['wrd2idx'])
    opt.ent_vocab_size = len(data['ent2idx'])
    opt.car_vocab_size = len(data['car2idx'])
    embedding_init = get_ent_embedding("../data/title.emb", opt.wrd_vocab_size)
    opt.n_bins = len(opt.mu)
    print opt

    #========= Preparing DataLoader =========#

    test_data = DataLoaderTest(
        ent_des=data["ent_des"],
        ent_wrd=data["ent_wrd"],
        inputs_qw=data_test['qw'],
        inputs_qe=data_test['qe'],
        inputs_dw=data_test['dw'],
        inputs_de=data_test['de'],
        inputs_i=data_test['i'],
        inputs_n=data_test['n'],
        batch_size=opt.batch_size,
        test=True,
        cuda=opt.cuda)


    training_data = DataLoader(
        file_name = opt.train,
        ent_des = data["ent_des"],
        ent_des_dict = data["ent_des_dict"],
        ent_wrd = data["ent_wrd"],
        ent_wrd_dict = data["ent_wrd_dict"],
        ent2idx = data["ent2idx"],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    validation_data = DataLoader(
        file_name = opt.valid,
        ent_des = data["ent_des"],
        ent_des_dict = data["ent_des_dict"],
        ent_wrd = data["ent_wrd"],
        ent_wrd_dict = data["ent_wrd_dict"],
        ent2idx = data["ent2idx"],
        batch_size=opt.batch_size,
        test=True,
        cuda=opt.cuda)

    model = KNRM.knrm(opt, embedding_init)
    crit = nn.MarginRankingLoss(margin=1, size_average=True)
    if opt.cuda:
        model = model.cuda()
        crit = crit.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=1e-5)
    #checkpoint = torch.load("trained.chkpt")
    #model.load_state_dict(checkpoint['model'])
    train(model, opt, crit, optimizer, training_data, validation_data, test_data)


if __name__ == "__main__":
	main()

#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
import KNRM
import CKNRM
import sys
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from DataLoader import Dataloader, DataloaderTest



def data_evaluate(model, evaluate_data, flag):
    eval_dict = dict()
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    for batch in evaluate_data:
        inputs_q, inputs_d, mask_q, mask_d, label, id = batch
        output = model(inputs_q, inputs_d, mask_q, mask_d)
        output = output.data.tolist()
        tuples = zip(id, label, output)
        for item in tuples:
            if item[0] not in eval_dict:
                eval_dict[item[0]] = []
            eval_dict[item[0]].append((item[1], item[2]))
    for qid, value in eval_dict.items():
        res = sorted(value, key=lambda x: x[1], reverse=True)
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] > 0:
                count += 1
                score += count / (i+1)
        for i in range(len(res)):
            if res[i][0] > 0:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else:
            reduce_num += 1

    MAP = c_1_j / float(len(eval_dict) - reduce_num)
    MRR = c_2_j / float(len(eval_dict) - reduce_num)
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    logging.info(" evaluate on " + flag + " MAP: %f" % MAP)
    logging.info(" evaluate on " + flag + ' MRR: %f' % MRR)
    return MAP, MRR




def train(model, opt, crit, optimizer, train_data, dev_data, test_data):
    ''' Start training '''
    step = 0
    best_map_dev = 0.0
    best_mrr_dev = 0.0
    best_map_test = 0.0
    best_mrr_test = 0.0
    for epoch_i in range(opt.epoch):
        total_loss = 0.0
        for batch in train_data:
            # prepare data
            inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch

            # forward
            optimizer.zero_grad()
            outputs_pos = model(inputs_q, inputs_d_pos, mask_q, mask_d_pos)
            outputs_neg = model(inputs_q, inputs_d_neg, mask_q, mask_d_neg)
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
                print(' Epoch %d Training step %d loss %f' %(epoch_i, step, total_loss))
                map_dev, mrr_dev = data_evaluate(model, dev_data, "dev")
                map_test, mrr_test = data_evaluate(model, test_data, "test")
                total_loss = 0
                if map_dev >= best_map_dev:
                    best_map_dev = map_dev
                    best_map_test = map_test
                    best_mrr_dev = mrr_dev
                    best_mrr_test = mrr_test
                    print ("best dev-- mrr %f map %f; test-- mrr %f map %f" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'settings': opt,
                        'epoch': epoch_i}
                    if opt.save_mode == 'all':
                        model_name = opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = opt.save_model + '.chkpt'
                        if map_dev > best_map_dev:
                            best_map_dev = map_dev
                            best_map_test = map_test
                            best_mrr_dev = mrr_dev
                            best_mrr_test = mrr_test
                            print ("best dev-- mrr %f map %f; test-- mrr %f map %f" %(best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                            torch.save(checkpoint, model_name)
                            print('    - [Info] The checkpoint file has been updated.')


def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in xrange(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-eval_step', type=int, default=10)
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_bins', type=float, default=21)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.mu =  kernal_mus(opt.n_bins)
    opt.sigma = kernel_sigmas(opt.n_bins)
    print opt

    # ========= Preparing DataLoader =========#
    if opt.task == "wikiqa":
        train_filename = "./data/wikiqa/wiki_train_pair.pkl"
        test_filename = "./data/wikiqa/wiki_test.pkl"
        dev_filename = "./data/wikiqa/wiki_dev.pkl"
        train_data = pickle.load(open(train_filename, 'r'))
        test_data = pickle.load(open(test_filename, 'r'))
        dev_data = pickle.load(open(dev_filename, 'r'))
        weights = np.load("./data/wikiqa/embed.txt")

    elif opt.task == "trecqa-clean":
        train_filename = "./data/trecqa/trec_train_pair.pkl"
        test_filename = "./data/trecqa/trec_test_clean.pkl"
        dev_filename = "./data/trecqa/trec_dev_clean.pkl"
        train_data = pickle.load(open(train_filename, 'r'))
        test_data = pickle.load(open(test_filename, 'r'))
        dev_data = pickle.load(open(dev_filename, 'r'))
        weights = np.load("./data/trecqa/embed.txt")
    elif opt.task == "trecqa-all":
        train_filename = "./data/trecqa/trec_train_pair.pkl"
        test_filename = "./data/trecqa/trec_test_all.pkl"
        dev_filename = "./data/trecqa/trec_dev_all.pkl"
        train_data = pickle.load(open(train_filename, 'r'))
        test_data = pickle.load(open(test_filename, 'r'))
        dev_data = pickle.load(open(dev_filename, 'r'))
        weights = np.load("./data/trecqa/embed.txt")
    else:
        raise ("Not implement!")
    train_data = Dataloader(data = train_data, opt = opt, shuffle=True)
    test_data = DataloaderTest(data = test_data, opt = opt)
    dev_data = DataloaderTest(data = dev_data, opt = opt)
    if opt.model == "knrm":
        model = KNRM.knrm(opt, weights)
    elif opt.model == "cknrm":
        model = CKNRM.knrm(opt, weights)
    else:
        raise ("No such model!")
    crit = nn.MarginRankingLoss(margin=1, size_average=True)

    if opt.cuda:
        model = model.cuda()
        crit = crit.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    train(model, opt, crit, optimizer, train_data, dev_data, test_data)


if __name__ == "__main__":
    main()

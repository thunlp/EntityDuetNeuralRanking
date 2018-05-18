''' Handling the data io '''
import argparse
import torch
import sys



def read_vocab_idx(vocab_path):
    ''' build vocab '''

    word2idx = {"_PAD" : 0}

    with open(vocab_path) as f:
        for line in f:
            tokens = line.decode("utf8").strip("\n").split("\t")
            no = int(tokens[1])
            word2idx[tokens[0]] = no

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)))
    return word2idx

def read_ent_des(inst_file):
    ent_des_dict = dict()
    ent_des = list()
    ent_des.append([0] * 20)
    with open(inst_file) as f:
        for step, line in enumerate(f):
            tokens = line.decode("utf8").strip().split()
            ent_des_dict[tokens[0]] = step + 1
            ent_des.append([int(token) for token in tokens[1:]][:20])
    return ent_des, ent_des_dict

def read_ent_car(inst_file):
    ent_wrd_dict = dict()
    ent_wrd = list()
    ent_wrd.append([0] * 10)
    with open(inst_file) as f:
        for step, line in enumerate(f):
            tokens = line.decode("utf8").strip().split()
            ent_wrd_dict[tokens[0]] = step + 1
            ent_wrd.append([int(token) for token in tokens[1:]][:10])
    return ent_wrd, ent_wrd_dict


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-ent_des', required=True)
    parser.add_argument('-ent_car', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-wrd_vocab', required=True)
    parser.add_argument('-ent_vocab', required=True)
    parser.add_argument('-car_vocab', required=True)

    opt = parser.parse_args()
    wrd2idx = read_vocab_idx(opt.wrd_vocab)
    ent2idx = read_vocab_idx(opt.ent_vocab)
    car2idx = read_vocab_idx(opt.car_vocab)
    ent_des, ent_des_dict = read_ent_des(opt.ent_des)
    ent_wrd, ent_wrd_dict = read_ent_car(opt.ent_car)
    data = {
        'settings': opt,
        'wrd2idx': wrd2idx,
        'ent2idx': ent2idx,
        'car2idx': car2idx,
        'ent_des_dict' : ent_des_dict,
        'ent_des' : ent_des,
        'ent_wrd_dict': ent_wrd_dict,
        'ent_wrd': ent_wrd}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
''' Handling the data io '''
import argparse
import torch
import sys
from collections import defaultdict


def cover_text2int(sentence):
    tokens = sentence.strip().split(",")
    tokens = [int(token) for token in tokens if int(token) >= 0]
    if len(tokens) == 0:
        return [0]
    else:
        return tokens



def cover_text2inte(sentence, ent2idx, ent_des_dict, ent_wrd_dict,  max_entity):
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


def read_instances_from_file(inst_file, ids_file, ent2idx, ent_des_dict, ent_wrd_dict):
    ''' Convert file into word seq lists '''
    qw = []
    qe = []
    dw = []
    de = []
    n = []
    i = []
    error_sent_count = 0
    with open(inst_file) as f1, open(ids_file) as f2:
        lines = zip(f1, f2)
        for line in lines:
            tokens = line[0].strip().split("\t")
            if len(tokens) != 4:
                error_sent_count += 1
                continue
            qw.append(cover_text2int(tokens[0]))
            qe.append(cover_text2inte(tokens[2], ent2idx, ent_des_dict, ent_wrd_dict, 5))
            dw.append(cover_text2int(tokens[1]))
            de.append(cover_text2inte(tokens[3], ent2idx, ent_des_dict, ent_wrd_dict, 10))
            tokens = line[1].strip().split("\t")
            n.append(tokens[2])
            i.append(tokens[0])

    print('[Info] Get {} instances from {}'.format(len(qw), inst_file))

    if error_sent_count > 0:
        print('[Warning] error instances {}.'.format(error_sent_count))
    return qw, qe, dw, de, n, i




def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-inst_file', required=True)
    parser.add_argument('-ids_file', required=True)

    opt = parser.parse_args()
    data = torch.load(opt.data)
    # Training set
    ent2idx = data["ent2idx"]
    ent_des_dict = data["ent_des_dict"]
    ent_wrd_dict = data["ent_wrd_dict"]
    qw, qe, dw, de, n, i = read_instances_from_file(opt.inst_file, opt.ids_file, ent2idx, ent_des_dict, ent_wrd_dict)

    data = {
        'settings': opt,
        'qw': qw,
        'qe': qe,
        'dw': dw,
        'de': de,
        'i': i,
        'n': n}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    main()

#-*-coding:utf-8 -*-
import sys
import numpy as np
import pickle
from collections import Counter

UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'



class Util(object):

    @staticmethod
    def get_initial_vocab(glovec_file):
        initial_vocab = dict()
        for line in file(glovec_file):
            units = line.strip().split(" ")
            word = units[0].lower()
            initial_vocab[word] = 1
        return initial_vocab


    @staticmethod
    def generate_vocab(file_list, output_file, initial_vocab={}, min_count=5):
        vf = open(output_file,'w')
        vocab = Counter()
        filenames = file_list
        for filename in filenames:
            for line in file(filename):
                sents = line.lower().strip().split("\t")
                for sent in sents[:2]:
                    vocab.update(sent.strip().split())
        vocab_list = vocab.most_common()
        vf.write(PAD_TOKEN + "\n")
        vf.write(UNKNOWN_TOKEN + "\n")
        for word, count in vocab_list:
            if word in initial_vocab or count >= min_count:
                vf.write(word+"\n")

    @staticmethod
    def generate_embed(vocab_file, glovec_file, output_file):
        vocab_dict = dict()
        for id, line in enumerate(file(vocab_file)):
            word = line.strip()
            vocab_dict[word] = id
        embeding_list = [[] for i in range(len(vocab_dict))]


        for line in file(glovec_file):
            units = line.strip().split(" ")
            word = units[0].lower()
            if word in vocab_dict:
                vector = map(float,units[1:])
                index = vocab_dict[word]
                embeding_list[index] = vector

        for i in range(len(vocab_dict)):
            if len(embeding_list[i]) == 0:
                temp_vec = (np.random.randn(300) * 0.2).tolist()
                embeding_list[i] = temp_vec
        embedding_vec = np.array(embeding_list)
        print embedding_vec.shape
        # print count
        embedding_vec.dump(output_file)

    @staticmethod
    def generate_data(input_file,vocab_file,output_file):
        vocab_dict = dict()
        for id, line in enumerate(file(vocab_file)):
            word = line.strip()
            vocab_dict[word] = id
        ff = open(output_file,"wb")
        data = []
        count = set()
        for line in file(input_file):
            units = line.lower().strip().split("\t")
            count.add(units[0])
            question = [vocab_dict.get(word, vocab_dict[UNKNOWN_TOKEN]) for word in units[0].strip().split(" ")]
            answer = [vocab_dict.get(word, vocab_dict[UNKNOWN_TOKEN]) for word in units[1].strip().split(" ")]
            label = int(units[2])
            id = units[3]
            data.append((question,answer,label,id))
        print len(count)
        pickle.dump(data,ff)

    @staticmethod
    def generate_pairwise_data(input_file,output_file):
        pair_list = list()
        train_data = pickle.load(open(input_file, 'r'))
        right = dict()
        wrong = dict()
        for units in train_data:
            if units[2] == 1:
                if units[3] not in right:
                    right[units[3]] = []
                right[units[3]].append((units[0], units[1]))
            else:
                if units[3] not in wrong:
                    wrong[units[3]] = []
                wrong[units[3]].append((units[0], units[1]))
        for id, values in right.items():
            for value in values:
                if id in wrong:
                    for item in wrong[id]:
                        pair_list.append((value[0], value[1], item[1]))
        ff = open(output_file,"wb")
        print len(pair_list)
        print len(right)
        pickle.dump(pair_list,ff)



def preprocess_trecqa(input_dir,output_dir):
    def generate_combine_data(sub_dir):
        b_file = open(input_dir+sub_dir+"/b.toks",'r')
        s_file = open(input_dir+sub_dir+"/sim.txt",'r')
        i_file = open(input_dir+sub_dir+"/id.txt",'r')
        new_file = open(output_dir+sub_dir+".txt",'w')
        for a_line in file(input_dir+sub_dir+"/a.toks"):
            a_line = a_line.strip()
            b_line = b_file.readline().strip()
            s_line = s_file.readline().strip()
            i_line = i_file.readline().strip()
            new_file.write(a_line+"\t"+b_line+"\t"+s_line+"\t"+i_line+"\n")
    generate_combine_data("clean-dev")
    generate_combine_data("clean-test")
    generate_combine_data("train-all")
    generate_combine_data("raw-dev")
    generate_combine_data("raw-test")

def preprocess_wikiqa(input_dir,output_dir):
    def generate_combine_data(sub_dir):
        s_file = open(input_dir+"WikiQA-"+sub_dir+".txt",'r')
        i_file = open(input_dir+"WikiQA-"+sub_dir+".ref",'r')
        new_file = open(output_dir+sub_dir+".txt",'w')
        for s_line in s_file:
            s_line = s_line.strip()
            i_line = i_file.readline().strip().split()[0].strip()
            new_file.write(s_line+"\t"+i_line+"\n")
    generate_combine_data("dev")
    generate_combine_data("train")
    generate_combine_data("test")


if __name__ == "__main__":
    task = sys.argv[1]
    inital_vocab = Util.get_initial_vocab("./data/glove/glove.840B.300d.txt")
    if task == "wikiqa":
        preprocess_wikiqa(input_dir="./data/raw_data/WikiQA/WikiQACorpus/", output_dir="./data/wikiqa/")
        print "generate vocab"
        Util.generate_vocab(file_list=["./data/wikiqa/train.txt","./data/wikiqa/dev.txt","./data/wikiqa/test.txt"],output_file="./data/wikiqa/vocab.txt",initial_vocab=inital_vocab)
        print "generate emb"
        Util.generate_embed(vocab_file="./data/wikiqa/vocab.txt",glovec_file="./data/glove/glove.840B.300d.txt",output_file="./data/wikiqa/embed.txt")
        print "generate data pkl"
        Util.generate_data("./data/wikiqa/train.txt","./data/wikiqa/vocab.txt","./data/wikiqa/wiki_train.pkl")
        Util.generate_data("./data/wikiqa/dev.txt","./data/wikiqa/vocab.txt","./data/wikiqa/wiki_dev.pkl")
        Util.generate_data("./data/wikiqa/test.txt","./data/wikiqa/vocab.txt","./data/wikiqa/wiki_test.pkl")
        Util.generate_pairwise_data("./data/wikiqa/wiki_train.pkl", "./data/wikiqa/wiki_train_pair.pkl")
    elif task == "trecqa":
        preprocess_trecqa(input_dir="./data/raw_data/TrecQA/",output_dir="./data/trecqa/")
        print "generate vocab"
        Util.generate_vocab(file_list=["./data/trecqa/train-all.txt","./data/trecqa/clean-dev.txt","./data/trecqa/clean-test.txt"],output_file="./data/trecqa/vocab.txt",initial_vocab=inital_vocab)
        print "generate emb"
        Util.generate_embed(vocab_file="./data/trecqa/vocab.txt",glovec_file="./data/glove/glove.840B.300d.txt",output_file="./data/trecqa/embed.txt")
        print "generate data pkl"
        Util.generate_data("./data/trecqa/clean-dev.txt","./data/trecqa/vocab.txt","./data/trecqa/trec_dev_clean.pkl")
        Util.generate_data("./data/trecqa/clean-test.txt","./data/trecqa/vocab.txt","./data/trecqa/trec_test_clean.pkl")
        Util.generate_data("./data/trecqa/train-all.txt","./data/trecqa/vocab.txt","./data/trecqa/trec_train.pkl")
        Util.generate_data("./data/trecqa/raw-dev.txt","./data/trecqa/vocab.txt","./data/trecqa/trec_dev_all.pkl")
        Util.generate_data("./data/trecqa/raw-test.txt","./data/trecqa/vocab.txt","./data/trecqa/trec_test_all.pkl")
        Util.generate_pairwise_data("./data/trecqa/trec_train.pkl", "./data/trecqa/trec_train_pair.pkl")
    else:
        sys.stderr.write("illegal param")

import sys
reload(sys)
sys.setdefaultencoding('UTF8')

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file", help="your ranking file to be evaluated", type=argparse.FileType('r'))
    parser.add_argument("top1_file", help="a list of qid top1_clicled_docid", type=argparse.FileType('r'))
    args = parser.parse_args()

    # read trec
    trec = {}
    all_doc = set()
    for line in args.trec_file:
        #  1 Q0 sogou-15630 1 0.601785 # ranklib
        items = line.split()
        qid = int(items[0])
        docid = items[2].strip()
        rank = int(items[3])
        all_doc.add(docid)
        #if rank > 10: # only eval top10
        #    continue
        if qid not in trec:
            trec[qid] = {}
        trec[qid][docid] = rank

    # read reversed pairs
    query_total = [0] * 1001
    query_correct = [0] * 1001
    query_mrr = [0] * 1001

    for line in args.top1_file:
        qid, pos = line.strip().split('\t')
        qid = int(qid)
        if qid not in trec:
            continue
        if pos in trec[qid]:
            rank1 = trec[qid][pos]
            query_total[qid] += 1.0
            if rank1 == 1: # correct
                query_correct[qid] += 1.0
            query_mrr[qid] += 1/float(rank1)

    print "runid,topic,p@1,mrr@1"
    mean = 0
    mean2 = 0
    nq = 0
    for qid in range(1, 1001):
        if query_total[qid] == 0:
            continue
        res = query_correct[qid]/query_total[qid]
        res2 = query_mrr[qid]/query_total[qid]
        mean += res
        mean2 += res2
        nq += 1
        print '#, {0}, {1}, {2}'.format(qid, res, res2) 
    
    print '#, mean, {0}, {1}'.format(mean/nq, mean2/nq)


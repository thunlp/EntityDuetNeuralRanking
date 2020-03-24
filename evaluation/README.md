# Evaluation for Entity-Duet Neural Ranking Model

Put your evaluation file ``test.trec`` and run.

## Testing RAW
``
python2 eval_top1.py  test.trec test_raw.qrel
``


## Testing DIFF and Tesing SAME
* Set K to 1,3,5 and get NDCG@K

``
./gdeval.pl -c -k 1 ./test_diff.qrel test.trec
``

``
./gdeval.pl -c -k 1 ./test_same.qrel test.trec
``

* Return and the 0.3407 is NDGC@K

``
knrm,amean,0.30407,0.08468
``

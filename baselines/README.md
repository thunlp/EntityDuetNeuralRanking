## Baselines:
- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)
- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)



## Prerequisites
- Python 2.x
- Pytorch 0.3

## Data
Sogou query log is not published now. Therefore, we evaluate K-NRM and Conv-KNRM on two published data sets for question answering.
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [TrecQA: Answer Selction Task](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Data Process

### Pretrained Word Embeddings

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove
```
```
unzip -o -d data/glove/ data/glove/glove.840B.300d.zip
```
### WikiQA Preprocess
**Note**: dowload \"WikiQACorpus.zip\" to the path "./data/raw_data/WikiQA/".
>WikiQACorpus.zip download link: https://www.microsoft.com/en-us/download/details.aspx?id=52419
```
python preprocess.py wikiqa
```
### TrecQA Preprocess
**Note**: You can copy the directory [TrecQA_of_CIKM2016_Rao](https://github.com/castorini/NCE-CNN-Torch/tree/master/data/TrecQA) to our path "./data/raw_data/"
```
python preprocess.py trecqa
```

Because I have uploaded my preprocess data of WikiQA, if you want to cast a glance at our model, you can skip above operations.

## Running

```
usage: main.py [-task TASK] [-model MODEL] [-eval_step EVAL_STEP] [-epoch EPOCH] [-d_word_vec EMBDEDDING_DIMENTION] [-batch_size BATCH_SIZE] [-save_model CHACKPOINT_NAME] [-save_mode SAVE_ALL_OR_BEST] [-no_cuda TRAIN_WITHOUT_GPU] [-lr LEARNING_RATE] [-n_bins KERNEL]
```

### WikiQA
Basic approach: **listwise**
```
python main.py -task wikiqa -model knrm
```
or
```
python main.py -task wikiqa -model cknrm
```


### TrecQA
Basic approach: **listwise**
```
python main.py -task trecqa -model knrm
```
or
```
python main.py -task trecqa -model cknrm
```

## Results
In all experiments, we selected training models that obtain the best MAP and MRR scores on the development set for testing.

Kernel size is 21, which is not same as EDRM.

Development Set:

|           | WikiQA   |  TrecQA (all)  |   TrecQA (clean) |  
| --------  | -------- | --------  | --------  |
| K-NRM     | mrr 0.697134; map 0.690313 |  mrr 0.887622; map 0.797774 | mrr 0.814342; map 0.737469	 |
| Conv-KNRM | mrr 0.743666; map 0.727295 |  mrr 0.811153; map 0.759652 | mrr 0.755000; map 0.699610 |



Testing Set:

|           | WikiQA   |  TrecQA (all)  |   TrecQA (clean) |  
| --------  | -------- | --------  | --------  |
| K-NRM     | mrr 0.662426; map 0.645415 |  mrr 0.831841; map 0.753466 | mrr 0.779068; map 0.679276 |
| Conv-KNRM | mrr 0.663512; map 0.650243 |  mrr 0.789182; map 0.726721 | mrr 0.799825; map 0.709013 |




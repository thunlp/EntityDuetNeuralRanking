# Entity-Duet Neural Ranking Model
There are source codes for Entity-Duet Neural Ranking Model (EDRM) [Paper](http://aclweb.org/anthology/P18-1223).


![model](https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/model.png)

## Baselines

There are codes for our main baselines: K-NRM and Conv-KNRM.

- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)
- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)


## EDRM

There are codes for our work based on Conv-KNRM.


## Results

The ranking results. All results are in trec format.



| Method | Testing\-SAME \(NDCG@1\) | Testing\-SAME \(NDCG@10\)| Testing\-DIFF \(NDCG@1\)| Testing\-DIFF \(NDCG@10\)|  Testing\-RAW \(MRR\)|
| --------  | -------- | --------  | --------  |  --------  | --------  |
|K\-NRM|0\.2645|0\.4197|0\.3000|0\.4228|0\.3447|
|Conv\-KNRM|0\.3357|0\.4810|0\.3384|0\.4318|0\.3582|
|EDRM\-KNRM|0\.3096|0\.4547|0\.3327|0\.4341|0\.3616|
|EDRM\-CKNRM	|0\.3397|0\.4821|0\.3708|0\.4513|0\.3892|



Results on ClueWeb09 and CluWeb12. All models are trained on Anchor-Doc pairs in ClueWeb. These results only leverage entity embedding and entity description.
For EDRM of English version, please refer to our [OpenMatch tookit](https://github.com/thunlp/OpenMatch).


ClueWeb09:

| Method | NDCG@20 | ERR@20 |
| ------ | ------- | ------ |
|Conv\-KNRM|0\.2893|0\.1521|
|EDRM|0\.2922|0\.1642|

ClueWeb12:

| Method | NDCG@20 | ERR@20 |
| ------ | ------- | ------ |
|Conv\-KNRM|0\.1142|0\.0930|
|EDRM|0\.1183|0\.0968|

## Citation
```
@inproceedings{liu2018EntityDuetNR,
  title={Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval},
  author={Zhenghao Liu and Chenyan Xiong and Maosong Sun and Zhiyuan Liu},
  booktitle={Proceedings of ACL},
  year={2018}
}
```

## Contact
If you have questions, suggestions and bug reports, please email 
```
liuzhenghao0819@gmail.com.
```

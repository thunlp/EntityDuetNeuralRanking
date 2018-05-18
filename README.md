# Entity-Duet Neural Ranking Model
There are source codes for Entity-Duet Neural Ranking Model (EDRM).


![model](https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/model.png)

## Baselines

There are code for our main baseline K-NRM and Conv-KNRM:

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


## Citation
```
Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval. Zhenghao Liu, Chenyan Xiong, Maosong Sun and Zhiyuan Liu.
```

## Copyright

All Rights Reserved.


## Contact
If you have questions, suggestions and bug reports, please email liuzhenghao0819@gmail.com.

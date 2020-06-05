# Data for Entity-Duet Neural Ranking Model

## Download Data
All data you can find on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1f57be663018465ab0ad/). You should download them.

## Generate Training Files
* run ``process.sh`` to generate files for training

## Files Description
* train_ent_des_expansion; valid_ent_des_expansion; test_ent_des_expansion
    * query ids \t document ids \t qurey entities \t document entities
* type2id
    * mapping of entity type and entity type id
* word2id
    * mapping of word and word id
* ent2id
    * mapping of entity and entity id
* ent2des
    * mapping of entity and word ids of description sentence
* ent2type
    * mapping of entity and entity type
* test.trec
    * original testing files
* test_diff.qrel; test_same.qrel
    * Evaluation files
* wordemb.txt
	* The pretrained word embedding provided by [K-NRM](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)
* tag_result
	* The text of query and document after entity linking
	* Format: query \t query entities \t docuent \t document entities
* data.pt; test.pt
    * Generated files with our script

## Knowledge Graph

We use [CN-DBpedia](http://kw.fudan.edu.cn/cndbpedia/intro/) in this experiments. You can download this resource from [here](http://openkg.cn/dataset/cndbpedia).

We provide an example CMNS entity linking code for you. It is apparent that lots of problems occur because of the link ambiguious and mention errors. Nevertheless, no effective and effecient entity linker is established.


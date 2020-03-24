# Data for Entity-Duet Neural Ranking Model

## Training and Testing Data
All data you can find on [Tsinghua Cloud]
(https://cloud.tsinghua.edu.cn/d/1f57be663018465ab0ad/).

* train.pt is the training data for EDRM. There are several fileds in this file. You can follow below instructions
	* settings: experiments settings
	* wrd2idx: word dictionary (convert word to index)
	* ent2idx: entity dictionary (convert entity to index)
	* car2idx: entity type dictionary (convert entity type to index)
	* ent_des_dict : mapping entity to description list index
	* ent_des : entity description list
	* ent_wrd_dict: mapping entity to entity type list index
	* ent_wrd: entity types list
* test.pt is the testing data for EDRM. There are several fileds in this file. You can follow below instructions
	* settings: experiments settings
	* qe: query entity information
	* de: document entity infomation
	* qw: query word information
	* dw: document word infomation
    * i : query id
    * n : query name (e.g. Sogou-XXX)
    * Note that, for qe and qe, each contains four kinds of data (query/ document word list, entity description list index, entity type list index, entity occurance time)
* title.emb
	* The pretrained word embedding provided by [K-NRM](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)
* tag_result
	* The text of query and document after entity linking
	* Format: query \t query entities \t docuent \t document entities


## Knowledge Graph

We use [CN-DBpedia](http://kw.fudan.edu.cn/cndbpedia/intro/) in this experiments. You can download this resource from [here](http://openkg.cn/dataset/cndbpedia).

We provide an example CMNS entity linking code for you. It is apperantly that lots of problems occur because of the link ambiguious and mention errors. Nevertheless, no effective and effecient entity linker is established. 


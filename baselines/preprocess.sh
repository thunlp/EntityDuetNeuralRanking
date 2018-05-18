task=$1
GloVe="./data/glove/glove.840B.300d.txt"
WikiQA="./data/raw_data/WikiQA/WikiQACorpus.zip"
TrecQA="./data/raw_data/TrecQA"

if [ ! -f "$GloVe" ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove
	unzip -o -d data/glove/ data/glove/glove.840B.300d.zip
fi;

if [ "$task" = "wikiqa" ]; then
	if [ -f "$WikiQA" ]; then
		unzip -o -d "data/raw_data/WikiQA/" "data/raw_data/WikiQA/WikiQACorpus.zip"
	else
		echo "!!!!!!!!Please dowload the file \"WikiQACorpus.zip\" to the path ./data/raw_data/WikiQA/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419"
	fi;
	python preprocess.py wikiqa
elif [ "$task" = "trecqa" ]; then
	if [ ! -d "$TrecQA" ]; then
		svn checkout https://github.com/castorini/NCE-CNN-Torch/trunk/data/TrecQA  ./data/raw_data/TrecQA
	fi;
	python preprocess.py trecqa
fi

python preprocess.py \
-car_vocab type2id \
-ent_car ent2type \
-ent_des ent2des \
-ent_vocab ent2id \
-save_data data.pt \
-wrd_vocab word2id

python test_preprocess.py \
-data data.pt \
-save_data test.pt \
-inst_file ./test_ent_des_expansion \
-ids_file ./test.trec
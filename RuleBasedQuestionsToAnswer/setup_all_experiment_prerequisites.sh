
# Section 2. STs, response classifiers and baselines experiments setup
# cd RuleBasedQuestionsToAnswer/
# > QADialogueSystem/RuleBasedQuestionsToAnswer


# To setup the Syntactic Transformations STs
# 1. Get lexAccess2016lite
wget https://lexsrv3.nlm.nih.gov/LexSysGroup/Projects/lexAccess/2013+/release/lexAccess2016lite.tgz
tar xvzf lexAccess2016lite.tgz
# 2. Get simplenlg-v4.4.8 for verb transformation
wget https://github.com/simplenlg/simplenlg/releases/download/v4.4.8/simplenlg-v4.4.8.zip
unzip simplenlg-v4.4.8.zip
# 3. Get stanford-corenlp-full-2018-02-27 for lexparser
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip

# For Language Model Baseline experiment
# 1. Create LMs directory 
mkdir LMs
cd LMs
# > QADialogueSystem/RuleBasedQuestionsToAnswer/LMs
# 2. download required LMs
wget https://www.keithv.com/software/giga/lm_giga_64k_nvp_2gram.zip
unzip lm_giga_64k_nvp_2gram.zip
wget https://www.keithv.com/software/giga/lm_giga_64k_nvp_3gram.zip
unzip lm_giga_64k_nvp_3gram.zip
wget https://www.keithv.com/software/giga/lm_giga_64k_vp_2gram.zip
unzip lm_giga_64k_vp_2gram.zip
wget https://www.keithv.com/software/giga/lm_giga_64k_vp_3gram.zip
unzip lm_giga_64k_vp_3gram.zip
cd ..
# > QADialogueSystem/RuleBasedQuestionsToAnswer
# 3. For Decomposable Attention + Elmo model
# Download and save elmo embedding
mkdir data/
cd data/
# > QADialogueSystem/RuleBasedQuestionsToAnswer/data
mkdir elmo
cd elmo/
# > QADialogueSystem/RuleBasedQuestionsToAnswer/data/elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5
cd ../../
# > QADialogueSystem/RuleBasedQuestionsToAnswer
# 4. BERT (Logistic)
# We need pytorch-transformers
git clone https://github.com/huggingface/transformers
cd transformers
# > QADialogueSystem/RuleBasedQuestionsToAnswer/transformers
git reset --hard a95ced6260494f108df72f0f9ffe9c60498365ad
cd examples
# > QADialogueSystem/RuleBasedQuestionsToAnswer/transformers/examples
# Copy files from pytorch-transformers/examples to transformers/examples
cp ../../pytorch-transformers/examples/* .
cd ../../
# > QADialogueSystem/RuleBasedQuestionsToAnswer

# cd ..

# Section 3 data-augmentation and generation experiments setup

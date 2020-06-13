# RuleBasedQuestionsToAnswer
This folder contains the code for the Syntactic Transformations (STs), response classifiers and baselines (Section 2).
STs generate a large list of responses given the question and the candidate answer.

# Setup for RuleBasedQuestionsToAnswer
Run `sh setup_all_experiment_prerequisites.sh`

## Response Classifiers and Baselines
Download and extract [preprocessed train test and val data files](https://mega.nz/#!wQR0jI7D!_mZo7vSQvwNvCjd9RMMbhuU77EPSRshwhCYLFWpOaKI) in `"train_data"` folder within the `"RuleBasedQeustionToAnswer"` parent directory.
1. Shortest Response and Language model baseline: `python shortest_response_and_langauge_model_baseline.py`
2. Feature based linear model (Logistic): `python logistic_regression_baseline.py`
3. Feature based linear model (Softmax): `python softmax_baseline.py`
4. Decomposable Attention (Logistic):
   To train the model: `python decomposable_attention_model_training.py`
   To evaluate the model: `python test_decomposable_attention_model.py`
5. Decomposable Attention (Softmax):
   To train the model: `python decomposable_attention_model_softmax_training.py`
   To evaluate the model: `python test_decomposable_attention_softmax_model.py`
6. Decomposable Attention + ELMo (Logistic):
   To train the model: `python decomposable_attention_model_training.py -e`
   To evaluate the model: `python test_decomposable_attention_model.py -e`
7. Decomposable Attention + ELMo (Softmax):
   To train the model: `python decomposable_attention_model_softmax_training.py -e`
   To evaluate the model: `python test_decomposable_attention_softmax_model.py -e`
8. BERT (Logistic):
   Recommended to create a separate environment to run this as its code is dependent on pytorch-transformers (older version of hugginface transformers)
   requirements: `python -m pip install torch==1.1.0 tensorboardX==1.9`
   ```
   cd transformers
   python setup.py install
   ```
   TODO: Write the correct training commands
   To evaluate run `python ensemble_evaluator.py`
9. BERT (Softmax):
   To train: `python bert_softmax_classification_model.py --model_type bert --model_name_or_path bert-base-uncased --data_dir train_data/bert_softmax_classifier/data_cache --output_dir train_data/bert_softmax_classifier/ckpt --do_train --per_gpu_train_batch_size 50 --save_steps 8000 --logging_steps 2 --learning_rate 5e-6`
   To test: `python bert_softmax_classification_model.py --model_type bert --model_name_or_path train_data/bert_softmax_classifier/ckpt --output_dir train_data/bert_softmax_classifier/ckpt --data_dir train_data/bert_softmax_classifier/data_cache --do_eval --per_gpu_eval_batch_size 50`


### Feature extractor:
- `kenlm` - https://github.com/kpu/kenlm

# TODOs
- Convert the java code from hardcoded files to commandline arguments

## Problems with current OpenNMT decoding
- translate will only work with -batch_size 1
- training with sgd with standard parameters gave slightly lower training performance but better validation performance than adadelta
- After lot of experiments, sgd with lr 1.0 for 5 epochs and then 0.5 decay every next epoch seems to work the best

## What does each file do:
- `create_Opennmt_train_val_test_src_and_tgt_file_from_top_k_responses.py` - Takes the predictions generated from `get_top_k_responses_from_decomposable_attention_softmax_model.py` and stores them into relevant folders in `RuleBasedQuestionsToAnswer`. These files will be later used to train OpenNMT models
- `collect_passages_for_qas.py` - Takes the processed test files from `RuleBasedQuestionsToAnswer/squad_seq2seq_train_moses_tokenized` and adds the passage information for them
- `create_test_from_squad2_dev.py` - This will read the squad2 dev set and create a test set for the squad, coqa and Answer generation Seq2Seq models
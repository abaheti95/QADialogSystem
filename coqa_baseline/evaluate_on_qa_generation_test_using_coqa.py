import os
import json
from sogou_mrc.dataset.coqa import CoQAReader,CoQAEvaluator
from sogou_mrc.libraries.BertWrapper import BertDataHelper
from sogou_mrc.model.bert_coqa import BertCoQA
from sogou_mrc.data.vocabulary import  Vocabulary
from sogou_mrc.data.batch_generator import BatchGenerator
import tensorflow as tf
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

coqa_reader = CoQAReader(-1)
data_folder= os.path.join("/", "home", "baheti", "QADialogueSystem", "Data", "QA_datasets", "coqa/")
train_filename = "coqa-train-v1.0.json"
eval_filename = "coqa-dev-v1.0.json"
vocab = Vocabulary(do_lowercase=True)
vocab_filepath = os.path.join("models", "vocab.txt")
if os.path.exists(vocab_filepath):
	print("loading from filepath")
	# load from the filepath
	vocab.load(vocab_filepath)
else:
	print("creating vocab as new")
	train_data = coqa_reader.read(data_folder+train_filename, 'train')
	eval_data = coqa_reader.read(data_folder+eval_filename,'dev')
	vocab.build_vocab(train_data+eval_data)
	vocab.save(vocab_filepath)

# Squad seq2seq_train_moses_tokenized
# DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_train_moses_tokenized")
# coqa_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_responses_test_coqa_format.json")
# src_squad_seq2seq_predicted_responses_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_responses_test.txt")
# predictions_save_file = "coqa_predictions_on_squad_seq2seq_predicted_responses_test.txt"

# SQUAD seq2seq dev moses tokenized
DATA_DIR = os.path.join("..", "RuleBasedQuestionsToAnswer", "squad_seq2seq_dev_moses_tokenized")
coqa_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_dev_moses_test_coqa_format.json")
src_squad_seq2seq_predicted_responses_file = os.path.join(DATA_DIR, "src_squad_seq2seq_dev_moses_test.txt")
predictions_save_file = "coqa_predictions_on_squad_seq2seq_dev_moses_test.txt"

test_data = coqa_reader.read(coqa_format_test_save_file, 'test')
evaluator = CoQAEvaluator(coqa_format_test_save_file)

best_model_path = os.path.join('models', 'best_weights')
bert_dir = 'uncased_L-12_H-768_A-12'
bert_data_helper = BertDataHelper(bert_dir)
test_data = bert_data_helper.convert(test_data,data='coqa')

model = BertCoQA(bert_dir=bert_dir,answer_verification=True)
print("loading model")
model.load(best_model_path)
print("model loaded")

my_batch_size = 6
test_batch_generator = BatchGenerator(vocab, test_data,training=False,batch_size=my_batch_size,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'])

# To fix FailedPreconditionError: https://github.com/sogou/SMRCToolkit/issues/30
model.session.run(tf.local_variables_initializer())
pred_answer = model.evaluate(test_batch_generator, evaluator)

print(pred_answer)
# Generate readable output from pred_answer
count = 0			# Trying to figure out how many CoQA answers are exact matches
with open(src_squad_seq2seq_predicted_responses_file, "r") as reader, open(predictions_save_file, "w") as writer:
	for i, line in enumerate(reader):
		gold_q, gold_a = line.strip().split(" ||| ")
		gold_q = gold_q.strip()
		gold_a = gold_a.strip()
		pred_a = pred_answer[(str(i), 1)]

		writer.write("Q {}\t\t:{}\n".format((i+1), gold_q))
		writer.write("Gold\t\t:{}\n".format(gold_a))
		writer.write("Coqa Pred\t:{}\n\n".format(pred_a))
		if gold_a == pred_a.lower():
			count += 1

print(i, count)

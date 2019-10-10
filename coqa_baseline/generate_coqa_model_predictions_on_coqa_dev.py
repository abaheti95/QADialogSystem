# We will generate coqa predictions on the CoQA Dev set so that we can later run our seq2seq model on it

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

from sacremoses import MosesTokenizer
mt = MosesTokenizer()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

coqa_reader = CoQAReader(-1)
data_folder= os.path.join("/", "home", "baheti", "QADialogueSystem", "Data", "QA_datasets", "coqa/")
train_filename = "coqa-train-v1.0.json"
eval_filename = "coqa-dev-v1.0.json"
eval_filepath = os.path.join(data_folder, "coqa-dev-v1.0.json")
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

DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_train_moses_tokenized")
coqa_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_responses_test_coqa_format.json")
src_squad_seq2seq_predicted_responses_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_responses_test.txt")

val_data = coqa_reader.read(data_folder+eval_filename,'dev')
evaluator = CoQAEvaluator(data_folder+eval_filename)

best_model_path = os.path.join('models', 'best_weights')
bert_dir = 'uncased_L-12_H-768_A-12'
bert_data_helper = BertDataHelper(bert_dir)
val_data = bert_data_helper.convert(val_data,data='coqa')

model = BertCoQA(bert_dir=bert_dir,answer_verification=True)
print("loading model")
model.load(best_model_path)
print("model loaded")

my_batch_size = 6
eval_batch_generator = BatchGenerator(vocab,val_data,training=False,batch_size=my_batch_size,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'])

# To fix FailedPreconditionError: https://github.com/sogou/SMRCToolkit/issues/30
model.session.run(tf.local_variables_initializer())
pred_answer = model.evaluate(eval_batch_generator, evaluator)

# load the input data from coqa dev file
with open(eval_filepath) as f:
	eval_data = json.load(f)
eval_data = eval_data["data"]
output_save_file = "coqa_dev_data_predictions.txt"
output_src_file_for_opennmt_seq2seq_models = "src_coqa_dev_data_predictions.txt"
output_src_gold_answer_file_for_opennmt_seq2seq_models = "src_coqa_dev_data.txt"
with open(output_save_file, "w") as writer, open(output_src_file_for_opennmt_seq2seq_models, "w") as s_writer, open(output_src_gold_answer_file_for_opennmt_seq2seq_models, "w") as g_writer:
	for instance in eval_data:
		id = instance["id"]
		for question, answer in zip(instance["questions"], instance["answers"]):
			q = question["input_text"]
			turn_number = question["turn_id"]
			writer.write("Gold q{}\t:{}\n".format(turn_number, q))
			a = answer["input_text"]
			turn_number = answer["turn_id"]
			writer.write("Gold a{}\t:{}\n".format(turn_number, a))
			pred_a = pred_answer[(id, turn_number)]
			writer.write("Pred a{}\t:{}\n".format(turn_number, pred_a))
			tokenized_q = mt.tokenize(q.lower(), return_str=True, escape=False).strip()
			tokenized_pred_a = mt.tokenize(pred_a.lower(), return_str=True, escape=False).strip()
			tokenized_a = mt.tokenize(a.lower(), return_str=True, escape=False).strip()
			if not tokenized_q.endswith("?"):
				# Manually inserting question mark when there is'nt
				tokenized_q += " ?"
			s_writer.write("{} ||| {}\n".format(tokenized_q, tokenized_pred_a))
			g_writer.write("{} ||| {}\n".format(tokenized_q, tokenized_a))
		writer.write("\n\n")
print(pred_answer)
# We want to collect the passages for the QA pairs in the test files
# These will be useful when we want to get predictions of the coqa baseline models on them
import sys
import os
import json
sys.path.append(os.path.join("/", "home", "baheti", "QADialogueSystem", "Data"))
from QA_datasets.squad_reader import *
from sacremoses import MosesTokenizer
mt = MosesTokenizer()
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
spacy_tokenizer = SpacyWordSplitter(language='en_core_web_sm')


def get_coqa_json_format_from_instance(q, a, paragraph, id):
	# print("from the dataset")
	# print(q)
	# print(a)
	# coqa keeps the paragraph untokenized

	# We will take the q, a and the passage and convert them into json so that it can be
	# read directly by the coqa dataset reader
	json_dict = dict()
	# Add the passage
	json_dict["source"] = "wikipedia"
	json_dict["filename"] = "squad_seq2seq_train_moses_tokenized"
	json_dict["name"] = "squad_seq2seq_train_moses_tokenized"
	json_dict["story"] = paragraph
	json_dict["id"] = id
	# Find the start and end index of answer
	answer_start_index = paragraph.index(a)
	answer_size = len(a)
	answer_end_index = answer_start_index + answer_size

	# save the questions as a list of questions
	json_dict["questions"] = [{"input_text": q, "turn_id": 1}]
	# save the answer as a list of answers
	json_dict["answers"] = [{"span_start": answer_start_index, "span_end": answer_end_index, "span_text": a, "input_text": a, "turn_id": 1}]

	return json_dict

def create_coqa_format_file_from_qa_and_passages(squad_qas, src_file, coqa_format_save_file):
	# Create a dictionary mapping from squad_train_qas
	squad_qas_dict = dict()
	for q, a, passage in squad_qas:
		q_moses = mt.tokenize(q.lower().strip(), return_str=True, escape=False)
		squad_qas_dict[q_moses] = (q.strip(),a.strip(),passage.strip())

	with open(src_file, "r") as s_reader, open(coqa_format_save_file, "w") as writer:
		all_dicts = list()
		for i, src in enumerate(s_reader):
			q, a = src.strip().split(" ||| ")
			q = q.strip()
			a = a.strip()
			# print(src)
			# print(q)
			# print(a)
			json_dict = get_coqa_json_format_from_instance(*squad_qas_dict[q], str(i))
			all_dicts.append(json_dict)
		writer.write("{}\n".format(json.dumps({"data":all_dicts}, indent=4)))

# squad_train_qas = read_squad_train(True)

# DATA_DIR = "squad_seq2seq_train_moses_tokenized"
# src_test_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_responses_test.txt")
# tgt_test_file = os.path.join(DATA_DIR, "tgt_squad_seq2seq_predicted_responses_test.txt")
# coqa_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_responses_test_coqa_format.json")
# create_coqa_format_file_from_qa_and_passages(squad_train_qas, src_test_file, coqa_format_test_save_file)

squad_dev_qas = read_squad_dev(True)

DATA_DIR = "squad_seq2seq_dev_moses_tokenized"
src_test_file = os.path.join(DATA_DIR, "src_squad_seq2seq_dev_moses_test.txt")
coqa_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_dev_moses_test_coqa_format.json")
create_coqa_format_file_from_qa_and_passages(squad_dev_qas, src_test_file, coqa_format_test_save_file)





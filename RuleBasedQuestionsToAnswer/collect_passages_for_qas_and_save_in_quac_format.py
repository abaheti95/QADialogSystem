# We want to collect the passages for the QA pairs in the test files
# These will be useful when we want to get predictions of the QUAC baseline models on them
import sys
import os
import json
sys.path.append(os.path.join("/", "home", "baheti", "QADialogueSystem", "Data"))
from QA_datasets.squad_reader import *
from sacremoses import MosesTokenizer
mt = MosesTokenizer()
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
spacy_tokenizer = SpacyWordSplitter(language='en_core_web_sm')


DATA_DIR = "squad_seq2seq_train_moses_tokenized"
src_test_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_responses_test.txt")
tgt_test_file = os.path.join(DATA_DIR, "tgt_squad_seq2seq_predicted_responses_test.txt")
quac_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_responses_test_quac_format.txt")
squad_qas = read_squad_train(True)

# Squad dev test set
DATA_DIR = "squad_seq2seq_dev_moses_tokenized"
src_test_file = os.path.join(DATA_DIR, "src_squad_seq2seq_dev_moses_test.txt")
quac_format_test_save_file = os.path.join(DATA_DIR, "squad_dev_test_quac_format.txt")
squad_qas = read_squad_dev(True)

# Create a dictionary mapping from squad_qas
squad_train_qas_dict = dict()
for q, a, passage in squad_qas:
	q_moses = mt.tokenize(q.lower().strip(), return_str=True, escape=False)
	squad_train_qas_dict[q_moses] = (q,a,passage)


def get_quac_json_format_from_instance(q, a, paragraph):
	# NOTE: fixing a specific tokenization error
	paragraph = paragraph.replace(".[", ". [").replace("~11,600", "~ 11,600").replace("Japanese imports became", "Japan ese imports became").replace("[citation needed]", " [ citation needed ] ")
	if "@@UNKNOWN@@" in paragraph:
		print(q)
		print(a)
		print(paragraph)
		exit()
	# print("from the dataset")
	# print(q)
	# print(a)
	# QUAC expects everything to be spacy word tokenized

	# We will take the q, a and the passage and convert them into json so that it can be
	# read directly by the allennlp reader
	json_dict = dict()
	json_dict["paragraphs"] = list()
	json_dict["paragraphs"].append(dict())
	# Add the passage
	tokenized_paragraph = spacy_tokenizer.split_words(paragraph.strip())
	tokenized_answer = spacy_tokenizer.split_words(a.strip())
	# Find the start and end index of tokenized answer
	# print(tokenized_paragraph)
	# print(tokenized_answer)
	current_answer_pos = 0
	answer_size = len(tokenized_answer)
	answer_start_index, answer_end_index = -1, -1
	for i, tok in enumerate(tokenized_paragraph):
		# try:
		# 	tokenized_answer[current_answer_pos]
		# except IndexError:
		# 	print(tokenized_answer)
		# 	print(current_answer_pos)
		# 	print(answer_size)
		# 	exit()
		if tok.text == tokenized_answer[current_answer_pos].text or \
			(tokenized_answer[current_answer_pos].text.isdigit() and \
				(tok.text.startswith(tokenized_answer[current_answer_pos].text) or \
				tok.text.endswith(tokenized_answer[current_answer_pos].text)) \
			):
			if current_answer_pos == 0:
				answer_start_index = i
			if current_answer_pos == (answer_size-1):
				# found the answer indices
				answer_end_index = i
				break
			current_answer_pos += 1
		else:
			# reset everything
			current_answer_pos = 0
			answer_start_index, answer_end_index = -1, -1
	if answer_start_index == -1:
		print("Error!")
		print(q)
		print(a)
		print(tokenized_paragraph)
		exit()
	# save the start and the end index in quac format
	# print(answer_start_index, answer_end_index)

	json_dict["paragraphs"][0]['context'] = paragraph.strip()
	json_dict["paragraphs"][0]['qas'] = list()
	qa_dict = dict()
	qa_dict["question"] = q.strip()
	qa_dict["followup"] = "n"
	qa_dict["yesno"] = "x"
	qa_dict["id"] = "0"
	qa_dict["answers"] = list()
	qa_dict["answers"].append({"text": a, "answer_start": answer_start_index, "answer_end": answer_end_index})
	json_dict["paragraphs"][0]['qas'].append(qa_dict)
	json_dict["paragraphs"][0]["id"] = "0"

	return json_dict

with open(src_test_file, "r") as s_reader, open(quac_format_test_save_file, "w") as writer:
	for i, (src) in enumerate(s_reader):
		q, a = src.strip().split(" ||| ")
		q = q.strip()
		a = a.strip()
		# print(src)
		# print(q)
		# print(a)
		json_dict = get_quac_json_format_from_instance(*squad_train_qas_dict[q])
		writer.write("{}\n".format(json.dumps(json_dict)))
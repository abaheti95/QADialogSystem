import os
import json

dirname = os.path.dirname(__file__)
COQA_TRAIN_SET = os.path.join(dirname, "coqa", "coqa-train-v1.0.json")
COQA_DEV_SET = os.path.join(dirname, "coqa", "coqa-dev-v1.0.json")

def read_coqa_json(filename):
	# we will return a list of dialogs. Each dialog is a tuple of question, answer and span answer
	all_dialogs = list()
	with open(filename) as json_data:
		coqa = json.load(json_data)['data']
		# coqa is a list of dicts. Each dict being one dialog
		for dialog in coqa:
			questions = dialog['questions']
			answers = dialog['answers']
			# print(dialog.keys())
			# Some entries don't have addtional_answer and we are not using it anyway so simply commenting out this part
			# try:
			# 	additional_answers = dialog['additional_answers']
			# except KeyError:
			# 	print(dialog.keys())

			# questions is a list of dicts, each dict has turn id and question text
			question_texts = [q['input_text'] for q in questions]
			answers_spans = [a['span_text'] for a in answers]
			answers_texts = [a['input_text'] for a in answers]
			# Note not keeping the answer text
			# dialogs = [(q,a_t,a_s) for q,a_t,a_s in zip(question_texts, answers_texts, answers_spans)]
			dialogs = [(q,a_s) for q,a_s in zip(question_texts, answers_spans)]
			# dialog = (question_texts, answers_texts, answers_spans)
			# print(dialog)
			#TODO: ignoring the additional answers for now
			# dialogs.append(dialog)
			all_dialogs.extend(dialogs)
	return all_dialogs

def read_coqa_dev():
	# read the json file and get all the dev questions and answers
	return read_coqa_json(COQA_DEV_SET)

def read_coqa_train():
	# read the json file and get all the train questions and answers
	return read_coqa_json(COQA_TRAIN_SET)

def print_list(l):
	for e in l:
		print(e)
	print("")

# print_list(read_coqa_dev())
# print_list(read_coqa_train())
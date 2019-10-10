# we will extract all the QA pairs from the quac dataset
import os
import json

abs_path = os.path.abspath(os.path.dirname(__file__))

QUAC_TRAIN_SET = os.path.join(abs_path, "quac", "train_v0.2.json")
QUAC_DEV_SET = os.path.join(abs_path, "quac", "val_v0.2.json")

def read_quac_json(filename):
	# we will return a list of dialogs. Each dialog is a tuple of questions, answers
	dialogs = list()
	with open(filename) as json_data:
		quac = json.load(json_data)['data']
		for experiment_dict in quac:
			for paragraphs_dict in experiment_dict["paragraphs"]:
				# print(paragraphs_dict.keys())
				qas = paragraphs_dict["qas"]
				for qa in qas:
					q = qa["question"]
					ans = list(set([a_dict["text"] for a_dict in qa["answers"]]))
					# In case of multiple answer we will make multiple pairs
					for a in ans:
						dialogs.append((q, a))
					# print((q, a))
	return dialogs

def read_quac_dev():
	# read the json file and get all the dev questions and answers
	return read_quac_json(QUAC_DEV_SET)

def read_quac_train():
	# read the json file and get all the train questions and answers
	return read_quac_json(QUAC_TRAIN_SET)

def print_list(l):
	for e in l:
		print(e)
	print("")

# print_list(read_quac_dev())
# print_list(read_quac_train())
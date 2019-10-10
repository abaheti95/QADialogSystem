import os
import json
import codecs

dirname = os.path.dirname(__file__)
SQUAD_DIR = os.path.join(dirname, "SQuAD dataset")
squad_train_json = os.path.join(SQUAD_DIR, "train-v2.0.json")
squad_dev_json = os.path.join(SQUAD_DIR, "dev-v2.0.json")

def read_squad_json(filename, store_passage=False):
	all_qas = list()
	with codecs.open(filename, "r", "utf8") as reader:
		train_data = json.load(reader)['data']
		for i in range(len(train_data)):
			paragraphs = train_data[i]['paragraphs']
			# Paragraphs is a list of paragraph from the document and each entry in that list is one paragraph with list of questions and answers
			for paragraph in paragraphs:
				context = paragraph['context']
				qas = paragraph['qas']
				for qa in qas:
					question = qa['question']
					id = qa['id']
					is_impossible = qa['is_impossible']
					if not is_impossible:
						answer = qa['answers'][0]['text']
						if store_passage:
							all_qas.append((question, answer, context))
						else:
							all_qas.append((question, answer))
	return all_qas

def read_squad_dev(store_passage=False):
	# read the json file and get all the dev questions and answers
	return read_squad_json(squad_dev_json, store_passage)

def read_squad_train(store_passage=False):
	# read the json file and get all the train questions and answers
	return read_squad_json(squad_train_json, store_passage)

# we will extract all the QA pairs from the RACE dataset
import os
import json

abs_path = os.path.abspath(os.path.dirname(__file__))

#NOTE: a lot of answers here have less than 5 words. Also some of the questions have fill in the blanks style question which I am not sure is very relevant. Still preprocessing it to see what it looks like
# Also these questions are organized in different files rather than in one file so we will have dev, test and train directories instead of files
RACE_DEV_HIGH_DIR = os.path.join(abs_path, "RACE", "dev", "high")
RACE_DEV_MIDDLE_DIR = os.path.join(abs_path, "RACE", "dev", "middle")
RACE_TRAIN_HIGH_DIR = os.path.join(abs_path, "RACE", "train", "high")
RACE_TRAIN_MIDDLE_DIR = os.path.join(abs_path, "RACE", "train", "middle")

def read_questions_from_dir(dirname):
	all_qas = list()
	for filename in os.listdir(dirname):
		full_filename = os.path.join(dirname, filename)
		# Read the json from the file
		with open(full_filename) as json_data:
			qa_json = json.load(json_data)
			keys = [ord(key) - ord('A') for key in qa_json["answers"]]
			# keys = qa_json["answers"]
			options = qa_json["options"]
			questions = qa_json["questions"]
			answers = [options[i][keys[i]] for i in range(len(options))]
			qas = [(q, a) for q,a in zip(questions, answers)]
			# NOTE: filter if question doesn't have a question mark ? and if the answer length is less than 5
			# filtered_qas = [qa for qa in qas if ('?' in qa[0] and len(qa[1].split()) >= 5)]
			filtered_qas = [qa for qa in qas if '?' in qa[0]]
			# print(filtered_qas)
			all_qas.extend(filtered_qas)
	return all_qas

def read_race_dev():
	all_qas = list()
	all_qas.extend(read_questions_from_dir(RACE_DEV_HIGH_DIR))
	all_qas.extend(read_questions_from_dir(RACE_DEV_MIDDLE_DIR))
	return all_qas

def read_race_train():
	all_qas = list()
	all_qas.extend(read_questions_from_dir(RACE_TRAIN_HIGH_DIR))
	all_qas.extend(read_questions_from_dir(RACE_TRAIN_MIDDLE_DIR))
	return all_qas

def print_list(l):
	for e in l:
		print(e)
	print(len(l))

# print_list(read_race_dev())
# print_list(read_race_train())
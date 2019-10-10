import os
import json
import gzip
from pprint import pprint
abs_path = os.path.dirname(__file__)
DATA_DIR = os.path.join(abs_path, "natural_questions", "v1.0")
DEV_DATA_DIR = os.path.join(DATA_DIR, "dev")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

discarded_question_count = 0

def get_answer_from_tokens(document_tokens, start_token, end_token):
	answer = ""
	for i in range(start_token, end_token):
		answer += document_tokens[i]["token"] + " "
	return answer.strip()

def print_all_short_answers(short_answers):
	for s_answer in short_answers:
		print(s_answer)
	print()

# ref: https://stackoverflow.com/a/2556252/4535284
def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)

def read_natural_questions_data(filepath):
	global discarded_question_count
	all_qas = list()
	with open(filepath, "r") as reader:
		for line in reader:
			data = json.loads(line)
			question = data['question_text']
			# print(data.keys())
			short_answers = list()
			for annotation in data["annotations"]:
				short_answer = annotation['short_answers']
				# print(short_answer)
				if len(short_answer) > 1:
					answer = ""
					for s_answer in short_answer:
						answer += get_answer_from_tokens(data["document_tokens"], s_answer["start_token"], s_answer["end_token"]) + " , "
					# print("BHAIS KI TAANG")
					answer = answer.strip()
					if answer.endswith(','):
						answer = rreplace(answer, ',', "", 1)
					short_answers.append(answer)
				elif len(short_answer) == 1:
					# try:
					# print(type(short_answer[0]["start_token"]))
					# print(type(short_answer[0]["end_token"]))
					short_answers.append(get_answer_from_tokens(data["document_tokens"], short_answer[0]["start_token"], short_answer[0]["end_token"]))
					# print()
					# print(get_answer_from_tokens(data["document_tokens"], short_answer[0]["start_token"], short_answer[0]["end_token"]))
					# print()

			# print(data['annotations'][0].keys())
			# print(len(data['annotations']))
			
			# remove duplicate answers
			short_answers = list(set(short_answers))
			if(len(short_answers) > 0):
				# print("Question: " + question)
				question = question.strip()
				if not question.endswith("?"):
					question = question + " ?"
				# print(question)
				answer = min(short_answers, key=len)
				all_qas.append(question, answer)
			else:
				# discarding question
				discarded_question_count += 1
	print(len(all_qas))
	return all_qas

def read_from_folder(dirname):
	all_qas = list()
	print(dirname)
	for filename in os.listdir(dirname):
		# only read the files with extention .jsonl
		if filename.endswith(".jsonl"):
			all_qas.extend(read_natural_questions_data(os.path.join(dirname, filename)))
	return all_qas

def read_natural_questions_dev():
	# read the jsonl file and get all the dev questions and answers
	return read_from_folder(DEV_DATA_DIR)

def read_natural_questions_train():
	# read the jsonl file and get all the train questions and answers
	return read_from_folder(TRAIN_DATA_DIR)
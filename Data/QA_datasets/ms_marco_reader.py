import os
import json
dirname = os.path.dirname(__file__)
MS_MARCO_DEV_SET = os.path.join(dirname, "ms_marco", 'dev_v2.1.json')
MS_MARCO_TRAIN_SET = os.path.join(dirname, "ms_marco", 'train_v2.1.json')

def print_one_sample_from_dict(d):
	for key, value in d.iteritems():
		print(key, value)
		break

def read_ms_marco_json(filename):
	# list of questions and answers
	qa_list = list()
	with open(filename) as json_data:
		ms_marco = json.load(json_data)
		query = ms_marco['query']
		query_id = ms_marco['query_id']
		query_type = ms_marco['query_type']
		answers = ms_marco['answers']

		query_ids = query.keys()
		for id in query_ids:
			# print(type(query[id]), query[id])
			# print(type(answers[id][0]), answers[id][0])
			qa_list.append((query[id], answers[id][0]))
		# print_one_sample_from_dict(query)
		# print_one_sample_from_dict(query_id)
		# print_one_sample_from_dict(query_type)
		# print_one_sample_from_dict(answers)
	return qa_list

def read_ms_marco_dev():
	return read_ms_marco_json(MS_MARCO_DEV_SET)

def read_ms_marco_train():
	return read_ms_marco_json(MS_MARCO_TRAIN_SET)

def print_list(l):
	for e in l:
		print(e)
	print("")

# print_list(read_ms_marco_dev())
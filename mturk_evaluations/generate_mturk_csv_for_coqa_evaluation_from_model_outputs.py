# We want to create a csv file for annoation from all the different system's output.
# First we will accumulate all the outputs of the different systems in a list.
# Then we will randomize/shuffle that list and write them in buckets of 10

import os
import csv
import ast
import random
import string
from sacremoses import MosesTokenizer
mt = MosesTokenizer()

random.seed(901)

def print_list(l):
	for e in l:
		print(e)
	print()

correct_answers_dict = dict()

def read_dev_test_q_and_a(src_dev_file):
	global correct_answers_dict
	# we will make a list of tuples containing ids, question and answer from the dev test src file
	q_list = list()
	q_set = set()
	with open(src_dev_file, "r") as reader:
		for i, line in enumerate(reader):
			q, a = line.strip().split(" ||| ")
			q_set.add(q)
			correct_answers_dict[q] = a
			q_list.append(((i+1), q, a))
	#NOTE: just checked. All questions are unique
	# print("set size:", len(q_set))
	return q_list


def read_coqa_src_q_and_a(coqa_src_file):
	q_list = list()
	with open(coqa_src_file, "r") as reader:
		for i, line in enumerate(reader):
			q, a = line.strip().split(" ||| ")
			correct_a = correct_answers_dict[q]
			q_list.append(((i+1), q, correct_a, a))
	return q_list

def read_coqa_output(coqa_output_file):
	# dictionary of id and question and response
	qr_dict = dict()

	with open(coqa_output_file, "r") as reader:
		current_id = -1
		for line in reader:
			line = line.strip()
			if line.startswith("Q "):
				current_id_str, question = line.split("\t\t:")
				new_id = int(current_id_str.replace("Q ", ""))
				# print(new_id)
				# print(question)
				if new_id != current_id:
					# print(current_id, new_id)
					current_id = new_id
				qr_dict[current_id] = question
			if line.startswith("Gold\t\t:"):
				gold_answer = line.replace("Gold\t\t:", "")
				qr_dict[current_id] = (qr_dict[current_id], gold_answer)
			if line.startswith("Coqa Pred\t:"):
				coqa_prediction_answer = line.replace("Coqa Pred\t:", "")
				qr_dict[current_id] = (*qr_dict[current_id], coqa_prediction_answer)
				# print(current_id, qr_dict[current_id])
	sorted_qr = sorted(qr_dict.items(), key=lambda kv: kv[0])
	final_qar_list = list()
	for id, (q, a, r) in sorted_qr:
		correct_a = correct_answers_dict[q]
		final_qar_list.append((id, q, correct_a, r))
	return final_qar_list

def read_opennmt_output(opennmt_output_file):
	# dictionary of id and question and response
	qr_dict = dict()

	with open(opennmt_output_file, "r") as reader:
		for line in reader:
			line = line.strip()
			line = line.replace("PRED SCORE: ", "")
			line = line.replace("PRED AVG SCORE: ", "")
			if line.startswith("SENT "):
				sent_part, src_list_part = line.split(": [")
				id = int(sent_part.replace("SENT ", ""))
				src_list_part = "[" + src_list_part
				src_list = ast.literal_eval(src_list_part)
				question = ' '.join(src_list[:src_list.index('|||')])
				answer = ' '.join(src_list[src_list.index('|||')+1:])
				qr_dict[id] = (question, answer)
			if line.startswith("PRED "):
				pred_part, response = line.split(": ", 1)
				id = int(pred_part.replace("PRED ", ""))
				qr_dict[id] = (*qr_dict[id], response)
	sorted_qr = sorted(qr_dict.items(), key=lambda kv: kv[0])
	final_qar_list = list()
	for id, (q, a, r) in sorted_qr:
		correct_a = correct_answers_dict[q]
		final_qar_list.append((id, q, correct_a, r))
		# print(id, q, ":::", a)
		# print(r)
	return final_qar_list


DATA_FOLDER = "data"
src_test_file = os.path.join(DATA_FOLDER, "src_coqa_dev_data_filtered_mturk_sample.txt")
coqa_output = os.path.join(DATA_FOLDER, "src_coqa_dev_data_predictions_filtered_mturk_sample.txt")
basic_and_pretraining_and_glove_and_am_output = os.path.join(DATA_FOLDER, "predictions_on_coqa_dev_data_filtered_mturk_sample_output_alternate_model_reranked.txt")
basic_and_pretraining_and_glove_and_am_output_on_coqa_model_predictions = os.path.join(DATA_FOLDER, "predictions_on_coqa_dev_data_predictions_filtered_mturk_sample_output_alternate_model_reranked.txt")
## LABELS
# coqa model response = c
# basic + pretraining + glove + am = bpga
# basic + pretraining + glove + am + squad model predictions = bpga_c


def attach_label_to_qar_list(qar_list, label):
	# Also simultaneously check how many questions did each model get correct
	new_qar_list = list()
	correct_count = 0
	for tup in qar_list:
		_, q, a, r = tup
		a = a.lower()
		if a in r:
			correct_count += 1
		# elif label == 'quac':
		# 	# print(a, "::", r)
		# 	pass
		new_qar_list.append((*tup, label))
	print(label, ":", correct_count, "/", len(qar_list))
	return new_qar_list

id_q_a_list = read_dev_test_q_and_a(src_test_file)
c_qar_list = attach_label_to_qar_list(read_coqa_src_q_and_a(coqa_output), "c")
bpga_c_qar_list = attach_label_to_qar_list(read_opennmt_output(basic_and_pretraining_and_glove_and_am_output_on_coqa_model_predictions), "bpga_c")
bpga_qar_list = attach_label_to_qar_list(read_opennmt_output(basic_and_pretraining_and_glove_and_am_output), "bpga")

# Final batch with all squad model predictions
all_qars = [c_qar_list, bpga_c_qar_list, bpga_qar_list]

# Group responses based on q,a
total_unique_response = 0
all_qar_dict = dict()
for id, q, a in id_q_a_list:
	# For each question we will aggregate same responses from different models into one instance. While appending the labels
	# Gather responses and labels from all models for current q,a
	current_responses_and_labels = dict()
	for qar_list in all_qars:
		try:
			qar_id, qar_q, qar_a, qar_response, qar_label = qar_list[id-1]
		except IndexError:
			print(id)
			exit()
		if id != qar_id:
			print("Serious error in gathering!")
			exit()
		if qar_response in current_responses_and_labels:
			current_responses_and_labels[qar_response] += ":" + qar_label
		else:
			current_responses_and_labels[qar_response] = qar_label
	total_unique_response += len(current_responses_and_labels)
	all_qar_dict[q] = (id, a, current_responses_and_labels)
	# print(id, len(current_responses_and_labels))
print(total_unique_response, 300)
print(len(all_qar_dict))

def convert_dict_to_array(qars):
	qars_list = list()
	# sort by ids
	sorted_qars = sorted(qars.items(), key=lambda kv: kv[1][0])
	# N = len(sorted_qars)
	# First batch
	N_start = 0
	N_end = 100
	# convert all instances into list and shuffle
	all_instances = list()
	instance_counts = 0
	for i in range(N_start, N_end):
		q, (id, a, responses_dict) = sorted_qars[i]
		sorted_responses_dict = sorted(responses_dict.items(), key=lambda kv: kv[0])
		for response, label in sorted_responses_dict:
			all_instances.append((id, q, a, response, label))
	random.shuffle(all_instances)
	print(len(all_instances))
	return all_instances



def save_qars_to_mturk_csv(qars, csv_save_file):
	n = 10
	with open(csv_save_file, "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(["id0","question0","answer0","response0","label0","id1","question1","answer1","response1","label1","id2","question2","answer2","response2","label2","id3","question3","answer3","response3","label3","id4","question4","answer4","response4","label4","id5","question5","answer5","response5","label5","id6","question6","answer6","response6","label6","id7","question7","answer7","response7","label7","id8","question8","answer8","response8","label8","id9","question9","answer9","response9","label9"])
		#TODO: continue here!
		all_qars_list = convert_dict_to_array(qars)
		# print_list(all_qars_list)
		for i in range(0, len(all_qars_list), n):
			hit_qars = all_qars_list[i:i+n]
			if len(hit_qars) < n:
				# fill the remaining from the top of the list
				hit_qars.extend(all_qars_list[0:n-len(hit_qars)])
				# print("yess:", len(hit_qars))
			hit_qars = list(sum([(id,q,a,r,label) for (id,q,a,r,label) in hit_qars], ()))
			writer.writerow(hit_qars)


output_csv = "coqa_evaluation_batch1_new_format.csv"
save_qars_to_mturk_csv(all_qar_dict, output_csv)




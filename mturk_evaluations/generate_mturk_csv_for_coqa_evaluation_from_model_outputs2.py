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

def read_dialogpt_model_outputs(dialogpt_output_file):
	# dictionary of id and question and response
	final_qar_list = list()
	q_flag = True
	r_flag = False
	with open(dialogpt_output_file, "r") as reader:

		for i,line in enumerate(reader):
			if q_flag:
				# read q and id
				q_id, question = line.split(":", 1)
				question = question.strip()
				# print(q_id, question)
				q_flag = False
				r_flag = True
			elif not q_flag and r_flag:
				# if not line.strip():
				# 	print("WTF:", i, q_id, question)
				# 	print(":", line, ":")
				# read the response for this id and save
				line_spl = line.split("\t")
				response = line_spl[-1].strip()
				# Save q and response id in the list
				q = question.split(" ||| ")[0].strip()
				final_qar_list.append((int(q_id)+1, q, correct_answers_dict[q], response))
				r_flag = False
			elif not line.split():
				# reset q_flag
				q_flag = True
	return final_qar_list


DATA_FOLDER = "data2"
src_test_file = os.path.join(DATA_FOLDER, "src_coqa_dev_data_filtered_mturk_sample.txt")
coqa_output = os.path.join(DATA_FOLDER, "src_coqa_dev_data_predictions_filtered_mturk_sample.txt")
dgpt_output_on_coqa_dev_test = os.path.join(DATA_FOLDER, "dialoGPT_ss_plus_finetuned_predictions_on_coqa_dev_test_length_normalized_scores_new_best.txt")
dgpt_output_on_coqa_dev_test_coqa_model_predictions = os.path.join(DATA_FOLDER, "dialoGPT_ss_plus_finetuned_predictions_on_coqa_dev_test_with_coqa_model_length_normalized_scores_new_best.txt")
## LABELS
# coqa model response = c
# dgpt = dgpt
# dgpt + oracle answers = dgpt-o


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
dgpt_qar_list = attach_label_to_qar_list(read_dialogpt_model_outputs(dgpt_output_on_coqa_dev_test_coqa_model_predictions), "dgpt")
dgpt_o_qar_list = attach_label_to_qar_list(read_dialogpt_model_outputs(dgpt_output_on_coqa_dev_test), "dgpt-o")

# Final batch with all squad model predictions
all_qars = [c_qar_list, dgpt_qar_list, dgpt_o_qar_list]

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


output_csv = "new_final_coqa_evaluation_batch1.csv"
save_qars_to_mturk_csv(all_qar_dict, output_csv)




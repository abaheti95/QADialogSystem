# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
import re
import random
from collections import Counter

random.seed(901)

def evaluate(scores, predictions, gold, gold_bucket_indices):
	# MRR
	# divide the prediction scores and gold labels into buckets
	first_index = 0
	MRR = 0.0
	avg_pos = 0.0
	all_pos = list()
	ignored_count = 0.0
	# list of scores, predictions and gold excluding the ignore cases
	sub_scores = list()
	sub_predictions = list()
	sub_gold = list()
	for last_index in gold_bucket_indices:
		# print("gold", type(gold))
		# print("scores", type(scores))
		if sum(gold[first_index:last_index]) == 0:
			ignored_count += 1.0
			first_index = last_index
			continue
		# print("Comparing:", sum(gold[first_index:last_index]), sum(scores[first_index:last_index]))
		tuples = zip(list(range(first_index, last_index)), gold[first_index:last_index], scores[first_index:last_index])
		sub_scores.extend(scores[first_index:last_index])
		sub_predictions.extend(predictions[first_index:last_index])
		sub_gold.extend(gold[first_index:last_index])
		# print("All golds in a block:",sum(gold[first_index:last_index]))

		sorted_by_score = sorted(tuples, key=lambda tup: tup[2], reverse=True)
		# find the rank of first correct gold
		pos = 1.0
		for index, gold_label, score in sorted_by_score:
			if gold_label == 1:
				break
			pos += 1.0
		MRR += 1.0/pos
		avg_pos += pos
		all_pos.append(pos)
		# if pos == 1.0:
		# 	print(val_data.iloc[index])
		# print(pos, sorted_by_score[int(pos)-1])
		# print(sorted_by_score)
		first_index = last_index
	acc = metrics.accuracy_score(sub_gold, sub_predictions)
	roc_auc = metrics.roc_auc_score(sub_gold, sub_scores)
	precision, recall, thresholds = metrics.precision_recall_curve(sub_gold, sub_scores, pos_label=1)
	f1s = [(2.0*p*r/(p+r), p,r) for p,r in zip(precision, recall)]
	f1_max, p_max, r_max = max(f1s, key=lambda tup: tup[0])
	print("MAX F1:", f1_max, p_max, r_max)
	pr_auc = metrics.auc(recall, precision)
	ap = metrics.average_precision_score(sub_gold, sub_scores)
	# exit()
	MRR /= (float(len(gold_bucket_indices)) - ignored_count)
	avg_pos /= (float(len(gold_bucket_indices)) - ignored_count)
	all_pos_counter = Counter(all_pos)
	print("All pos")
	print(all_pos_counter)
	print("ignored_count:", ignored_count)
	print("\n\nAVG POS:", avg_pos, "\n\n")
	cm = metrics.confusion_matrix(gold, predictions)
	classification_report = metrics.classification_report(gold, predictions, output_dict=True)
	classification_report_str = metrics.classification_report(gold, predictions)

	return acc, cm, roc_auc, pr_auc, ap, f1_max, precision, recall, thresholds, MRR, all_pos_counter[1.0]/(float(len(gold_bucket_indices)) - ignored_count), classification_report, classification_report_str

def verify_and_generate_bucket_indices(df):
	# we need to verify if the data is organized as blocks of questions and thier responses and not any other way
	explored_qa = set()
	current_qa = None
	current_1_flag = True			# if this flag is True then it means that we are expecting all count > 0. Once we encounter a zero count we will set it to False
	gold_bucket_indices = list()	# This is the list of indices at which the buckets end
	for index, row in df.iterrows():
		q, a, count = row[0], row[1], row[102]
		if current_qa != (q,a):
			if current_qa in explored_qa:
				print("Verification Error: This question is coming again:", current_qa)
				exit()
			if index != 0:
				gold_bucket_indices.append(index)
			explored_qa.add(current_qa)
			current_qa = (q,a)
			current_1_flag = True
		if count == 0:
			current_1_flag = False
		if not current_1_flag and count > 0:
			# This is unexpected
			print("Verification ERROR in index:", index)
			exit()
	gold_bucket_indices.append(index+1)
	print("Verification Successful")
	return gold_bucket_indices

import os
DATA_DIR = "train_data"
SHORTEST_RESPONSE_RESULTS_DIR = os.path.join(DATA_DIR, "shortest_response_results")
LM_RESULTS_DIR = os.path.join(DATA_DIR, "lm_results")

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)

make_dir_if_not_exists(SHORTEST_RESPONSE_RESULTS_DIR)
make_dir_if_not_exists(LM_RESULTS_DIR)
NEGATIVE_PERCENT = 100

val_file = os.path.join(DATA_DIR, "val_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")
train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENT))

# With new count2 files
val_file = os.path.join(DATA_DIR, "val_count2_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count2_squad_final_train_data_features.tsv")
train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENT))

# With new shortest response count2 files
val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_features.tsv")
train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENT))

# val_file = os.path.join(DATA_DIR, "val_count1_squad_final_train_data_features_removed_short_response_affinity_workers.tsv")
# test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features_removed_short_response_affinity_workers.tsv")
# train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative_removed_short_response_affinity_workers.tsv".format(NEGATIVE_PERCENT))

# test_file = val_file

# test_data = pd.read_csv(test_file, sep='\t', header=None)
# test_bucket_indices = verify_and_generate_bucket_indices(test_data)

def compute_labels_based_on_lm(responses_rules_and_lm_probs):
	f = lambda i: responses_rules_and_lm_probs[i][2]
	argmax = max(range(len(responses_rules_and_lm_probs)), key=f)
	labels = [0] * len(responses_rules_and_lm_probs)
	labels[argmax] = 1
	lm_scores = [lm_prob for r, rule, lm_prob in responses_rules_and_lm_probs]
	return labels, lm_scores

def compute_labels_based_on_shortest_response(responses_rules_and_lm_probs):
	# Find all the shortest response in the list and then pick randomly
	shortest_response_indices = list()
	shortest_r_tok = 100000000.0
	for i, (r, rule, lm_prob) in enumerate(responses_rules_and_lm_probs):
		n_tok_in_r = len(r.lower().strip().split())
		if n_tok_in_r < shortest_r_tok:
			shortest_r_tok = n_tok_in_r
			shortest_response_indices = [i]
		elif n_tok_in_r == shortest_r_tok:
			shortest_response_indices.append(i)
	# Randomly break ties
	argmin = random.choice(shortest_response_indices)

	labels = [0] * len(responses_rules_and_lm_probs)
	labels[argmin] = 1
	scores = [-len(r.strip().split()) for r, rule, lm_prob in responses_rules_and_lm_probs]
	if sum(labels) != 1:
		print("Error: number of shortest responses:", sum(labels))
		exit()
	return labels, list(labels)

def check_prep_or_det_added(rule):
	return "R_custom_prep_added" in rule or "R_custom_det_added" in rule

with open(test_file, 'r') as features_file:
	print("Reading Generated Responses and questions instances from features file: %s", test_file)
	current_qa = None
	current_responses_rules_and_lm_probs = list()
	lm_labels = list()
	lm_scores = list()
	shortest_response_labels = list()
	shortest_response_scores = list()
	gold_labels = list()
	my_test_bucket_indices = list()
	current_gold_counts = list()
	question_count = 0
	flag_count = 0
	for line_no, line in enumerate(features_file):
		# TODO: remove this after debugging
		# if i==10000:
		# 	break
		line = line.strip()
		row = re.split('\t|\\t', line)

		q = row[0].strip()
		q = q.lower()
		a = row[1].strip()
		a = a.lower()
		r = row[2].strip()
		r = r.lower()
		rule = row[3].strip()
		lm_prob = float(row[22]) # 3gram lm prob for response
		count = int(row[-1])
		
		if (q,a) != current_qa:
			if not current_qa:
				current_qa = (q,a)
			else:
				my_test_bucket_indices.append(line_no)
				#TODO: Move ignore_count to here
				# compute labels for current_responses_rules_and_lm_probs based on lm probs and number of tokens
				current_lm_labels, current_lm_scores = compute_labels_based_on_lm(current_responses_rules_and_lm_probs)
				lm_labels.extend(current_lm_labels)
				lm_scores.extend(current_lm_scores)

				current_shortest_response_labels, current_shortest_response_scores = compute_labels_based_on_shortest_response(current_responses_rules_and_lm_probs)
				shortest_response_labels.extend(current_shortest_response_labels)
				shortest_response_scores.extend(current_shortest_response_scores)
				# Convert counts to labels
				# Only consider those questions whose selected response has "R_custom_prep_added" or "R_custom_det_added"
				flag = False
				for i in range(len(current_gold_counts)):
					if current_gold_counts[i] > 0 and check_prep_or_det_added(current_responses_rules_and_lm_probs[i][1]):
						flag = True
						break
				if flag:
					flag_count += 1
				current_gold_labels = [0]*len(current_gold_counts)
				# if not flag:
				if True:
					# Keep this question only if the flag is true
					# generate labels from gold counts
					for i in range(len(current_gold_counts)):
						if current_shortest_response_labels[i] == 1:
							# Assign label 1 only if there are at least 3 gold counts
							if current_gold_counts[i] > 1:
								current_gold_labels[i] = 1
						else:
							if current_gold_counts[i] > 0:
								current_gold_labels[i] = 1
					print(current_qa)
					# print shortest response
					# print("Score and labels:", sum(current_shortest_response_scores), sum(current_shortest_response_labels))
					print("Shortest response:", current_responses_rules_and_lm_probs[current_shortest_response_labels.index(1)])
					# print gold responses
					indices = [i for i, e in enumerate(current_gold_labels) if e == 1]
					print("Gold responses")
					if current_shortest_response_labels.index(1) in indices:
						print("SHORTEST RESPONSE IN GOLD")
					for i in indices:
						print(current_responses_rules_and_lm_probs[i])
					print("\n")
				gold_labels.extend(current_gold_labels)
				question_count += 1
				
				# print(current_qa, (q,a))
				current_responses_rules_and_lm_probs = list()
				current_gold_counts = list()
				current_qa = (q,a)
		current_gold_counts.append(count)
		current_responses_rules_and_lm_probs.append((r, rule, lm_prob))
	if len(current_responses_rules_and_lm_probs) > 0:
		my_test_bucket_indices.append(line_no+1)
		# deal with the last question and its responses
		current_lm_labels, current_lm_scores = compute_labels_based_on_lm(current_responses_rules_and_lm_probs)
		lm_labels.extend(current_lm_labels)
		lm_scores.extend(current_lm_scores)
		current_shortest_response_labels, current_shortest_response_scores = compute_labels_based_on_shortest_response(current_responses_rules_and_lm_probs)
		shortest_response_labels.extend(current_shortest_response_labels)
		shortest_response_scores.extend(current_shortest_response_scores)
		flag = False
		for i in range(len(current_gold_counts)):
			if current_gold_counts[i] > 0 and check_prep_or_det_added(current_responses_rules_and_lm_probs[i][1]):
				flag = True
				break
		if flag:
			flag_count += 1
		current_gold_labels = [0]*len(current_gold_counts)
		# if not flag:
		if True:
			# Keep this question only if the flag is true
			# generate labels from gold counts
			for i in range(len(current_gold_counts)):
				if current_shortest_response_labels[i] == 1:
					# Assign label 1 only if there are at least 3 gold counts
					if current_gold_counts[i] > 1:
						current_gold_labels[i] = 1
				else:
					if current_gold_counts[i] > 0:
						current_gold_labels[i] = 1
			print(current_qa)
			# print shortest response
			print("Shortest response:", current_responses_rules_and_lm_probs[current_shortest_response_labels.index(1)])
			# print gold responses
			indices = [i for i, e in enumerate(current_gold_labels) if e == 1]
			print("Gold responses")
			for i in indices:
				print(current_responses_rules_and_lm_probs[i])
			print("\n")
		gold_labels.extend(current_gold_labels)
		print("final time")
		print(len(gold_labels), len(lm_labels), len(shortest_response_scores))
		question_count += 1
		current_responses_rules_and_lm_probs = list()
		current_gold_counts = list()
		current_qa = None
	# print(test_bucket_indices)
	# print(my_test_bucket_indices)
	print(question_count)
	acc, cm, roc_auc, pr_auc, ap, f1_max, precision, recall, thresholds, MRR, precision_at_1, classification_report, classification_report_str = evaluate(lm_scores, lm_labels, gold_labels, my_test_bucket_indices)
	print("LM model")
	print("MRR:", MRR)
	print("Precision@1:", precision_at_1)
	print("f1_max:", f1_max)
	# first_index = 0
	# for last_index in my_test_bucket_indices:
	# 	scores = shortest_response_scores[first_index:last_index]
	# 	print("Scores sum:", sum(scores))
	# 	first_index = last_index
	acc, cm, roc_auc, pr_auc, ap, f1_max, precision, recall, thresholds, MRR, precision_at_1, classification_report, classification_report_str = evaluate(shortest_response_scores, shortest_response_labels, gold_labels, my_test_bucket_indices)
	print("Shortest response model")
	print("MRR:", MRR)
	print("Precision@1:", precision_at_1)
	print("PR_auc:", pr_auc)
	print("Max F1:", f1_max)

	print("FLAG COUNT:", flag_count)

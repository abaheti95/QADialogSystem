# We will read the batches from the results dir and save the analysis in the same dir

import os
import csv
from collections import Counter, OrderedDict
from statistics import mean
from copy import deepcopy
import random
from random import shuffle
random.seed(904)
from nltk import agreement
from sklearn.metrics import cohen_kappa_score
import numpy as np

def print_list(l):
	for e in l:
		print(e)
	print()

DATA_DIR = "data2"
RESULTS_DIR = "results4"

final_batch1_results_file = os.path.join(RESULTS_DIR, "Batch_3839345_batch_results.csv")
final_batch1_results_file = os.path.join(RESULTS_DIR, "Batch_3842077_batch_results.csv")
final_batch2_results_file = os.path.join(RESULTS_DIR, "Batch_3846806_batch_results.csv")
final_batch3_results_file = os.path.join(RESULTS_DIR, "Batch_3848737_batch_results.csv")
final_batch4_results_file = os.path.join(RESULTS_DIR, "Batch_3852883_batch_results.csv")

# Final evaluations
# batch_results_files = [final_batch1_results_file]
batch_results_files = [final_batch1_results_file, final_batch2_results_file, final_batch3_results_file, final_batch4_results_file]

worker_time_list = list()
all_annotations = list()

def read_bad_workers(bad_workers_file):
	bad_workers = set()
	with open(bad_workers_file, "r") as reader:
		for line in reader:
			bad_workers.add(line.strip())
	return bad_workers

bad_workers_file = "blocked_workers_list.txt"
bad_workers = read_bad_workers(bad_workers_file)

ignored_HITS_count = total_HITS_count = 0
for batch_results_file in batch_results_files:
	with open(batch_results_file, "r") as in_file:
		reader = csv.reader(in_file, delimiter=',')
		head_row = next(reader)
		# ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.id0', 'Input.input0', 'Input.answer0', 'Input.output0', 'Input.label0', 'Input.id1', 'Input.input1', 'Input.answer1', 'Input.output1', 'Input.label1', 'Input.id2', 'Input.input2', 'Input.answer2', 'Input.output2', 'Input.label2', 'Input.id3', 'Input.input3', 'Input.answer3', 'Input.output3', 'Input.label3', 'Input.id4', 'Input.input4', 'Input.answer4', 'Input.output4', 'Input.label4', 'Input.id5', 'Input.input5', 'Input.answer5', 'Input.output5', 'Input.label5', 'Input.id6', 'Input.input6', 'Input.answer6', 'Input.output6', 'Input.label6', 'Input.id7', 'Input.input7', 'Input.answer7', 'Input.output7', 'Input.label7', 'Input.id8', 'Input.input8', 'Input.answer8', 'Input.output8', 'Input.label8', 'Input.id9', 'Input.input9', 'Input.answer9', 'Input.output9', 'Input.label9', 'Answer.QA0', 'Answer.QA1', 'Answer.QA2', 'Answer.QA3', 'Answer.QA4', 'Answer.QA5', 'Answer.QA6', 'Answer.QA7', 'Answer.QA8', 'Answer.QA9', 'Approve', 'Reject']
		# print(head_row)
		worker_id_key = "WorkerId"
		worker_time_key = "WorkTimeInSeconds"
		input_id0_key = "Input.id0"
		answer_qa0_key = "Answer.QA0"
		worker_id_index = head_row.index(worker_id_key)
		worker_time_index = head_row.index(worker_time_key)
		input_id0_index = head_row.index(input_id0_key)
		answer_qa0_index = head_row.index(answer_qa0_key)
		# Extract all the inputs, answers, workers and their times
		for row in reader:
			worker_id = row[worker_id_index]
			total_HITS_count += 1
			if worker_id in bad_workers:
				# ignore this HIT
				ignored_HITS_count += 1
				continue
			worker_time = int(row[worker_time_index])
			worker_time_list.append((worker_id, worker_time))
			# Extract inputs and annotations
			inputs = row[input_id0_index:input_id0_index+50]
			annotations = row[answer_qa0_index:answer_qa0_index+10]
			# print(input_id0_index+50, answer_qa0_index)
			# Save all the inputs, answers and worker times in a list
			for i in range(10):
				current_input = inputs[i*5:(i+1)*5]
				current_input[0] = int(current_input[0])
				current_annotation = annotations[i]
				all_annotations.append((*current_input, current_annotation, worker_id, worker_time))
print("Ignored HITS:", ignored_HITS_count, "/", total_HITS_count)

def compute_inter_annotator_agreement(annotations):
	# First combine the instances across annotations
	combined_annotations = dict()
	for index, q, a, r, label, assignment, worker, time in annotations:
		instance = (index, q, a, r, label)
		combined_annotations.setdefault(instance, list())
		combined_annotations[instance].append(assignment)
	# Convert combined_annotations to task data
	task_data = list()
	for instance, assignments in combined_annotations.items():
		index, q, a, r, label = instance
		shuffle(assignments)
		for i, assignment in enumerate(assignments):
			task_data.append((str(i), str(index)+label, assignment))
	ratingtask = agreement.AnnotationTask(data=task_data)
	print("kappa " +str(ratingtask.kappa()))
	print("fleiss " + str(ratingtask.multi_kappa()))
	print("alpha " +str(ratingtask.alpha()))
	print("scotts " + str(ratingtask.pi()))

# compute_inter_annotator_agreement(all_annotations)

def comupte_worker_cohens_kappa_vs_majority_vote(annotations):
	# First combine the instances across annotations
	combined_annotations = dict()
	for index, q, a, r, label, assignment, worker, time in annotations:
		instance = (index, q, a, r, label)
		combined_annotations.setdefault(instance, list())
		combined_annotations[instance].append(assignment)
	# Now create dict with workers and their assignments
	workers_annotations = dict()
	for index, q, a, r, label, assignment, worker, time in annotations:
		workers_annotations.setdefault(worker, list())
		instance = (index, q, a, r, label)
		workers_annotations[worker].append((instance, assignment, time))

	# For every worker compute cohen's kappa against the majority vote
	worker_cohens_kappas = list()
	for worker, worker_assignments in workers_annotations.items():
		w_assignments = list()
		majority_assignments = list()
		for instance, assignment, time in worker_assignments:
			all_instance_assignments = combined_annotations[instance]
			# Find majority assignments
			all_instance_assignments_counter = Counter(all_instance_assignments)
			majority_value, majority_count = all_instance_assignments_counter.most_common()[0]
			w_assignments.append(assignment)
			majority_assignments.append(majority_value)
		# Compute cohen's kappa
		worker_cohens_kappas.append((worker, cohen_kappa_score(w_assignments, majority_assignments), len(worker_assignments)))
	# Sort the list by cohen's kappas
	sorted_worker_cohens_kappas = sorted(worker_cohens_kappas, key = lambda x: x[1])
	for w, kappa, number_of_annotations in sorted_worker_cohens_kappas:
		print(w,"\t", kappa, "\t", number_of_annotations)
	# compute the weighted average cohen's kappa
	scores = [e for _, e, _ in sorted_worker_cohens_kappas]
	weights = [e for _, _, e in sorted_worker_cohens_kappas]
	print("Weighted average cohen's kappa:", np.average(scores, weights=weights))

comupte_worker_cohens_kappa_vs_majority_vote(all_annotations)
print()


# print_list(all_annotations)

# Given a model label and all annotations we want to extract the annotations specific to the given label in a new list
def extract_label_annotation(annotations, label):
	label_annotations = list()
	for e in annotations:
		labels = e[4].split(":")
		
		# print(labels)
		new_labels = list()
		for l in labels:
			if l == "gpt_ss+_oqa_r":
				l = "gpt_ss+_sm_r"
			elif l == "gpt_ss+_oqa_r_oracle":
				l = "gpt_ss+_sm_r_oracle"
			new_labels.append(l)
		labels = new_labels
		# print(labels)
		
		if label in labels:
			new_e = list(e)
			new_e[4] = label
			label_annotations.append(tuple(new_e))
	return label_annotations

# Combine scores from all annotators into a list for a set of annotations
suspicious_workers = dict()
def combine_scores_for_annotations(annotations):
	global suspicious_workers
	# we will keep a scores list and a score worker time tuple list
	instance_scores_dict = dict()
	for e in annotations:
		input = e[:5]
		score = e[5]
		worker = e[6]
		worker_time = e[7]
		instance_scores_dict.setdefault(e[:5], (list(), list()))
		instance_scores_dict[e[:5]][0].append(score)
		instance_scores_dict[e[:5]][1].append((score, worker, worker_time))
	# condense the instance score dict to a list
	instance_scores_dict = OrderedDict(sorted(instance_scores_dict.items()))
	instance_scores_list = list()
	agreement_fail_count = total_count = 0
	for instance_input, (scores_list, scores_worker_list) in instance_scores_dict.items():
		scores_counter = Counter(scores_list)
		majority_value, majority_count = scores_counter.most_common()[0]
		if majority_count < 3:
			agreement_fail_count += 1
			# Find all the words with same majority_count
			values = [k for k,v in scores_counter.items() if v == majority_count]
			values = sorted(values)
			majority_value = random.choice(values)
			# print(values, majority_value)
		total_count += 1
		instance_scores_list.append((*instance_input, majority_value, scores_list, scores_worker_list))
	# print(agreement_fail_count, "/", total_count)
	# print(suspicious_workers)
	return instance_scores_list, agreement_fail_count, total_count

def save_annotations_scores(save_file, annotation_scores, label, majority_failed_count, total_count):
	annotations = dict()
	with open(save_file, "w") as tsv_out:
		writer = csv.writer(tsv_out, delimiter='\t')
		head_row = ["avg score", "scores", "id", "question", "answer", "response"]
		writer.writerow(head_row)
		# Sort annotations by avg score
		annotation_scores = sorted(annotation_scores, key=lambda tup: tup[5], reverse=True)
		annotation_counter = Counter([e[5] for e in annotation_scores])
		avg_scores = [1 if e[5]=='e' else 0 for e in annotation_scores]
		print(label, "\t\t", majority_failed_count, "/", total_count, end="\t")
		for e in ["a", "b", "c", "d", "e"]:
			print("{0:.2f}".format(annotation_counter[e]/float(len(annotation_scores))*100.0), end="\t&\t")
		print()
		# print(avg_scores)
		for e in annotation_scores:
			annotations[int(e[0])] = [label, e[5],e[6],e[1],e[2],e[3],e[7]]
			writer.writerow([e[5],e[6],e[0],e[1],e[2],e[3],e[7]])
	return annotations

all_model_labels = ["c", "quac", "bert", "ss_pgn", "ss+_pgn", "ss_pgn_pre", "ss+_pgn_pre", "gpt_ss", "gpt_ss+", "gpt_ss_oqa", "gpt_ss+_oqa", "gpt_ss_sm", "gpt_ss+_sm", "gpt_ss+_sm_o"]
# Extract annotations for each label
# Then aggregate the score and finally save them into a file
annotations_save_file = os.path.join(RESULTS_DIR, "{}_model_annotations.tsv")
all_annotation_scores = dict()

all_annotations_dict = dict()

for label in all_model_labels:
	label_annotations = extract_label_annotation(all_annotations, label)
	label_annotations_scores, mojority_failed_count, total_count = combine_scores_for_annotations(label_annotations)
	# Save file now
	all_annotations_dict[label] = save_annotations_scores(annotations_save_file.format(label), label_annotations_scores, label, mojority_failed_count, total_count)
	all_annotation_scores[label] = sorted(label_annotations_scores, key=lambda tup: tup[0])

N = 500
all_annotations_list = list()
label_to_text = {"c":"{\coqa} B.", "quac":"{\quac} B.", "bert":"LGRs+BERT B.", "ss_pgn":"{\pgn} with SS", "ss+_pgn":"{\pgn} with SS+", "ss_pgn_pre":"{\pgnp} with SS", "ss+_pgn_pre":"{\pgnp} with SS+", "gpt_ss":"{\gpt} with SS", "gpt_ss+":"{\gpt} with SS+", "gpt_ss_oqa":"{\gptp} with SS", "gpt_ss+_oqa":"{\gptp} with SS+", "gpt_ss_sm":"{\dgpt} with SS", "gpt_ss+_sm":"{\dgpt} with SS+", "gpt_ss+_sm_o":"{\dgpt} with SS+ (o)"}
vote_to_text = {"a":"\\xmark & \\xmark & - & a \\\\ \\hline", "b":"\\cmark & \\xmark & - & b \\\\ \\hline", "c":"\\xmark & \\cmark & - & c \\\\ \\hline", "d":"\\cmark & \\cmark & \\xmark & d \\\\ \\hline", "e":"\\cmark & \\cmark & \\cmark & e \\\\ \\hline"}

save_file = os.path.join(RESULTS_DIR, "final_table.txt")
with open(save_file, "w") as writer:
	for i in range(N):
		# get all the responses for different labels in current questions
		final_string = ""
		flag = True
		all_model_labels = ["c", "quac", "bert", "ss+_pgn_pre", "gpt_ss+", "gpt_ss+_oqa", "gpt_ss+_sm", "gpt_ss+_sm_o"]
		for label in all_model_labels:
			_, majority_vote ,all_votes,q,a,r,worker_votes = all_annotations_dict[label][(i+1)]
			if flag:
				question_string = "Model  & \\textbf{Q}:" + q + " & \\rotatebox[origin=c]{270}{correctness} & \\rotatebox[origin=c]{270}{complete-sentence} & \\rotatebox[origin=c]{270}{grammaticality} & \\rotatebox[origin=c]{270}{majority option}  \\\\ \\hline \n"
				writer.write(question_string)
				flag = False
			final_string += label_to_text[label] + " & " + "\\textbf{R}: " + r + " & " + vote_to_text[majority_vote] + "\n"
		writer.write(final_string)
		writer.write("\n")





# extract K indices from all the models
"""
K = 10
random.seed(901)
K_sample_indices = random.sample(list(range(500)), K)
print(K_sample_indices)
for index in K_sample_indices:
	for label, label_annotations_scores in all_annotation_scores.items():
		instance = label_annotations_scores[index]
		print((instance[4], instance[0],instance[1], instance[2], instance[3], instance[5], instance[6]))
	print()
"""

# sum all the values of suspicious workers
print(sum(suspicious_workers.values()))
# Compare 2 models side by side
def compare_models_side_by_side_and_save(label1, label2, all_annotation_scores, save_filename):
	# We have to align the annotations by id
	# Sort both annotation_scores by id
	model1_annotation_scores = sorted(all_annotation_scores[label1], key=lambda tup: tup[0])
	model2_annotation_scores = sorted(all_annotation_scores[label2], key=lambda tup: tup[0])
	# once the id's are aligned create 2 lists
	# one's were model1 did better than model2 in terms of avg score and vice-versa
	m1_winner = list()
	m2_winner = list()
	for m1_annotation, m2_annotation in zip(model1_annotation_scores, model2_annotation_scores):
		id = m1_annotation[0]
		question = m1_annotation[1]
		m1_response = m1_annotation[3]
		m1_avg_score = m1_annotation[5]
		m1_scores = m1_annotation[6]
		m1_workers_and_scores = m1_annotation[7]
		m2_response = m2_annotation[3]
		m2_avg_score = m2_annotation[5]
		m2_scores = m2_annotation[6]
		m2_workers_and_scores = m2_annotation[7]
		if m1_response == m2_response:
			continue
		if m1_avg_score >= m2_avg_score:
			# m1_winner.append((id, question, m1_response, m1_avg_score, m1_scores, m2_response, m2_avg_score, m2_scores, m1_workers_and_scores, m2_workers_and_scores))
			m1_winner.append((id, question, m1_response, m1_avg_score, m1_scores, m2_response, m2_avg_score, m2_scores))
		else:
			# m2_winner.append((id, question, m1_response, m1_avg_score, m1_scores, m2_response, m2_avg_score, m2_scores, m1_workers_and_scores, m2_workers_and_scores))
			m2_winner.append((id, question, m1_response, m1_avg_score, m1_scores, m2_response, m2_avg_score, m2_scores))
	with open(save_filename.format(label1, label2), "w") as writer:
		writer.write("{} Winner\n".format(label1))
		for e in m1_winner:
			writer.write("{}\n".format(e))
		writer.write("\n{} Winner\n".format(label2))
		for e in m2_winner:
			writer.write("{}\n".format(e))
	return m1_winner, m2_winner

save_filename = os.path.join(RESULTS_DIR, "{}_vs_{}_annotation_comparison.txt")

# c_winner, bpga_winner = compare_models_side_by_side_and_save('c', 'bpga', all_annotation_scores, save_filename)
# b_winner, bp_winner = compare_models_side_by_side_and_save('b', 'bp', all_annotation_scores, save_filename)
# bp_winner, bpg_winner = compare_models_side_by_side_and_save('bp', 'bpg', all_annotation_scores, save_filename)
# bp_winner, bpga_winner = compare_models_side_by_side_and_save('bp', 'bpga', all_annotation_scores, save_filename)
# print("C Winner")
# print_list(c_winner)
# print("bpga Winner")
# print_list(bpga_winner)
"""
d_annotations = extract_label_annotation(all_annotations, "d")
c_annotations = extract_label_annotation(all_annotations, "c")
b_annotations = extract_label_annotation(all_annotations, "b")
bp_annotations = extract_label_annotation(all_annotations, "bp")
bpg_annotations = extract_label_annotation(all_annotations, "bpg")
bpgl_annotations = extract_label_annotation(all_annotations, "bpgl")
bpga_annotations = extract_label_annotation(all_annotations, "bpga")

d_annotations_scores = combine_scores_for_annotations(d_annotations)
c_annotations_scores = combine_scores_for_annotations(c_annotations)
b_annotations_scores = combine_scores_for_annotations(b_annotations)
bp_annotations_scores = combine_scores_for_annotations(bp_annotations)
bpg_annotations_scores = combine_scores_for_annotations(bpg_annotations)
bpgl_annotations_scores = combine_scores_for_annotations(bpgl_annotations)
bpga_annotations_scores = combine_scores_for_annotations(bpga_annotations)

# bpga_annotations_scores = sorted(bpga_annotations_scores, key=lambda tup: tup[5])
# # print_list([(e[5],e[1],e[3],e[7]) for e in bpga_annotations_scores])
# print_list([(e[5],e[6],e[1],e[3]) for e in bpga_annotations_scores])

# d_annotations_scores = sorted(d_annotations_scores, key=lambda tup: tup[5])
# # print_list([(e[5],e[1],e[3],e[7]) for e in d_annotations_scores])
# print_list([(e[5],e[6],e[1],e[3]) for e in d_annotations_scores])

c_annotations_scores = sorted(c_annotations_scores, key=lambda tup: tup[5])
# print_list([(e[5],e[1],e[3],e[7]) for e in c_annotations_scores])
print_list([(e[5],e[6],e[1],e[3]) for e in c_annotations_scores])


# sort the worker_time_list w.r.t. time and then print the sorted list
sorted_worker_time_list = sorted(worker_time_list, key=lambda tup: tup[1])
# print_list(sorted_worker_time_list)
"""

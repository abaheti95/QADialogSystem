import sys
import csv
import ast
import numpy as np
#ref: https://stackoverflow.com/a/15063941/4535284
csv.field_size_limit(sys.maxsize)
import os
from collections import Counter

# results_file = os.path.join("mturk_experiments", "Batch_3526671_batch_results.csv")
# results_file = os.path.join("mturk_experiments", "Batch_3529051_batch_results.csv")
# SQUAD first 1000 try 1 partial results
results_file1 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_1_results.csv")
results_file2 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_2_results.csv")
results_file3 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_3_results.csv")
results_file4 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_4_results.csv")
results_file5 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_5_results.csv")
results_file6 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_6_results.csv")
results_file7 = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_batch_7_results.csv")
data_file = os.path.join("mturk_experiments", "squad_questions_generated_final_3000_sample_input_corrected.csv")
results_files = [results_file1,results_file2,results_file3,results_file4,results_file5,results_file6,results_file7]
# results_files = [results_file5]
header = ["HITId","HITTypeId","Title","Description","Keywords","Reward","CreationTime","MaxAssignments","RequesterAnnotation","AssignmentDurationInSeconds","AutoApprovalDelayInSeconds","Expiration","NumberOfSimilarHITs","LifetimeInSeconds","AssignmentId","WorkerId","AssignmentStatus","AcceptTime","SubmitTime","AutoApprovalTime","ApprovalTime","RejectionTime","RequesterFeedback","WorkTimeInSeconds","LifetimeApprovalRate","Last30DaysApprovalRate","Last7DaysApprovalRate","Input.question1","Input.answer1","Input.question2","Input.answer2","Input.question3","Input.answer3","Input.question4","Input.answer4","Input.question5","Input.answer5","Input.question6","Input.answer6","Input.question7","Input.answer7","Input.question8","Input.answer8","Input.question9","Input.answer9","Input.question10","Input.answer10","Input.responses1","Input.responses2","Input.responses3","Input.responses4","Input.responses5","Input.responses6","Input.responses7","Input.responses8","Input.responses9","Input.responses10","Answer.response-selector1","Answer.response-selector10","Answer.response-selector2","Answer.response-selector3","Answer.response-selector4","Answer.response-selector5","Answer.response-selector6","Answer.response-selector7","Answer.response-selector8","Answer.response-selector9","Approve","Reject"]
bad_workers = {"A2QJP5BZ7B523H", "A2G7F16RAOFTXG", "A423QQ5WN43B9", "ARG392N6HWZCJ", "A3OVS29S2TYBQR", "A2UO3QJZNC2VOE", "A1FWP2MESHFZQG", "A2JJF5OFND2KL5", "A1TYMXIYUUUL6F", "A2GZNOA5CIVKUB", "A3ES4ZR2BMKEET", "A1J6Z70T78B35V", "AITP2LUW8GPB", "A3AGF9EJPNNZH5", "A11BLQKVW1951L", "A33QSLM9P8111P", "A7SXWHGK8B40R"}
#  "A5BMKZRGHNSRT" is a special case. This worker has very high similarity with other users yet its HIT timings are as low as 32 seconds which is shocking. It could be that some HITs are genuine and others are not. Anyhow I have blocked this worker for future HITs.
# save_train_data_file = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_train_data.tsv")
save_train_data_file = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_train_data_removed_short_response_affinity_workers.tsv")

# dictionary of different responses
grouped_responses = dict()
workers_and_time = list()
worker_responses = dict()
all_generated_responses = dict()
# populate all_generated_responses from the original data file
def string_to_list(string):
	string = "[" + string + "]"
	return ast.literal_eval(string)

with open(data_file, "r") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	head_row = next(csv_reader)
	for question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5, answer5, question6, answer6, question7, answer7, \
			question8, answer8, question9, answer9, question10, answer10, responses1, responses2, responses3, responses4, responses5, responses6, responses7, responses8, \
			responses9, responses10, rules1, rules2, rules3, rules4, rules5, rules6, rules7, rules8, rules9, rules10 in csv_reader:
		all_generated_responses[(question1, answer1)] = (string_to_list(responses1), string_to_list(rules1))
		all_generated_responses[(question2, answer2)] = (string_to_list(responses2), string_to_list(rules2))
		all_generated_responses[(question3, answer3)] = (string_to_list(responses3), string_to_list(rules3))
		all_generated_responses[(question4, answer4)] = (string_to_list(responses4), string_to_list(rules4))
		all_generated_responses[(question5, answer5)] = (string_to_list(responses5), string_to_list(rules5))
		all_generated_responses[(question6, answer6)] = (string_to_list(responses6), string_to_list(rules6))
		all_generated_responses[(question7, answer7)] = (string_to_list(responses7), string_to_list(rules7))
		all_generated_responses[(question8, answer8)] = (string_to_list(responses8), string_to_list(rules8))
		all_generated_responses[(question9, answer9)] = (string_to_list(responses9), string_to_list(rules9))
		all_generated_responses[(question10, answer10)] = (string_to_list(responses10), string_to_list(rules10))

def index_of_the_response_in_the_list(r, list_responses):
	# Sort the list responses wrt number of words
	sorted_responses = sorted(list_responses, key=lambda e: len(e.strip().split()))
	for i, response in enumerate(sorted_responses):
		if response.strip() == r.strip():
			return i+1
	print("ERROR: Response not in the list")
	print(sorted_responses)
	print(r)
	exit()

for results_file in results_files:
	with open(results_file, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		head_row = next(csv_reader)
		workerid_index = head_row.index("WorkerId")
		time_taken_index = head_row.index("WorkTimeInSeconds")
		question_start_index = head_row.index("Input.question1")
		list_responses_start_index = head_row.index("Input.responses1")
		responses_start_index = head_row.index("Answer.response-selector1")
		reject_index = head_row.index("Reject")
		reject_time_index = head_row.index("RejectionTime")
		# TODO: remember that there is response selctor 1 then 10 then 2-9
		for row in csv_reader:
			# if row[reject_time_index].strip():
			if row[reject_time_index].strip() or row[workerid_index] in bad_workers:
				# ignore this annotation as its rejected
				continue 
			input_qs_and_as = row[question_start_index: question_start_index+20]
			responses = row[responses_start_index: responses_start_index+10]
			list_responses = row[list_responses_start_index: list_responses_start_index+10]
			for i in range(10):
				#ref: https://stackoverflow.com/questions/1894269/convert-string-representation-of-list-to-list
				list_responses[i] = ast.literal_eval("[" + list_responses[i] + "]")
			final_response = responses[1]
			del responses[1]
			responses.append(final_response)
			workerid = row[workerid_index]
			workers_and_time.append((workerid,int(row[time_taken_index])))
			worker_responses.setdefault(workerid, dict())
			# print(row)
			# print(len(head_row), len(row), row[reject_time_index])


			# reject = row[reject_index]
			# print(reject)
			i = 0
			# ref: https://stackoverflow.com/a/21752677/4535284
			for q, a, list_response in zip(input_qs_and_as[::2],input_qs_and_as[1::2], list_responses):
				grouped_responses.setdefault((q,a), list())
				# sort the list responses and find the index of the response 
				rank = index_of_the_response_in_the_list(responses[i], list_response)
				# print(rank)
				grouped_responses[(q,a)].append((responses[i], workerid, int(row[time_taken_index]), rank))
				# all_generated_responses[(q,a)] = list_response
				if (q,a) not in worker_responses[workerid]:
					worker_responses[workerid][(q,a)] = (responses[i], rank)
				else:
					print("ERROR: A worker should not have answered same question twice!")
				i += 1

def print_dict(d):
	for k, v in d.items():
		flag = False
		for (r, w, t) in v:
			if w == 'A1UI4LQGH05BII':
				flag = True
		if flag:
			print(k)
			for (r, w, t) in v:
				if w == 'A1UI4LQGH05BII':
					print(w, r)
				else:
					print(r)
			print()

def print_counter_dict(d):
	for k, v in d.items():
		print(k)
		for k2, v2 in v.items():
			print(v2, "\t", k2)
		print()

def convert_responses_to_counter_dict(d):
	counter_d = dict()
	for k, v in d.items():
		# counter_d[k] = Counter([(e[0], e[1]) for e in v])
		counter_d[k] = Counter([e[0] for e in v])
		# make every value as a tuple of count and list
		for k2, v2 in counter_d[k].items():
			counter_d[k][k2] = (v2, list(), list())
	# Add worker information in the counter dict
	for k, v in d.items():
		for r, w, t, rank in v:
			counter_d[k][r][1].append(w)
			counter_d[k][r][2].append(rank)
	return counter_d

print_counter_dict(convert_responses_to_counter_dict(grouped_responses))

# print_dict(grouped_responses)

def print_list(l):
	for e in l:
		print(e)

def get_histogram_for_workers(worker_responses, grouped_responses):
	# For each worker we need to check for each of his/her answered question whether it matches another worker's answer
	worker_time_similarity = list()
	worker_time_similarity_lookup = dict()
	for worker, worker_responses_dict in worker_responses.items():
		size = float(len(worker_responses_dict))
		matches = 0.0
		all_ts = list()
		all_ranks = list()
		for (q,a), (response, rank) in worker_responses_dict.items():
			all_q_a_responses = grouped_responses[(q,a)]
			# Count the number of times current response occured in all_q_a_responses. Should be more that 1
			count = 0
			for r, w, t, rank2 in all_q_a_responses:
				if w == worker:
					all_ts.append(t)
					all_ranks.append(rank2)
				if r == response:
					count += 1
			if count < 1:
				print("ERROR: bug in the code. At least current worker's answer should be in the list")
			if count > 1:
				matches += 1.0
		similarity_with_other_users = matches / size
		all_ranks_counter = Counter(all_ranks)
		worker_time_similarity.append((worker, size, similarity_with_other_users, all_ts[::10], all_ranks_counter[1]/float(len(all_ranks)), all_ranks_counter))
		worker_time_similarity_lookup[worker] = (size, similarity_with_other_users, all_ts[::10], all_ranks_counter[1]/float(len(all_ranks)), all_ranks_counter)
	return worker_time_similarity, worker_time_similarity_lookup

worker_time_similarity, worker_time_similarity_lookup = get_histogram_for_workers(worker_responses, grouped_responses)


def generate_train_data(counter_dict):
	failed_examples = 0
	success_examples = 0
	count_2_examples = 0
	unique_count_2_examples = set()
	train_data = list()
	for k, v in counter_dict.items():
		flag = False
		all_counts = 0
		for r, (count, workers, ranks) in v.items():
			all_counts += count
			worker_similarities = [worker_time_similarity_lookup[w][1] for w in workers]
			if count >= 1:
				flag = True
				train_data.append((k, r, workers, worker_similarities, count))
			if count >=2:
				count_2_examples += 1
				unique_count_2_examples.add(k)
		# if all_counts != 5:
		# 	print(k, all_counts)
		if not flag:
			failed_examples += 1
		else:
			success_examples += 1
	return train_data, failed_examples, success_examples, count_2_examples, len(unique_count_2_examples)

train_data, failed_examples, success_examples, count_2_examples, unique_count_2_examples = generate_train_data(convert_responses_to_counter_dict(grouped_responses))


print("Sorting by time:")
print("Worker - time taken in a particular hit")
print_list(sorted(workers_and_time, key=lambda x: x[1]))

print("\nSorting by similarity:")
print("Worker - number of questions - similarity with other users - time taken in the hits as list - rank score - rank counter")
print_list(sorted(worker_time_similarity, key=lambda x: x[2]))
similarities = [x[2] for x in worker_time_similarity]
number_of_questions = [x[1] for x in worker_time_similarity]
weighted_avg_similarity = np.average(similarities, weights=number_of_questions)
print("weighted average worker similarity:", weighted_avg_similarity)

print("\nSorting by rank_score:")
print("Worker - number of questions - similarity with other users - time taken in the hits as list - rank score - rank counter")
print_list(sorted(worker_time_similarity, key=lambda x: x[4], reverse=True))

n = 0
for w, n_qs, s, ts, rank_score, all_ranks_counter in sorted(worker_time_similarity, key=lambda x: x[2]):
	if s >= .35:
		n += n_qs
# print(n)

with open(save_train_data_file, "w") as writer:
	writer.write("Question\tAnswer\tResponse\tWorkers\tSimilarities\tCount\n")
	for (q,a), r, ws, sims, count in train_data:
		rule = None
		if r=="{}":
			#NOTE: Don't know how the worker marked this answer. It is not present in the list of available answers.
			continue
		# remove r from the list responses so that we don't include it in the negative samples
		# print(len(all_generated_responses[(q,a)]))
		try:
			responses, rules = all_generated_responses[(q,a)]
			index = responses.index(r)
			rule = rules[index]
			del responses[index]
			del rules[index]
		except ValueError:
			print("VALUE ERROR:!")
			print((q,a))
			print(r)
			print(all_generated_responses[(q,a)])
			exit()
		# print(len(all_generated_responses[(q,a)]))
		if not rule:
			print("RULE NOT FOUND!!")
			exit()
		writer.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(q,a,r,rule,ws,sims,count))
	already_explored_questions = set()
	for (q,a), r, ws, sims, count in train_data:
		if (q,a) in already_explored_questions:
			continue
		list_response, list_rule = all_generated_responses[(q,a)]
		for r, rule in zip(list_response, list_rule):
			writer.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(q,a,r,rule,list(),list(),0))
		already_explored_questions.add((q,a))

print(failed_examples, success_examples, len(train_data), count_2_examples, unique_count_2_examples)
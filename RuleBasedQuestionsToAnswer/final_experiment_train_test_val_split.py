# We want to split the generated features to a train set and a test set
import os
import csv
import re
import random
from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer()
random.seed(901)

DATA_DIR = "train_data"

# full_features_file = os.path.join(DATA_DIR, "squad_final_train_data_features_bad_tokenization.tsv")
full_features_file = os.path.join(DATA_DIR, "squad_final_train_data_features.tsv")
full_features_file = os.path.join(DATA_DIR, "squad_final_train_data_features_removed_short_response_affinity_workers.tsv")

print("Reading from:", full_features_file)
# read the features file and get all the (q,a) pairs first
all_qas = list()
# This will store the entire dataset with all the rows
all_rows = dict()
with open(full_features_file, "r") as features_file:
	head_line=next(features_file)
	print(head_line)
	for line in features_file:
		# get the row from the line
		line = line.strip()
		row = re.split('\t|\\t', line)
		q = row[0].strip()
		q = q.lower()
		a = row[1].strip()
		a = a.lower()
		row[0] = q
		row[1] = a
		if (q,a) not in all_qas:
			all_qas.append((q,a))
		all_rows.setdefault((q,a), list())
		all_rows[(q,a)].append(row)
# split the all_qas into 3 segments for 80% train qas, 10% val qas, 10% test qas
indices = list(range(len(all_qas)))
random.shuffle(indices)
test_qas = [all_qas[i] for i in indices[:700]]
val_qas = [all_qas[i] for i in indices[700:1000]]
train_qas = [all_qas[i] for i in indices[1000:]]
print(len(test_qas), len(val_qas), len(train_qas), len(all_qas))
print(indices[:700])
# print(test_qas)

NEGATIVE_PERCENTAGE = 100

train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
val_file = os.path.join(DATA_DIR, "val_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")

train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative_removed_short_response_affinity_workers.tsv".format(NEGATIVE_PERCENTAGE))
val_file = os.path.join(DATA_DIR, "val_count1_squad_final_train_data_features_removed_short_response_affinity_workers.tsv")
test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features_removed_short_response_affinity_workers.tsv")

train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
val_file = os.path.join(DATA_DIR, "val_count2_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count2_squad_final_train_data_features.tsv")

# Modified files where for test and val data only the shortest response should have count 2 others can have any count
train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_features.tsv")

def verify_if_a_acceptable_datapoint(qa, rows, shortest_response_index, train_qas):
	if qa in train_qas:
		# check if the rows have any row with count >=2
		for row in rows:
			count = int(row[-1])
			if count >= 2:
				return True
	else:
		# for test and validation only shortest response with min-count >= 2 rest of the responses with min. count >= 1
		for i, row in enumerate(rows):
			count = int(row[-1])
			if i == shortest_response_index:
				if count >= 2:
					return True
			else:
				if count >= 1:
					return True
	return False

def get_the_shortest_response_row_index(rows):
	index, row = min(enumerate(rows), key=lambda x: len(x[1][2].lower().strip().split()))
	# print(row[2])
	return index

explored = set()
# We will first load the entire dataset in memory and then save it
skipped_qas = 0
skipped_train_qas = 0
skipped_val_qas = 0
skipped_test_qas = 0

train_positive_response_count = 0
train_negative_response_count = 0
val_positive_response_count = 0
val_negative_response_count = 0
test_positive_response_count = 0
test_negative_response_count = 0
with open(train_file, "w") as train_f, open(val_file, "w") as val_f, open(test_file, "w") as test_f:
	train_writer = csv.writer(train_f, delimiter='\t')
	val_writer = csv.writer(val_f, delimiter='\t')
	test_writer = csv.writer(test_f, delimiter='\t')

	for qa, rows in all_rows.items():
		shortest_response_index = get_the_shortest_response_row_index(rows)
		# check if the rows have any row with count >=2
		if not verify_if_a_acceptable_datapoint(qa, rows, shortest_response_index, train_qas):
			skipped_qas += 1
			if qa in test_qas:
				skipped_test_qas += 1
			elif qa in val_qas:
				skipped_val_qas += 1
			elif qa in train_qas:
				skipped_train_qas += 1
			else:
				print("Error: Every qa has to be in either train, val or test")
				exit(1)
			continue

		if qa in test_qas:
			for row in rows:
				q = row[0].lower()
				a = row[1].lower()
				r = row[2].lower()
				count = int(row[-1])
				if (q,a,r) in explored:
					print("Already explored", (q,a,r))
					continue
				else:
					explored.add((q,a,r))
				if i == shortest_response_index:
					if count >= 2  or count == 0:
						if count>=2:
							test_positive_response_count += 1
						elif count == 0:
							test_negative_response_count += 1
						test_writer.writerow(row)
				elif count >= 0:
					if count > 0:
						test_positive_response_count += 1
					elif count == 0:
						test_negative_response_count += 1
					test_writer.writerow(row)
				elif count < 0:
					# This should not happen
					print("ERROR count is negative:", count)
					print(row)
					exit(1)
		elif qa in val_qas:
			for i, row in enumerate(rows):
				q = row[0].lower()
				a = row[1].lower()
				r = row[2].lower()
				count = int(row[-1])
				if (q,a,r) in explored:
					print("Already explored", (q,a,r))
					continue
				else:
					explored.add((q,a,r))
				if i == shortest_response_index:
					if count >= 2  or count == 0:
						if count>=2:
							val_positive_response_count += 1
						elif count == 0:
							val_negative_response_count += 1
						val_writer.writerow(row)
				elif count >= 0:
					if count > 0:
						val_positive_response_count += 1
					elif count == 0:
						val_negative_response_count += 1
					val_writer.writerow(row)
				elif count < 0:
					# This should not happen
					print("ERROR count is negative:", count)
					print(row)
					exit(1)
		else:
			for i, row in enumerate(rows):
				q = row[0].lower()
				a = row[1].lower()
				r = row[2].lower()
				if (q,a,r) in explored:
					print("Already explored", (q,a,r))
					continue
				else:
					explored.add((q,a,r))
				count = int(row[-1])
				# Train data
				if count >= 2:
					train_positive_response_count += 1
					train_writer.writerow(row)
				elif count == 0:
					prob = random.uniform(0, 1)
					if prob <= NEGATIVE_PERCENTAGE * 0.01:
						train_negative_response_count += 1
						train_writer.writerow(row)
				elif count < 0:
					# This should not happen
					print("ERROR count is negative:", count)
					print(row)
					exit(1)

print("Total Skipped QAs:", skipped_qas)
print("Total Skipped Train QAs:", skipped_train_qas)
print("Total Skipped Val QAs:", skipped_val_qas)
print("Total Skipped Test QAs:", skipped_test_qas)

print("train_positive:", train_positive_response_count)
print("train_negative:", train_negative_response_count)
print("val_positive:", val_positive_response_count)
print("val_negative:", val_negative_response_count)
print("test_positive:", test_positive_response_count)
print("test_negative:", test_negative_response_count)



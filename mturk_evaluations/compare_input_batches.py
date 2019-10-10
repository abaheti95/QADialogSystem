# We want to check if each batch has unique/different instances for annotations.
import csv

def load_all_batch_instances(batch_file):
	# read and load all the instances into a list
	with open(batch_file, "r") as b_csv:
		reader = csv.reader(b_csv, delimiter=',')
		all_annotations = set()
		for row in reader:
			# each row is a hit of size 10
			n = 5
			all_annotations |= set([tuple(row[i:i + n]) for i in range(0, len(row), n)])
	return all_annotations

def compare_batches(batch1_file, batch2_file):
	b1_annotations = load_all_batch_instances(batch1_file)
	b2_annotations = load_all_batch_instances(batch2_file)
	print(b1_annotations & b2_annotations)

batch1_file = "batch_input.csv"
batch2_file = "batch_input2.csv"
batch3_file = "batch_input3.csv"
batch4_file = "batch_input4.csv"

compare_batches(batch1_file, batch2_file)
compare_batches(batch1_file, batch3_file)
compare_batches(batch2_file, batch3_file)
compare_batches(batch4_file, batch1_file)
compare_batches(batch4_file, batch2_file)
compare_batches(batch4_file, batch3_file)
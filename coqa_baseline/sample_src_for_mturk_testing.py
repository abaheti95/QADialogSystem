import random
random.seed(901)

sample_indices = None
def create_sample(input_file, output_file):
	global sample_indices
	if not sample_indices:
		number_of_lines = 0
		with open(input_file, "r") as reader:
			for line in reader:
				number_of_lines += 1
		indices = list(range(number_of_lines))
		sample_indices = random.sample(indices, 100)
	with open(input_file, "r") as reader, open(output_file, "w") as writer:
		for i, line in enumerate(reader):
			if i in sample_indices:
				line = line.strip()
				writer.write("{}\n".format(line))

input_file = "src_coqa_dev_data_predictions_filtered.txt"
output_file = "src_coqa_dev_data_predictions_filtered_mturk_sample.txt"
create_sample(input_file, output_file)
input_file = "src_coqa_dev_data_filtered.txt"
output_file = "src_coqa_dev_data_filtered_mturk_sample.txt"
create_sample(input_file, output_file)

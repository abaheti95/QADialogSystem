# We have a 3000 final sample squad dataset
# We want to annotate this in batches of different size
# We will split the batches from starting to end based on the different batch sizes
import os

# ref: https://stackoverflow.com/a/15063941/4535284
import sys
import csv
csv.field_size_limit(sys.maxsize)

DATA_DIR = "mturk_experiments"
SAVE_DIR = os.path.join("mturk_experiments", "squad_final_3000_sample_batches")

data_file = os.path.join(DATA_DIR, "squad_questions_generated_final_3000_sample_input.csv")
batches = [(10, "squad_questions_generated_final_3000_sample_input_batch1_100.csv"), (40, "squad_questions_generated_final_3000_sample_input_batch2_400.csv"), (50, "squad_questions_generated_final_3000_sample_input_batch3_500.csv"), (50, "squad_questions_generated_final_3000_sample_input_batch4_500.csv"), (50, "squad_questions_generated_final_3000_sample_input_batch5_500.csv"), (50, "squad_questions_generated_final_3000_sample_input_batch6_500.csv"), (50, "squad_questions_generated_final_3000_sample_input_batch7_500.csv")]

#TODO: DEBUG code
# data_file = os.path.join(DATA_DIR, "squad_questions_generated_final_3000_sample_input_debug.csv")
# batches = [(10, "squad_questions_generated_final_3000_sample_input_batch1_100_debug.csv"), (90, "squad_questions_generated_final_3000_sample_input_batch2_900_debug.csv"), (100, "squad_questions_generated_final_3000_sample_input_batch3_1000_debug.csv"), (100, "squad_questions_generated_final_3000_sample_input_batch4_1000_debug.csv")]



with open(data_file, "r") as in_csv:
	reader = csv.reader(in_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	i = 0
	head_row = next(reader)
	batch_start_index = 0
	current_batch_size = batches[i][0]
	current_batch_file = open(os.path.join(SAVE_DIR, batches[i][1]), "w")
	current_batch_writer = csv.writer(current_batch_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	current_batch_writer.writerow(head_row[:len(head_row)-10])
	is_file_open = False
	for idx, row in enumerate(reader):
		if idx == batch_start_index + current_batch_size:
			# change the writer file
			i += 1
			current_batch_file.close()
			batch_start_index = idx
			current_batch_size = batches[i][0]
			current_batch_file = open(os.path.join(SAVE_DIR, batches[i][1]), "w")
			current_batch_writer = csv.writer(current_batch_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			current_batch_writer.writerow(head_row[:len(head_row)-10])
		current_batch_writer.writerow(row[:len(row)-10])
	current_batch_file.close()
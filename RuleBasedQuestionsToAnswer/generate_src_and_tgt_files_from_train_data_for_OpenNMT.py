# Here we will extract the questions and the list of responses from the train_data folder into src and tgt files which are compatible with OpenNMT
# We are doing this so that we can extract Embeddings for the Decomposable attention model with CoVe setup

import os
import re

def read_features_file(feature_file, src_save_file, tgt_save_file):
	with open(feature_file, 'r') as reader, open(src_save_file, "w") as s_writer, open(tgt_save_file, "w") as t_writer:
		print("Reading Generated Responses and questions instances from features file: %s", feature_file)
		for i, line in enumerate(reader):
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
			s_writer.write("{}\n".format(q))
			t_writer.write("{}\n".format(r))

# train file
DATA_FOLDER = "train_data"
NEGATIVE_PERCENTAGE = 100
train_file = os.path.join(DATA_FOLDER, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
val_file = os.path.join(DATA_FOLDER, "val_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_FOLDER, "test_count1_squad_final_train_data_features.tsv")
# New val and test files which are final
val_file = os.path.join(DATA_FOLDER, "val_shortest_count2_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_FOLDER, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)
SAVE_FOLDER = os.path.join("data", "CoVe")
make_dir_if_not_exists(SAVE_FOLDER)

src_save_file = os.path.join(SAVE_FOLDER, "s_{}_count{}_squad_final_train_data.txt")
tgt_save_file = os.path.join(SAVE_FOLDER, "t_{}_count{}_squad_final_train_data.txt")

# read_features_file(train_file, src_save_file.format("train", 2), tgt_save_file.format("train", 2))
read_features_file(val_file, src_save_file.format("val_shortest", 2), tgt_save_file.format("val_shortest", 2))
read_features_file(test_file, src_save_file.format("test_shortest", 2), tgt_save_file.format("test_shortest", 2))


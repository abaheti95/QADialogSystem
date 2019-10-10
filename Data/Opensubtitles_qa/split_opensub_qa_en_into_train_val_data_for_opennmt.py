# We want to split the moses tokenized opensub_qa_en files into 
# src and tgt files of train and val dataset for the OpenNMT py pretraining

import os
DATA_FOLDER = "opensub_qa_en"
train_tokenzied_file = os.path.join(DATA_FOLDER, "train_moses_tokenized.txt")
val_tokenzied_file = os.path.join(DATA_FOLDER, "valid_moses_tokenized.txt")

src_train_file = os.path.join(DATA_FOLDER, "src_train_moses_opensub_qa.txt")
tgt_train_file = os.path.join(DATA_FOLDER, "tgt_train_moses_opensub_qa.txt")
src_val_file = os.path.join(DATA_FOLDER, "src_val_moses_opensub_qa.txt")
tgt_val_file = os.path.join(DATA_FOLDER, "tgt_val_moses_opensub_qa.txt")

def split_tokenzied_data_file_to_src_and_tgt_files(data_file, src_save_file, tgt_save_file):
	print("Splitting {} into src {} and tgt {}".format(data_file, src_save_file, tgt_save_file
))
	with open(data_file, "r") as reader, open(src_save_file, "w") as s_writer, open(tgt_save_file, "w") as t_writer:
		for i, line in enumerate(reader):
			# if i == 10:
			# 	break
			src, tgt = line.strip().split("\t")
			# print(src)
			# print(tgt)
			# print()
			s_writer.write("{}\n".format(src))
			t_writer.write("{}\n".format(tgt))
split_tokenzied_data_file_to_src_and_tgt_files(train_tokenzied_file, src_train_file, tgt_train_file)
split_tokenzied_data_file_to_src_and_tgt_files(val_tokenzied_file, src_val_file, tgt_val_file)
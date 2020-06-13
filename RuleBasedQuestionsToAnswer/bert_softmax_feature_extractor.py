# source activate pytorch-bert
# First extract the embeddings using BERT
import torch
from transformers import *

import os
import csv
import random
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
from collections import Counter
import time

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)

DATA_DIR = "train_data"
RESULTS_DIR = os.path.join(DATA_DIR, "softmax_final_results")
make_dir_if_not_exists(RESULTS_DIR)

NEGATIVE_PERCENT = 100

train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENT))
# New shortest count2 data
val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_features.tsv")
# test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
# 									output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# load finetuned model here
model = BertForSequenceClassification.from_pretrained(os.path.join("transformers", "examples", "output"), output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(os.path.join("transformers", "examples", "output"))
# print("model loaded")
device = torch.device('cuda')
model.to(device)
model.eval()

def convert_data_file_to_bert_train_file(INPUT_file, OUTPUT_file):
	all_vectors = list()
	with open(INPUT_file, "r") as in_csv:
		reader = csv.reader(in_csv, delimiter="\t")
		start_time = time.time()
		for i, row in enumerate(reader):
			q,a,r = row[:3]
			count = row[-1]
			# print((q,a,r), count)
			q_ids = tokenizer.encode(q)
			r_ids = tokenizer.encode(r)
			input_ids = torch.tensor([tokenizer.add_special_tokens_sequence_pair(q_ids, r_ids)])
			input_ids = input_ids.to(device)
			# sub_start_time = time.time()
			last_hidden_layer, all_hidden_states = model(input_ids)
			# print("Time taken to get the hidden states:", time.time() - sub_start_time)
			CLS_vector = all_hidden_states[-1][0][0].tolist()
			all_vectors.append(CLS_vector)
			if ((i+1) % 1000) == 0:
				end_time = time.time()
				print("Time taken to do", (i+1), "pairs =", end_time - start_time, ", avg time:", ((end_time -start_time)/float(i+1)))
	all_vectors = np.array(all_vectors)
	np.save(OUTPUT_file, all_vectors)

# bert_softmax_train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_bert_features_{}_negative.npy".format(NEGATIVE_PERCENT))
# bert_softmax_val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_bert_features.npy")
# bert_softmax_test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_bert_features.npy")

# Embeddings from finetuned model
bert_softmax_train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_bert_finetuned_features_{}_negative.npy".format(NEGATIVE_PERCENT))
bert_softmax_val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_bert_finetuned_features.npy")
bert_softmax_test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_bert_finetuned_features.npy")


print("Processing Train file:", bert_softmax_train_file)
convert_data_file_to_bert_train_file(train_file, bert_softmax_train_file)
print("\n\nProcessing Val file:", bert_softmax_val_file)
convert_data_file_to_bert_train_file(val_file, bert_softmax_val_file)
print("\n\nProcessing Test file:", bert_softmax_test_file)
convert_data_file_to_bert_train_file(test_file, bert_softmax_test_file)


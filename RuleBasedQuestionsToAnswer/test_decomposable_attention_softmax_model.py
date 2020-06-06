"""
We will use Allennlp's pre-implemented version of "A Decomposable Attention Model for Natural Language Inference"
<https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_by Parikh et al., 2016

This is an SNLI model which can be also used for our task.
"""
import os
import re
import json
import time
# Allennlp uses typing for everything. We will need to annotate the type of every variable
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField
from allennlp.common import Params
# For every code that uses allennlp framework, we need to implement a dataset reader
# Essentially we will create a class that inherits the base class given by Allennlp, 
# which will read the data and produce an Instance iterator
from allennlp.data.dataset_readers import DatasetReader

# useful if your data is present in an URL
from allennlp.common.file_utils import cached_path

# We need to use these to index the sentences into allennlp recognizable list of tokens
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

# Will need vocabulary to convert sentences to tokens
from allennlp.data.vocabulary import Vocabulary

# this I beleive is just a wrapper around the pytorch module Model
from allennlp.models import Model
# Importing the model that we want to use
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum

# Useful for tracking accuracy on training and validation dataset
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from decomposable_attention_softmax_predictor import DecomposableAttentionSoftmaxPredictor
from decomposable_attention_softmax_model import DecomposableAttentionSoftmax
# from decomposable_attention_model_training import QuestionResponseReader

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder

from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from collections import Counter
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

def evaluate(scores, predictions, gold, gold_bucket_indices):
	acc = metrics.accuracy_score(gold, predictions)
	roc_auc = metrics.roc_auc_score(gold, scores)
	precision, recall, thresholds = metrics.precision_recall_curve(gold, scores, pos_label=1)
	f1s = [(2.0*p*r/(p+r), p,r) for p,r in zip(precision, recall)]
	f1_max, p_max, r_max = max(f1s, key=lambda tup: tup[0])
	print("MAX F1:", f1_max, p_max, r_max)
	pr_auc = metrics.auc(recall, precision)
	ap = metrics.average_precision_score(gold, scores)
	# MRR
	# divide the prediction scores and gold labels into buckets
	first_index = 0
	MRR = 0.0
	avg_pos = 0.0
	all_pos = list()
	for last_index in gold_bucket_indices:
		# print("gold", type(gold))
		# print("scores", type(scores))
		tuples = zip(list(range(first_index, last_index)), gold[first_index:last_index], scores[first_index:last_index])
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
	# exit()
	MRR /= float(len(gold_bucket_indices))
	avg_pos /= float(len(gold_bucket_indices))
	counter_all_pos = Counter(all_pos)
	precision_at_1 = counter_all_pos[1.0]/float(len(gold_bucket_indices))
	print(counter_all_pos)
	print("\n\nAVG POS:", avg_pos, "\n\n")
	cm = metrics.confusion_matrix(gold, predictions)
	classification_report = metrics.classification_report(gold, predictions, output_dict=True)
	classification_report_str = metrics.classification_report(gold, predictions)

	return acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str

def verify_and_generate_bucket_indices(df, last_column_index = 102):
	# we need to verify if the data is organized as blocks of questions and thier responses and not any other way
	explored_qa = set()
	current_qa = None
	current_1_flag = True			# if this flag is True then it means that we are expecting all count > 0. Once we encounter a zero count we will set it to False
	gold_bucket_indices = list()	# This is the list of indices at which the buckets end
	for index, row in df.iterrows():
		q, a, count = row[0], row[1], row[last_column_index]
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

class QuestionResponseSoftmaxReader(DatasetReader):
	def __init__(self,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 lazy: bool = False,
				 max_batch_size: int = 0) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
		self._max_batch_size = max_batch_size

	def update_max_batch_size(self, max_batch_size):
		self._max_batch_size = max_batch_size

	def _read(self, file_path: str):
		# if `file_path` is a URL, redirect to the cache
		file_path = cached_path(file_path)

		with open(file_path, 'r') as features_file:
			logger.info("Reading Generated Responses and questions instances from features file: %s", file_path)
			current_qa = None
			current_responses = list()
			current_labels = list()
			current_label_counts = 0
			for i, line in enumerate(features_file):
				# TODO: remove this after debugging
				# if i==10000:
				# 	break
				line = line.strip()
				row = re.split('\t|\\t', line)

				q = row[0].strip()
				q = q.lower()
				a = row[1].strip()
				a = a.lower()
				if current_qa != (q,a):
					# send the previous batch
					if len(current_responses) > 1:
						if current_label_counts > 0:
							if self._max_batch_size == 0:
								yield self.text_to_instance(current_qa[0], current_responses, current_labels)
							else:
								current_responses_and_labels = zip(current_responses, current_labels)
								current_positive_responses = [r_l for r_l in current_responses_and_labels if r_l[1] == "1"]
								# re-initialize the iterator so that we can run on in twice
								current_responses_and_labels = zip(current_responses, current_labels)
								current_negative_responses = [r_l for r_l in current_responses_and_labels if r_l[1] == "0"]
								# print("total responses", len(current_responses), "vs", len(current_labels))
								# print("positive responses", len(current_positive_responses))
								# print("negative responses", len(current_negative_responses))
								# shuffle negative_responses in place
								random.shuffle(current_negative_responses)
								# send all negative samples in batches with each having at least one positive response
								first_index = 0
								last_index = min(len(current_negative_responses), first_index + self._max_batch_size - 1)
								while True:
									current_sample = list()
									current_sample.append(random.choice(current_positive_responses))
									# print("first index", first_index)
									# print("last index", last_index)
									# sys.stdout.flush()
									current_sample.extend(current_negative_responses[first_index:last_index])
									# print("Current Sample size:", len(current_sample), " vs ", self._max_batch_size)
									# Get responses and labels list from current_sample
									current_sample_responses, current_sample_labels = [list(t) for t in zip(*current_sample)]
									yield self.text_to_instance(current_qa[0], current_sample_responses, current_sample_labels)
									if last_index == len(current_negative_responses):
										break
									# update first and last index
									first_index = last_index
									last_index = min(len(current_negative_responses), first_index + self._max_batch_size - 1)

					current_qa = (q,a)
					current_responses = list()
					current_labels = list()
					current_label_counts = 0
				r = row[2].strip()
				r = r.lower()
				rule = row[3].strip()
				count = row[-1]
				if int(count) > 0:
					label = "1"
					current_label_counts += 1
				else:
					label = "0"
				current_responses.append(r)
				current_labels.append(label)

			# yield the last batch
			if len(current_responses) > 1:
				if current_label_counts > 0:
					if self._max_batch_size == 0:
						yield self.text_to_instance(current_qa[0], current_responses, current_labels)
					else:
						current_responses_and_labels = zip(current_responses, current_labels)
						current_positive_responses = [r_l for r_l in current_responses_and_labels if r_l[1] == "1"]
						# re-initialize the iterator so that we can run on in twice
						current_responses_and_labels = zip(current_responses, current_labels)
						current_negative_responses = [r_l for r_l in current_responses_and_labels if r_l[1] == "0"]
						# print("total responses", len(current_responses), "vs", len(current_labels))
						# print("positive responses", len(current_positive_responses))
						# print("negative responses", len(current_negative_responses))
						# shuffle negative_responses in place
						random.shuffle(current_negative_responses)
						# send all negative samples in batches with each having at least one positive response
						first_index = 0
						last_index = min(len(current_negative_responses), first_index + self._max_batch_size - 1)
						while True:
							current_sample = list()
							current_sample.append(random.choice(current_positive_responses))
							# print("first index", first_index)
							# print("last index", last_index)
							# sys.stdout.flush()
							current_sample.extend(current_negative_responses[first_index:last_index])
							# print("Current Sample size:", len(current_sample), " vs ", self._max_batch_size)
							# Get responses and labels list from current_sample
							current_sample_responses, current_sample_labels = [list(t) for t in zip(*current_sample)]
							yield self.text_to_instance(current_qa[0], current_sample_responses, current_sample_labels)
							if last_index == len(current_negative_responses):
								break
							# update first and last index
							first_index = last_index
							last_index = min(len(current_negative_responses), first_index + self._max_batch_size - 1)
				current_qa = None
				current_responses = list()
				current_labels = list()
	
	def text_to_instance(self,  # type: ignore
						 premise: str,
						 hypotheses: List[str],
						 labels: List[str] = None) -> Instance:
		# pylint: disable=arguments-differ
		fields: Dict[str, Field] = {}
		premise_tokens = self._tokenizer.tokenize(premise)
		fields['premise'] = TextField(premise_tokens, self._token_indexers)
		all_hypotheses_fields = list()
		for hypothesis in hypotheses:
			hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
			all_hypotheses_fields.append(TextField(hypothesis_tokens, self._token_indexers))
		fields['hypotheses'] = ListField(all_hypotheses_fields)
		if labels:
			all_labels_fields = list()
			for label in labels:
				all_labels_fields.append(LabelField(label))
			fields['labels'] = ListField(all_labels_fields)
			metadata = {"labels": all_labels_fields}
			fields["metadata"] = MetadataField(metadata)
		return Instance(fields)

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)
# Since torch by default uses some amount of gpu 0 we will set it to 3 since 0 is not free
# export CUDA_VISIBLE_DEVICES=3

DATA_FOLDER = "train_data"
# EMBEDDING_TYPE = ""
# LOSS_TYPE = ""				# NLL
LOSS_TYPE = "_nll"				# NLL
# LOSS_TYPE = "_mse"			# MSE
# EMBEDDING_TYPE = ""
# EMBEDDING_TYPE = "_glove"
EMBEDDING_TYPE = "_bert"
# EMBEDDING_TYPE = "_elmo"
# EMBEDDING_TYPE = "_elmo_retrained"
# EMBEDDING_TYPE = "_elmo_retrained_2"
token_indexers = None
if EMBEDDING_TYPE == "_elmo" or EMBEDDING_TYPE == "_elmo_retrained" or EMBEDDING_TYPE == "_elmo_retrained_2":
	token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
MAX_BATCH_SIZE = 0
# MAX_BATCH_SIZE = 150 # for bert and elmo
reader = QuestionResponseSoftmaxReader(token_indexers=token_indexers, max_batch_size=MAX_BATCH_SIZE)
glove_embeddings_file = os.path.join("data", "glove", "glove.840B.300d.txt")
RESULTS_DIR = os.path.join(DATA_FOLDER, "decomposable_attention{}{}_softmax_results".format(LOSS_TYPE, EMBEDDING_TYPE))
make_dir_if_not_exists(RESULTS_DIR)
all_results_save_file = os.path.join(RESULTS_DIR, "all_decomposable_attention{}{}_softmax_results.txt".format(LOSS_TYPE, EMBEDDING_TYPE))
with open(all_results_save_file, "w") as all_writer:
	all_writer.write("NEGATIVE PERCENT\tAccuracy\troc_auc_score\tprecision recall auc\taverage precision\tMax F1 score\tPrecision for max F1\tRecall for max F1\tMRR\tPrecision@1\tConfusion Matrix\tClassification Report\n")
	print("Testing out model with", EMBEDDING_TYPE, "embeddings")
	print("Testing out model with", LOSS_TYPE, "loss")
	# for NEGATIVE_PERCENTAGE in [100,50,20,10,5,1]:
	for NEGATIVE_PERCENTAGE in [100]:
		train_file = os.path.join(DATA_FOLDER, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
		val_file = os.path.join(DATA_FOLDER, "val_count1_squad_final_train_data_features.tsv")
		test_file = os.path.join(DATA_FOLDER, "test_count1_squad_final_train_data_features.tsv")
		# New evalaution files with shortest response min count = 2
		val_file = os.path.join(DATA_FOLDER, "val_shortest_count2_squad_final_train_data_features.tsv")
		test_file = os.path.join(DATA_FOLDER, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")

		results_image = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_count1_{}_negative_squad_final.png")
		results_output = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_count1_{}_negative_squad_final.txt")
		results_predictions = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_count1_{}_negative_squad_final_predictions.txt")
		# New save files
		results_image = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_shortest_count2_{}_negative_squad_final.png")
		results_output = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_shortest_count2_{}_negative_squad_final.txt")
		results_predictions = os.path.join(RESULTS_DIR, "decomposable_atten{}{}_softmax_train_count2_test_shortest_count2_{}_negative_squad_final_predictions.txt")
		
		test_data = pd.read_csv(test_file, sep='\t', header=None)
		test_bucket_indices = verify_and_generate_bucket_indices(test_data, last_column_index=104)
		test_shortest_responses_labels = test_data[6].tolist()
		print("Shortest responses count:", sum(test_shortest_responses_labels))
		print("bucket indices len:", len(test_bucket_indices))

		model_file = os.path.join("saved_softmax_models", "decomposable_attention{}{}_model_{}.th".format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))

		vocabulary_filepath = os.path.join("saved_softmax_models","vocabulary{}{}_{}".format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))
		print("LOADING VOCABULARY")
		# Load vocabulary
		vocab = Vocabulary.from_files(vocabulary_filepath)

		EMBEDDING_DIM = 300
		PROJECT_DIM = 200
		DROPOUT = 0.2
		NUM_LAYERS = 2
		if EMBEDDING_TYPE == "":
			token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
									embedding_dim=EMBEDDING_DIM, projection_dim=PROJECT_DIM)
		elif EMBEDDING_TYPE == "_glove":
			token_embedding = Embedding.from_params(
								vocab=vocab,
								params=Params({'pretrained_file':glove_embeddings_file,
											   'embedding_dim' : EMBEDDING_DIM,
											   'projection_dim': PROJECT_DIM,
											   'trainable': False}))
		elif EMBEDDING_TYPE == "_elmo":
			# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
			# weights_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
			options_file = os.path.join("data", "elmo", "elmo_2x2048_256_2048cnn_1xhighway_options.json")
			weights_file = os.path.join("data", "elmo", "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5")
			# NOTE: using Small size as medium size gave CUDA out of memory error
			# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
			# weights_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
			# options_file = os.path.join("data", "elmo", "elmo_2x1024_128_2048cnn_1xhighway_options.json")
			# weights_file = os.path.join("data", "elmo", "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5")
			token_embedding = ElmoTokenEmbedder(options_file, weights_file, dropout=DROPOUT, projection_dim=PROJECT_DIM)
		elif EMBEDDING_TYPE == "_elmo_retrained":
			options_file = os.path.join("data", "bilm-tf", "elmo_retrained", "options.json")
			weights_file = os.path.join("data", "bilm-tf", "elmo_retrained", "weights.hdf5")
			token_embedding = ElmoTokenEmbedder(options_file, weights_file, dropout=DROPOUT, projection_dim=PROJECT_DIM)
		elif EMBEDDING_TYPE == "_elmo_retrained_2":
			options_file = os.path.join("data", "bilm-tf", "elmo_retrained", "options_2.json")
			weights_file = os.path.join("data", "bilm-tf", "elmo_retrained", "weights_2.hdf5")
			token_embedding = ElmoTokenEmbedder(options_file, weights_file, dropout=DROPOUT, projection_dim=PROJECT_DIM)
		elif EMBEDDING_TYPE == "_bert":
			print("Loading bert model")
			model = BertModel.from_pretrained('bert-base-uncased')
			token_embedding = BertEmbedder(model)
			PROJECT_DIM = 768
		else:
			print("Error: Some weird Embedding type", EMBEDDING_TYPE)
			exit()
		word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
		HIDDEN_DIM = 200
		params = Params({ 
			 'input_dim': PROJECT_DIM,
			 'hidden_dims': HIDDEN_DIM,
			 'activations': 'relu',
			 'num_layers': NUM_LAYERS,
			 'dropout': DROPOUT
			 })
		attend_feedforward = FeedForward.from_params(params)
		similarity_function = DotProductSimilarity()
		params = Params({ 
				 'input_dim': 2*PROJECT_DIM,
				 'hidden_dims': HIDDEN_DIM,
				 'activations': 'relu',
				 'num_layers': NUM_LAYERS,
				 'dropout': DROPOUT
				 })
		compare_feedforward = FeedForward.from_params(params)
		params = Params({ 
				 'input_dim': 2*HIDDEN_DIM,
				 'hidden_dims': 1,
				 'activations': 'linear',
				 'num_layers': 1
				 })
		aggregate_feedforward = FeedForward.from_params(params)
		model = DecomposableAttentionSoftmax(vocab, word_embeddings, attend_feedforward, similarity_function, compare_feedforward, aggregate_feedforward)
		print("MODEL CREATED")
		# Load model state
		with open(model_file, 'rb') as f:
		    model.load_state_dict(torch.load(f, map_location='cuda:0'))
		print("MODEL LOADED!")
		if torch.cuda.is_available():
			# cuda_device = 3
			# model = model.cuda(cuda_device)
			cuda_device = -1
		else:
			cuda_device = -1

		predictor = DecomposableAttentionSoftmaxPredictor(model, dataset_reader=reader)
		# Read test file and get predictions
		gold = list()
		predicted_labels = list()
		probs = list()
		print("Started Testing:", NEGATIVE_PERCENTAGE)
		# TODO: implement this later
		# if predictions file exists then load the predictions from that file
		if os.path.exists(results_predictions.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE)):
			print("LOADING Predictions from file:", results_predictions.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))
			# load the predictions from the file
			with open(results_predictions.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), "r") as predicitons_reader:
				for line in predicitons_reader:
					prediction, label_prob, gold_label = line.strip().split()
					prediction = int(prediction)
					label_prob = float(label_prob)
					gold_label = int(gold_label)
					# print(prediction, label_prob, gold_label)
					predicted_labels.append(prediction)
					probs.append(label_prob)
					gold.append(gold_label)

		else:
			with open(test_file, 'r') as features_file, open(results_predictions.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), "w") as predictions_writer:
				logger.info("Reading Generated Responses and questions instances from features file: %s", test_file)
				current_qa = None
				current_responses = list()
				current_labels = list()
				current_label_counts = 0
				example_no = 0
				start_time = time.time()
				SKIP_LINES = 0
				for i, line in enumerate(features_file):
					if i < SKIP_LINES:
						continue
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
					if current_qa != (q,a):
						# send the previous batch
						if len(current_responses) > 1:
							# print("Batch size:", len(current_responses))
							example_no += 1
							predictions = predictor.predict(current_qa[0], current_responses)
							# if example_no % 100 == 0:
							end_time = time.time()
							print("time taken till now {} secs. example: {}. of size {}. last i = {}".format(end_time - start_time, example_no, len(current_responses), i))
							label_probs = predictions["label_probs"]
							# Get predictions from label_probs
							gold.extend(current_labels)
							probs.extend(label_probs)
							current_predictions = [0]*len(current_labels)
							# Write in the predictions file
							for prediction, label, label_prob in zip(current_predictions, current_labels, label_probs):
								predictions_writer.write("{}\t{}\t{}\n".format(prediction, label_prob, label))
							label_probs = np.array(label_probs)
							current_predictions[np.argmax(label_probs)] = 1
							predicted_labels.extend(current_predictions)
						current_qa = (q,a)
						current_responses = list()
						current_labels = list()
					rule = row[3].strip()
					count = row[-1]
					if int(count) > 0:
						label = 1
					else:
						label = 0
					current_responses.append(r)
					current_labels.append(label)
				# Predict the last batch
				if len(current_responses) > 1:
					example_no += 1
					start_time = time.time()
					predictions = predictor.predict(current_qa[0], current_responses)
					end_time = time.time()
					print("time taken for {}th example is {} secs".format(example_no, end_time - start_time))

					label_probs = predictions["label_probs"]
					gold.extend(current_labels)
					probs.extend(label_probs)
					current_predictions = [0]*len(current_labels)
					label_probs = np.array(label_probs)
					current_predictions[np.argmax(label_probs)] = 1
					predicted_labels.extend(current_predictions)
					current_qa = None
					current_responses = list()
					current_labels = list()
					# print(count, predicted_label)
		print("Starting Evaluation:")
		acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = evaluate(probs, predicted_labels, gold, test_bucket_indices)
		plt.clf()
		step_kwargs = ({'step': 'post'}
						if 'step' in signature(plt.fill_between).parameters
						else {})
		plt.step(recall, precision, color='b', alpha=0.2,
					where='post')
		plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('PR curve: AP={0:0.2f} NEG%={1}%'.format(ap, NEGATIVE_PERCENTAGE))
		plt.savefig(results_image.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), dpi=300)

		# generate evaluation metrics
		with open(results_output.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), "w") as writer:

			writer.write("NEGATIVE PERCENTAGE{}\n".format(NEGATIVE_PERCENTAGE))
			print("NEGATIVE PERCENTAGE{}".format(NEGATIVE_PERCENTAGE))
			
			writer.write("Accuracy:{}\n".format(acc))
			print("Accuracy:{}".format(acc))
			writer.write("ROC_AUC_SCORE:{}\n".format(roc_auc))
			print("ROC_AUC_SCORE:{}".format(roc_auc))
			writer.write("PR_AUC_score:{}\n".format(pr_auc))
			print("PR_AUC_score:{}".format(pr_auc))
			writer.write("Average Precision Score:{}\n".format(ap))
			print("Average Precision Score:{}".format(ap))
			writer.write("Max F1:{}\n".format(f1_max))
			print("Max F1:{}".format(f1_max))
			writer.write("Precision for max F1:{}\n".format(p_max))
			print("Precision for max F1:{}".format(p_max))
			writer.write("Recall for max F1:{}\n".format(r_max))
			print("Recall for max F1:{}".format(r_max))
			writer.write("MRR:{}\n".format(MRR))
			print("MRR:{}".format(MRR))
			writer.write("Precision@1:{}\n".format(precision_at_1))
			print("Precision@1:{}".format(precision_at_1))

			writer.write("All Pos. Counter:\n{}\n".format(counter_all_pos))
			print("All Pos. Counter:\n{}".format(counter_all_pos))
			writer.write("CM:\n{}\n".format(cm))
			print("CM:\n{}".format(cm))
			writer.write("Classification report:\n{}\n".format(classification_report_str))
			print("Classification report:\n{}".format(classification_report_str))

			writer.write("Predictions:")
			
			# For each bucket we will print top 10 predictions
			first_index = 0
			counter = 0
			for last_index in test_bucket_indices:
				tuples = zip(list(range(first_index, last_index)), gold[first_index:last_index], probs[first_index:last_index], test_shortest_responses_labels[first_index:last_index])
				sorted_by_score = sorted(tuples, key=lambda tup: tup[2], reverse=True)
				l = [(index, gold_label,score, shortest_response_label) for index, gold_label,score, shortest_response_label in sorted_by_score if shortest_response_label == 1]
				if len(l) != 1:
					print("ERROR")
					print(l)
					for index, gold_label,score, shortest_response_label in l:
						print(test_data.iloc[index][0], test_data.iloc[index][2])
					print("\n")
					exit()
				index, gold_label,score, shortest_response_label = l[0]
				writer.write("Shortest response:{} -- {}\n".format(test_data.iloc[index][0], test_data.iloc[index][2]))
				count = 0
				for index, gold_label, score, shortest_response_label in sorted_by_score:
					writer.write("{}\t\t{}\t\t{}\t\t{}\t{}\t{}\n".format(test_data.iloc[index][0],
																	test_data.iloc[index][1],
																	test_data.iloc[index][2],
																	probs[index], predicted_labels[index],
																	gold[index]))
					assert(gold_label==gold[index])
					count += 1
					counter += 1
					if count == 10:
						break
				first_index = last_index
				if counter >= 5000:
					break


		# all_writer.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n{}\n{}\n{}\n{}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, MRR, precision_at_1, cm, classification_report_str, avg_pos, all_pos_dict))
		all_writer.write("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\n{10}\n{11}\n{12}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))

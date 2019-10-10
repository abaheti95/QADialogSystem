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

import multiprocessing

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from collections import Counter
import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sacremoses import MosesTokenizer
mt = MosesTokenizer()

class QuestionResponseSoftmaxReader(DatasetReader):
	def __init__(self,
				 q_file,
				 r_file,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 lazy: bool = False,
				 max_batch_size: int = 0) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
		self._max_batch_size = max_batch_size
		self._q_file = q_file
		self._r_file = r_file

	def update_max_batch_size(self, max_batch_size):
		self._max_batch_size = max_batch_size

	def _read(self, file_path: str):
		# if `file_path` is a URL, redirect to the cache
		q_file = cached_path(self._q_file)
		r_file = cached_path(self._r_file)

		with open(q_file, 'r') as q_reader, open(r_file, "r") as r_reader:
			logger.info("Reading questions from : %s", q_file)
			logger.info("Reading responses from : %s", r_file)
			q = next(q_reader).lower().strip()
			a = next(a_reader).lower().strip()
			current_qa = (q,a)
			current_responses = list()
			for i, response in enumerate(r_reader):
				response = response.strip()
				if response:
					print(response)
					exit()
					# Add it to the current responses
					current_responses.append(response)
				elif len(current_responses)	> 1:
					# Create a instance
					print(current_qa)
					print(current_responses)
					exit()
					yield self.text_to_instance(current_qa[0], current_responses)
					q = next(q_reader).lower().strip()
					a = next(a_reader).lower().strip()
					current_qa = (q,a)
					current_responses = list()
				else:
					# Serious Bug
					print("Serious BUG!!")
					print(current_qa)
					print(response)
					print(i)
					exit()
	
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

def get_answer_from_response(response):
	return response[response.find("{")+1:response.find("}")]

def remove_answer_brackets(response):
	return response.replace("{", "").replace("}", "").lower().strip()

def save_top_results(process_no, start_index, end_index):
	print("Starting process {} with start at {} and end at {}".format(process_no, start_index, end_index))
	DATA_FOLDER = "train_data"
	# EMBEDDING_TYPE = ""
	LOSS_TYPE = ""				# NLL
	LOSS_TYPE = "_mse"			# MSE
	# EMBEDDING_TYPE = ""
	# EMBEDDING_TYPE = "_glove"
	# EMBEDDING_TYPE = "_bert"
	EMBEDDING_TYPE = "_elmo"
	# EMBEDDING_TYPE = "_elmo_retrained"
	# EMBEDDING_TYPE = "_elmo_retrained_2"
	token_indexers = None
	if EMBEDDING_TYPE == "_elmo" or EMBEDDING_TYPE == "_elmo_retrained" or EMBEDDING_TYPE == "_elmo_retrained_2":
		token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
	MAX_BATCH_SIZE = 0
	# MAX_BATCH_SIZE = 150 # for bert and elmo
	# q_file = os.path.join("squad_seq2seq_train", "rule_based_system_squad_seq2seq_train_case_sensitive_saved_questions_lexparser_sh.txt")
	# r_file = os.path.join("squad_seq2seq_train", "rule_based_system_squad_seq2seq_train_case_sensitive_generated_answers_lexparser_sh.txt")
	# rules_file = os.path.join("squad_seq2seq_train", "rule_based_system_squad_seq2seq_train_case_sensitive_generated_answer_rules_lexparser_sh.txt")

	#NOTE: Squad dev test set
	q_file = os.path.join("squad_seq2seq_dev_moses_tokenized", "rule_based_system_squad_seq2seq_dev_test_saved_questions.txt")
	r_file = os.path.join("squad_seq2seq_dev_moses_tokenized", "rule_based_system_squad_seq2seq_dev_test_generated_answers.txt")
	rules_file = os.path.join("squad_seq2seq_dev_moses_tokenized", "rule_based_system_squad_seq2seq_dev_test_generated_answer_rules.txt")
	reader = QuestionResponseSoftmaxReader(q_file, r_file, token_indexers=token_indexers, max_batch_size=MAX_BATCH_SIZE)
	glove_embeddings_file = os.path.join("data", "glove", "glove.840B.300d.txt")
	# RESULTS_DIR = "squad_seq2seq_train2"
	#NOTE: All other experiments
	# RESULTS_DIR = "squad_seq2seq_train_moses_tokenized"
	# make_dir_if_not_exists(RESULTS_DIR)
	# all_results_save_file = os.path.join(RESULTS_DIR, "squad_seq2seq_train_predictions_start_{}_end_{}.txt".format(start_index, end_index))

	#NOTE: Squad dev test set
	RESULTS_DIR = "squad_seq2seq_dev_moses_tokenized"
	make_dir_if_not_exists(RESULTS_DIR)
	all_results_save_file = os.path.join(RESULTS_DIR, "squad_seq2seq_dev_test_predictions_start_{}_end_{}.txt".format(start_index, end_index))

	with open(all_results_save_file, "w") as all_writer:
		print("Testing out model with", EMBEDDING_TYPE, "embeddings")
		print("Testing out model with", LOSS_TYPE, "loss")
		# for NEGATIVE_PERCENTAGE in [100,50,20,10,5,1]:
		for NEGATIVE_PERCENTAGE in [100]:
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
				device = torch.device('cpu')
				model.load_state_dict(torch.load(f, map_location=device))
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
			total_time = avg_time = 0.0
			print("Started Testing:", NEGATIVE_PERCENTAGE)
			# before working on anything just save all the questions and responses in a list
			all_data = list()
			examples_count = processed_examples_count = 0
			with open(q_file, 'r') as q_reader, open(r_file, "r") as r_reader, open(rules_file, "r") as rule_reader:
				logger.info("Reading questions from : %s", q_file)
				logger.info("Reading responses from : %s", r_file)
				q = next(q_reader).lower().strip()
				q = mt.tokenize(q, return_str=True, escape=False)
				current_qa = (q, "")
				current_rules_and_responses = list()
				for i, (response, rule) in enumerate(zip(r_reader, rule_reader)):
					response = response.strip()
					rule = rule.strip()
					if response and rule:
						# get current_answer from response
						a = get_answer_from_response(response)
						if not current_qa[1]:
							current_qa = (q, a)
						else:
							# verify if the a is same as the one in current_qa
							if a != current_qa[1]:
								# print("answer phrase mismatch!!", current_qa, ":::", a, ":::", response)
								current_qa = (current_qa[0], a)
								# print(current_rules_and_responses)
								# exit()
						# Add it to the current responses
						current_rules_and_responses.append((response, rule))
					elif len(current_rules_and_responses) > 0:
						# Create a instance
						# print(current_qa)
						# print(current_rules_and_responses)
						# exit()
						if rule or response:
							print("Rule Response mismatch")
							print(current_qa)
							print(response)
							print(rule)
							print(examples_count)
							print(i)
							exit()

						if examples_count < start_index:
							examples_count += 1
							q = next(q_reader).lower().strip()
							q = mt.tokenize(q, return_str=True, escape=False)
							current_qa = (q, "")
							current_rules_and_responses = list()
							continue
						elif examples_count > end_index:
							break

						all_data.append((current_qa, current_rules_and_responses))
						try:
							q = next(q_reader).lower().strip()
							q = mt.tokenize(q, return_str=True, escape=False)
						except StopIteration:
							# previous one was the last question
							q = ""
						current_qa = (q, "")
						current_rules_and_responses = list()
						examples_count += 1
						# if(examples_count%100 == 0):
						# 	print(examples_count)
					else:
						# Serious Bug
						print("Serious BUG!!")
						print(current_qa)
						print(response)
						print(rule)
						print(examples_count)
						print(i)
						exit()
			print("{}:\tFINISHED IO".format(process_no))
			examples_count = start_index
			processed_examples_count = 0
			for current_qa, responses_and_rules in all_data:
				start_time = time.time()
				# Tokenize and preprocess the responses
				preprocessed_responses = [mt.tokenize(remove_answer_brackets(response), return_str=True, escape=False) for response, rule in responses_and_rules]
				# predictions = predictor.predict(current_qa[0], [remove_answer_brackets(response) for response, rule in responses_and_rules])
				predictions = predictor.predict(current_qa[0], preprocessed_responses)
				label_probs = predictions["label_probs"]
				tuples = zip(responses_and_rules, label_probs)
				sorted_by_score = sorted(tuples, key=lambda tup: tup[1], reverse=True)
				count = 0
				all_writer.write("{}\n".format(current_qa[0]))
				all_writer.write("{}\n".format(current_qa[1]))
				for index, ((response, rule), label_prob) in enumerate(sorted_by_score):
					if index == 3:
						break
					all_writer.write("{}\t{}\t{}\t{}\n".format(response, mt.tokenize(remove_answer_brackets(response), return_str=True, escape=False), rule, label_prob))
				all_writer.write("\n")
				all_writer.flush()
				end_time = time.time()
				processed_examples_count += 1
				examples_count += 1
				total_time += end_time - start_time
				avg_time = total_time / float(processed_examples_count)
				print("{}:\ttime to write {} with {} responses is {} secs. {} avg time".format(process_no, examples_count, len(responses_and_rules), end_time - start_time, avg_time))
			
# export CUDA_VISIBLE_DEVICES=3
if __name__ == '__main__':
	# Create 10 processes for different windows of questions and responses
	# get the number of lines in the questions file
	# NOTE: for all other experiments
	N_PROCESSES = 3
	q_file = os.path.join("squad_seq2seq_train", "rule_based_system_squad_seq2seq_train_case_sensitive_saved_questions_lexparser_sh.txt")
	# NOTE: for getting squad dev set ouptut
	N_PROCESSES = 1
	q_file = os.path.join("squad_seq2seq_dev_moses_tokenized", "rule_based_system_squad_seq2seq_dev_test_saved_questions.txt")
	
	num_lines = sum(1 for line in open(q_file))
	print("total questions:", num_lines)
	strides = int(num_lines/N_PROCESSES)
	print("Stride", strides)
	indices = [i for i in range(0, num_lines, strides)]
	indices[-1] = num_lines
	print(len(indices))
	start_indices = indices[0:N_PROCESSES]
	end_indices = indices[1:N_PROCESSES+1]
	if end_indices == []:
		start_indices = [0]
		end_indices = [num_lines]
	print(len(start_indices), start_indices)
	print(len(end_indices), end_indices)
	all_ps = list()
	for i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
		p = multiprocessing.Process(target=save_top_results, args=(i, start_index, end_index))
		p.start()
		all_ps.append(p)
	# join all ps
	for p in all_ps:
		p.join()

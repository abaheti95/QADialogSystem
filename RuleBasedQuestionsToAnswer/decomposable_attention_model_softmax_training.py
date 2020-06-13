"""
We will use Allennlp's pre-implemented version of "A Decomposable Attention Model for Natural Language Inference"
<https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_by Parikh et al., 2016

This is an SNLI model which can be also used for our task.
"""
import os
import re
import sys
# Allennlp uses typing for everything. We will need to annotate the type of every variable
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

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
from allennlp.models import DecomposableAttention
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum

# Useful for tracking accuracy on training and validation dataset
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors.decomposable_attention import DecomposableAttentionPredictor

from decomposable_attention_softmax_model import DecomposableAttentionSoftmax

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder

from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

import random

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--elmo", help="Flag to indicate if we want to use elmo embedding", action="store_true")
args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)

# def get_sampled_responses_and_labels(responses_and_labels, batch_size):
# 	correct_responses = [r_l for r_l in responses_and_labels if r_l[1] == "1"]
# 	incorrect_responses = [r_l for r_l in responses_and_labels if r_l[1] == "0"]
# 	sampled_incorrect_responses = random.sample(incorrect_responses, batch_size - len(correct_responses))
# 	final_sample = current_responses
# 	final_sample.extend(sampled_incorrect_responses)
# 	return [list(t) for t in zip(*final_sample)]

# we want to ouptut Fields similar to the SNLI reader
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


DATA_FOLDER = "train_data"
glove_embeddings_file = os.path.join("data", "glove", "glove.840B.300d.txt")
# model_save_filename = os.path.join("saved_softmax_models", "decomposable_attention_model_{}.th")
# vocab_save_filepath = os.path.join("saved_softmax_models","vocabulary_{}")
# model_save_filename = os.path.join("saved_softmax_models", "decomposable_attention_glove_model_{}.th")
# vocab_save_filepath = os.path.join("saved_softmax_models","vocabulary_glove_{}")

LOSS_TYPE = "_nll"
# LOSS_TYPE = "_mse"
NEGATIVE_PERCENTAGE = 10
# EMBEDDING_TYPE = ""
# EMBEDDING_TYPE = "_glove"
# EMBEDDING_TYPE = "_bert"
# EMBEDDING_TYPE = "_elmo"
# EMBEDDING_TYPE = "_elmo_retrained"
# EMBEDDING_TYPE = "_elmo_retrained_2"
# MAX_BATCH_SIZE = 0
MAX_BATCH_SIZE = 150 # for bert and elmo

if args.elmo:
	EMBEDDING_TYPE = "_elmo"
else:
	EMBEDDING_TYPE = ""

token_indexers = None
if EMBEDDING_TYPE == "_elmo" or EMBEDDING_TYPE == "_elmo_retrained" or EMBEDDING_TYPE == "_elmo_retrained_2":
	token_indexers = {"tokens": ELMoTokenCharactersIndexer()}

reader = QuestionResponseSoftmaxReader(token_indexers=token_indexers, max_batch_size=MAX_BATCH_SIZE)
model_save_filename = os.path.join("saved_softmax_models", "decomposable_attention{}{}_model_{}.th")
vocab_save_filepath = os.path.join("saved_softmax_models","vocabulary{}{}_{}")
# for NEGATIVE_PERCENTAGE in [1,5,10,20,50,100]:
for NEGATIVE_PERCENTAGE in [100]:
	print("Training with embedding type", EMBEDDING_TYPE, "embedddings")
	print("Training with loss type", LOSS_TYPE, "loss")
	print("Training for NEGATIVE_PERCENTAGE:", NEGATIVE_PERCENTAGE)
	train_file = os.path.join(DATA_FOLDER, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
	val_file = os.path.join(DATA_FOLDER, "val_count1_squad_final_train_data_features.tsv")
	test_file = os.path.join(DATA_FOLDER, "test_count1_squad_final_train_data_features.tsv")

	train_dataset = reader.read(train_file)
	val_dataset = reader.read(val_file)
	# test_dataset = reader.read(test_file)
	vocab = Vocabulary.from_instances(train_dataset + val_dataset)

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
	model = DecomposableAttentionSoftmax(vocab, word_embeddings, attend_feedforward, similarity_function, compare_feedforward, aggregate_feedforward, loss_type=LOSS_TYPE)
	if torch.cuda.is_available():
		# export CUDA_VISIBLE_DEVICES=3
		cuda_device = 0
		model = model.cuda(cuda_device)
	else:
		cuda_device = -1

	BATCH_SIZE = 1

	optimizer = optim.Adagrad(model.parameters(), lr=0.05, initial_accumulator_value=0.1)
	iterator = BucketIterator(batch_size=BATCH_SIZE, padding_noise=0.0, sorting_keys=[["premise", "num_tokens"]])
	# Iterator must make sure that the instances are indexed with vocabulary
	iterator.index_with(vocab)

	trainer = Trainer(model=model,
					  optimizer=optimizer,
					  iterator=iterator,
					  train_dataset=train_dataset,
					  validation_dataset=val_dataset,
					  patience=10,
					  num_epochs=30,
					  cuda_device=cuda_device)

	trainer.train()
	# Save model
	with open(model_save_filename.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), 'wb') as f:
		torch.save(model.state_dict(), f)
	# Save vocabulary
	vocab.save_to_files(vocab_save_filepath.format(LOSS_TYPE, EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))




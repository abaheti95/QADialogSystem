"""
We will use Allennlp's pre-implemented version of "A Decomposable Attention Model for Natural Language Inference"
<https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_by Parikh et al., 2016

This is an SNLI model which can be also used for our task.
"""
import os
import re
# Allennlp uses typing for everything. We will need to annotate the type of every variable
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
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

from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder

from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)

# we want to ouptut Fields similar to the SNLI reader
class QuestionResponseReader(DatasetReader):
	def __init__(self,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 lazy: bool = False) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		print("Provided token indexer was...", token_indexers)
		self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
		print("Current token indexer is....", self._token_indexers)

	def _read(self, file_path: str):
		# if `file_path` is a URL, redirect to the cache
		file_path = cached_path(file_path)

		with open(file_path, 'r') as features_file:
			logger.info("Reading Generated Responses and questions instances from features file: %s", file_path)
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
				r = row[2].strip()
				r = r.lower()
				rule = row[3].strip()
				count = row[-1]
				if int(count) > 0:
					label = "1"
				else:
					label = "0"
				
				premise = q
				hypothesis = r

				yield self.text_to_instance(premise, hypothesis, label)
	
	def text_to_instance(self,  # type: ignore
						 premise: str,
						 hypothesis: str,
						 label: str = None) -> Instance:
		# pylint: disable=arguments-differ
		fields: Dict[str, Field] = {}
		premise_tokens = self._tokenizer.tokenize(premise)
		hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
		fields['premise'] = TextField(premise_tokens, self._token_indexers)
		fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
		if label:
			fields['label'] = LabelField(label)

		metadata = {"premise_tokens": [x.text for x in premise_tokens],
					"hypothesis_tokens": [x.text for x in hypothesis_tokens]}
		fields["metadata"] = MetadataField(metadata)
		return Instance(fields)

DATA_FOLDER = "train_data"
LOSS_TYPE = ""
# LOSS_TYPE = "_mse"
NEGATIVE_PERCENTAGE = 10
# EMBEDDING_TYPE = ""
# EMBEDDING_TYPE = "_glove"
EMBEDDING_TYPE = "_bert"
# EMBEDDING_TYPE = "_elmo"
token_indexers = None
if EMBEDDING_TYPE == "_elmo":
	token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
reader = QuestionResponseReader(token_indexers=token_indexers)
glove_embeddings_file = os.path.join("data", "glove", "glove.840B.300d.txt")
# model_save_filename = os.path.join("saved_models", "decomposable_attention_model_{}.th")
# vocab_save_filepath = os.path.join("saved_models","vocabulary_{}")
model_save_filename = os.path.join("saved_models", "decomposable_attention{}_model_{}.th")
vocab_save_filepath = os.path.join("saved_models","vocabulary{}_{}")
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
			 'hidden_dims': 2,
			 'activations': 'linear',
			 'num_layers': 1
			 })
	aggregate_feedforward = FeedForward.from_params(params)
	model = DecomposableAttention(vocab, word_embeddings, attend_feedforward, similarity_function, compare_feedforward, aggregate_feedforward)
	if torch.cuda.is_available():
		cuda_device = 2
		model = model.cuda(cuda_device)
	else:
		cuda_device = -1

	# BATCH_SIZE = 128	# for ELMO
	BATCH_SIZE = 64		# for Bert

	optimizer = optim.Adagrad(model.parameters(), lr=0.05, initial_accumulator_value=0.1)
	iterator = BucketIterator(batch_size=BATCH_SIZE, padding_noise=0.0, sorting_keys=[["premise", "num_tokens"], ["hypothesis", "num_tokens"]])
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
	with open(model_save_filename.format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), 'wb') as f:
		torch.save(model.state_dict(), f)
	# Save vocabulary
	vocab.save_to_files(vocab_save_filepath.format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))




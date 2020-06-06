# export CUDA_VISIBLE_DEVICES=1
# source activate pytorch-bert
# we will get scores from multiple models on the test examples and then combine them

## Loading the bert model
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import csv
import time

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

import sys
sys.path.append(os.path.join("..", "pytorch-transformers", "examples"))

from utils_glue import (compute_metrics, convert_examples_to_features,
						output_modes, DataProcessor, InputExample)

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from collections import Counter

# Allennlp uses typing for everything. We will need to annotate the type of every variable
from typing import Iterator, List, Dict

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

# from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder

from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


per_gpu_eval_batch_size = 32

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)

class RuleResponsesProcessor(DataProcessor):
	"""Processor for the Rule Based Responses reranking data set (GLUE version)."""

	def get_train_examples(self, train_file):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(train_file), "train")

	def get_dev_examples(self, dev_file):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(dev_file), "dev")

	def get_test_examples(self, test_file):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(test_file), "test")

	def get_examples(self, list_examples, set_type, LABELS=False):
		"""See base class."""
		return self._create_examples(
			list_examples, set_type, LABELS)

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type, LABELS=False):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			# if i == 100:
			# 	break
			guid = "%s-%s" % (set_type, str(i))
			text_q = line[0]    # question
			text_r = line[2]    # response
			if LABELS:
				label = int(line[-1])    # label
				label = "1" if label > 0 else "0"
				examples.append(
				InputExample(guid=guid, text_a=text_q, text_b=text_r, label=label))
			else:
				examples.append(
				InputExample(guid=guid, text_a=text_q, text_b=text_r))
		return examples

def prepare_bert_model(model_name_or_path):
	processor = RuleResponsesProcessor()
	label_list = processor.get_labels()
	num_labels = len(label_list)
	output_mode = "classification"
	model_type = "bert"
	config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
	config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task="rule")
	tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
	model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	return model, tokenizer, config, processor, device

def load_examples(processor, list_examples, tokenizer, LABELS=False):
	output_mode = "classification"
	# Load data features from the list of provided examples
	label_list = processor.get_labels()
	examples = processor.get_examples(list_examples, "test", LABELS)
	max_seq_length = 128
	model_type = "bert"
	features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
		cls_token_at_end=bool(model_type in ['xlnet']),            # xlnet has a cls token at the end
		cls_token=tokenizer.cls_token,
		cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
		sep_token=tokenizer.sep_token,
		sep_token_extra=bool(model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
		pad_on_left=bool(model_type in ['xlnet']),                 # pad on the left for xlnet
		pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
		pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
	)

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	if LABELS:
		if output_mode == "classification":
			all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
		elif output_mode == "regression":
			all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	else:
		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
	return dataset

def get_bert_model_predictions(list_examples, output_dir, device, model, tokenizer, processor, LABELS=False):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("rule",)
	eval_outputs_dirs = (output_dir,)

	results = {}

	eval_dataset = load_examples(processor, list_examples, tokenizer, LABELS)
	 
	eval_batch_size = per_gpu_eval_batch_size
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

	# Eval!
	eval_loss = 0.0
	nb_eval_steps = 0
	preds = None
	out_label_ids = None
	model_type = "bert"
	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		batch = tuple(t.to(device) for t in batch)

		with torch.no_grad():
			inputs = {'input_ids':      batch[0],
					  'attention_mask': batch[1],
					  'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None}  # XLM and RoBERTa don't use segment_ids
			if LABELS:
				inputs.update({'labels':batch[3]})
			outputs = model(**inputs)
			tmp_eval_loss, logits = outputs[:2]
			# perform a softmax on logits

			eval_loss += tmp_eval_loss.mean().item()
		nb_eval_steps += 1
		if preds is None:
			preds = softmax(logits.detach().cpu().numpy(), axis=1)
			# print(preds)
			# print(softmax(preds, axis=1))
			# exit()
			if LABELS:
				out_label_ids = inputs['labels'].detach().cpu().numpy()
		else:
			preds = np.append(preds, softmax(logits.detach().cpu().numpy(), axis=1), axis=0)
			if LABELS:
				out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

	eval_loss = eval_loss / nb_eval_steps
	scores = preds[:,1]
	predicted_labels = [1 if s > 0.5 else 0 for s in scores]
	# remove the log by taking exponent
	# scores = np.exp(scores)
	if LABELS:
		return scores, predicted_labels, out_label_ids
	return scores, predicted_labels

# extract results from decomposable attention softmax + elmo model

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

def load_decomposable_attention_elmo_softmax_model():
	NEGATIVE_PERCENTAGE = 100
	# EMBEDDING_TYPE = ""
	# LOSS_TYPE = ""				# NLL
	# LOSS_TYPE = "_nll"				# NLL
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
	reader = QuestionResponseSoftmaxReader(token_indexers=token_indexers, max_batch_size=MAX_BATCH_SIZE)
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
	return model, predictor

def get_predictions_from_decomp_elmo_model(list_examples, model, predictor, LABELS=False):
	gold = list()
	predicted_labels = list()
	probs = list()

	current_qa = None
	current_responses = list()
	if LABELS:
		current_labels = list()
		current_label_counts = 0
	example_no = 0
	start_time = time.time()
	SKIP_LINES = 0
	for i, line in enumerate(list_examples):
		if i < SKIP_LINES:
			continue
		# if i == 100:
		# 	break
		q = line[0].strip()
		q = q.lower()
		a = line[1].strip()
		a = a.lower()
		r = line[2].strip()
		r = r.lower()
		if current_qa != (q,a):
			# send the previous batch
			if len(current_responses) > 0:
				# print("Batch size:", len(current_responses))
				example_no += 1
				predictions = predictor.predict(current_qa[0], current_responses)
				# if example_no % 100 == 0:
				end_time = time.time()
				print("time taken till now {} secs. example: {}. of size {}. last i = {}".format(end_time - start_time, example_no, len(current_responses), i))
				label_probs = predictions["label_probs"]
				# label_probs = [0]* len(current_responses)
				# Get predictions from label_probs
				probs.extend(label_probs)
				current_predictions = [0]*len(label_probs)
				if LABELS:
					gold.extend(current_labels)
					# print(len(probs), lnoen(gold))

				label_probs = np.array(label_probs)
				current_predictions[np.argmax(label_probs)] = 1
				predicted_labels.extend(current_predictions)
			current_qa = (q,a)
			current_responses = list()
			if LABELS:
				current_labels = list()
		rule = line[3].strip()
		if LABELS:
			count = line[-1]
			if int(count) > 0:
				label = 1
			else:
				label = 0
			current_labels.append(label)
		current_responses.append(r)
	# Predict the last batch
	if len(current_responses) > 0:
		example_no += 1
		start_time = time.time()
		predictions = predictor.predict(current_qa[0], current_responses)
		end_time = time.time()
		print("time taken till now {} secs. example: {}. of size {}. last i = {}".format(end_time - start_time, example_no, len(current_responses), i))

		label_probs = predictions["label_probs"]
		# label_probs = [0]*len(current_responses)
		probs.extend(label_probs)
		if LABELS:
			gold.extend(current_labels)
			# print(len(probs), len(gold))
		current_predictions = [0]*len(label_probs)
		label_probs = np.array(label_probs)
		current_predictions[np.argmax(label_probs)] = 1
		predicted_labels.extend(current_predictions)

		current_qa = None
		current_responses = list()
		if LABELS:
			current_labels = list()
	if LABELS:
		return probs, predicted_labels, gold
	else:
		return probs, predicted_labels

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

def my_evaluate(scores, predictions, gold, gold_bucket_indices):
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
		#   print(val_data.iloc[index])
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

def normalize_scores(scores, gold_bucket_indices):
	# Based on the gold_bucket_indices we want to normalize the scores
	first_index = 0
	new_scores = list()
	for last_index in gold_bucket_indices:
		current_bucket_scores = scores[first_index:last_index]
		total_sum_of_current_bucket_scores = sum(current_bucket_scores)
		if total_sum_of_current_bucket_scores == 1.0:
			print("Yeah baby!!")
		# normalize current scores based on total scores and extend the new_scores
		normalized_current_bucket_scores = [s/total_sum_of_current_bucket_scores for s in current_bucket_scores]
		new_scores.extend(normalized_current_bucket_scores)
		first_index = last_index
	return new_scores

def get_predicted_labels(scores, gold_bucket_indices):
	first_index = 0
	predicted_labels = list()
	for last_index in gold_bucket_indices:
		current_bucket_scores = np.array(scores[first_index:last_index])
		current_predictions = [0]*current_bucket_scores.shape[0]
		current_predictions[np.argmax(current_bucket_scores)] = 1
		predicted_labels.extend(current_predictions)
		first_index = last_index
	return predicted_labels

def combine_scores_and_compute_predicted_labels(list_scores, gold_bucket_indices):
	total_normalized_scores = None
	for i, scores in enumerate(list_scores):
		scores = np.array(normalize_scores(scores, gold_bucket_indices))
		if i == 0:
			total_normalized_scores = scores
		else:
			total_normalized_scores += scores
	# Now sum the scores and average them based on number of scores provided in the list_scores
	total_normalized_scores /= float(len(list_scores))
	# get predicted_labels from scores
	predicted_labels = get_predicted_labels(total_normalized_scores, gold_bucket_indices)
	return total_normalized_scores, predicted_labels

DATA_FOLDER = "train_data"
test_file = os.path.join(DATA_FOLDER, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")
RESULTS_FOLDER = os.path.join(DATA_FOLDER, "ensemble_results")
make_dir_if_not_exists(RESULTS_FOLDER)
results_file = os.path.join(RESULTS_FOLDER, "bert_finetune_and_decomp_elmo_softmax_ensemble_results.txt")
# Load bert model
bert_model_path = os.path.join("..", "pytorch-transformers", "examples", "output")
bert_model, bert_tokenizer, bert_config, bert_processor, device = prepare_bert_model(bert_model_path)
# Load decomposable attention + elmo softmax model
decomp_model, decomp_predictor = load_decomposable_attention_elmo_softmax_model()
# Evaluate both models on standard test file
with open(test_file, "r") as in_csv, open(results_file, "w") as writer:
	test_data = pd.read_csv(test_file, sep='\t', header=None)
	test_bucket_indices = verify_and_generate_bucket_indices(test_data, last_column_index=104)
	print("Generated test bucket indices!", len(test_bucket_indices))
	reader = csv.reader(in_csv, delimiter='\t')
	# get bert predictions
	bert_scores, bert_predicted_labels, gold_labels = get_bert_model_predictions(reader, bert_model_path, device, bert_model, bert_tokenizer, bert_processor, LABELS=True)
	print("Got BERT predictions on test!")
	print("bert scores size:", bert_scores.shape)
	print("Gold labels size:", len(gold_labels))
	in_csv.seek(0,0)
	# get decomp. + elmo predictions
	decomp_scores, decomp_predicted_labels, gold_labels2 = get_predictions_from_decomp_elmo_model(reader, decomp_model, decomp_predictor, LABELS=True)
	print("Got Decomp. + Elmo Softmax predictions on test!")
	print("Probs:", len(decomp_scores))
	# gold_labels is numpy array and gold_labels2 is list so they won't be equal
	# if gold_labels == gold_labels2:
	# 	print("BOTH gold labels are same")
	# else:
	# 	print("Error in gold labels! Please debug")

	# Get bucket indices before evaluate
	# Evaluate bert model
	acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = my_evaluate(bert_scores, bert_predicted_labels, gold_labels2, test_bucket_indices)
	NEGATIVE_PERCENTAGE = 100
	writer.write("BERT Results:\n")
	writer.write("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\n{10}\n{11}\n{12}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
	print("BERT evaluation done!")
	# Evaluate decomp. + elmo model
	acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = my_evaluate(decomp_scores, decomp_predicted_labels, gold_labels2, test_bucket_indices)

	writer.write("\n\nDecomposable Attention + Elmo Softmax Results:\n")
	writer.write("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\n{10}\n{11}\n{12}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
	print("Decomp. + Elmo softmax evaluation done!")
	# Combine scores and compute predicted labels
	ensemble_scores, ensemble_predicted_labels = combine_scores_and_compute_predicted_labels([bert_scores, decomp_scores], test_bucket_indices)
	# Evaluate ensemble model
	acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = my_evaluate(ensemble_scores, ensemble_predicted_labels, gold_labels2, test_bucket_indices)

	writer.write("\n\nEnsemble Results:\n")
	writer.write("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\n{10}\n{11}\n{12}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
	print("Ensemble evaluation done!")






















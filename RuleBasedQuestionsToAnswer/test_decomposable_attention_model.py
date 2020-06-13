"""
We will use Allennlp's pre-implemented version of "A Decomposable Attention Model for Natural Language Inference"
<https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_by Parikh et al., 2016

This is an SNLI model which can be also used for our task.
"""
import os
import re
import json
# Allennlp uses typing for everything. We will need to annotate the type of every variable
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np
import pandas as pd

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

# from decomposable_attention_model_training import QuestionResponseReader

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

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--elmo", help="Flag to indicate if we want to use elmo embedding", action="store_true")
args = parser.parse_args()


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

# we want to ouptut Fields similar to the SNLI reader
class QuestionResponseReader(DatasetReader):
	def __init__(self,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 lazy: bool = False) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

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
SAVED_MODEL_DIR = "saved_models"
LOSS_TYPE = ""
NEGATIVE_PERCENTAGE = 10

if args.elmo:
	EMBEDDING_TYPE = "_elmo"
else:
	EMBEDDING_TYPE = ""

token_indexers = None
if EMBEDDING_TYPE == "_elmo":
	token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
reader = QuestionResponseReader(token_indexers=token_indexers)

glove_embeddings_file = os.path.join("data", "glove", "glove.840B.300d.txt")
def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)
RESULTS_DIR = os.path.join(DATA_FOLDER, "decomposable_attention{}_logits_results".format(EMBEDDING_TYPE))
make_dir_if_not_exists(RESULTS_DIR)

NEGATIVE_PERCENTAGE = 10
all_results_save_file = os.path.join(RESULTS_DIR, "all_decomposable_attention_results.txt")
results_image = os.path.join(RESULTS_DIR, "decomposable_atten{}_train_count2_test_count1_{}_negative_squad_final.png")
results_output = os.path.join(RESULTS_DIR, "decomposable_atten{}_train_count2_test_count1_{}_negative_squad_final.txt")

with open(all_results_save_file, "w") as all_writer:
	all_writer.write("NEGATIVE PERCENT\tAccuracy\troc_auc_score\tprecision recall auc\taverage precision\tMax F1 score\tMRR\tConfusion Matrix\tClassification Report\n")
	print("Testing out model with", EMBEDDING_TYPE, "embeddings")
	# for NEGATIVE_PERCENTAGE in [1,5,10,20,50,100]:
	for NEGATIVE_PERCENTAGE in [100]:
		train_file = os.path.join(DATA_FOLDER, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
		val_file = os.path.join(DATA_FOLDER, "val_count1_squad_final_train_data_features.tsv")
		test_file = os.path.join(DATA_FOLDER, "test_count1_squad_final_train_data_features.tsv")
		# New evalaution files with shortest response min count = 2
		val_file = os.path.join(DATA_FOLDER, "val_shortest_count2_squad_final_train_data_features.tsv")
		test_file = os.path.join(DATA_FOLDER, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")

		
		test_data = pd.read_csv(test_file, sep='\t', header=None)
		test_bucket_indices = verify_and_generate_bucket_indices(test_data, last_column_index=104)
		test_shortest_responses_labels = test_data[6].tolist()
		print("Shortest responses count:", sum(test_shortest_responses_labels))
		print("bucket indices len:", len(test_bucket_indices))

		model_file = os.path.join(SAVED_MODEL_DIR, "decomposable_attention{}_model_{}.th".format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))
		vocabulary_filepath = os.path.join(SAVED_MODEL_DIR,"vocabulary{}_{}".format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE))

		print("LOADING VOCABULARY")
		# Load vocabulary
		vocab = Vocabulary.from_files(vocabulary_filepath)
		print("VOCABULARY LOADED")
		# Create model
		EMBEDDING_DIM = 300
		PROJECT_DIM = 200
		DROPOUT = 0.2
		NUM_LAYERS = 2
		token_embedding = None
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
				 'input_dim': 2*HIDDEN_DIM,
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
		print("CREATING MODEL")
		model = DecomposableAttention(vocab, word_embeddings, attend_feedforward, similarity_function, compare_feedforward, aggregate_feedforward)
		print("MODEL CREATED")
		# Load model state
		with open(model_file, 'rb') as f:
			model.load_state_dict(torch.load(f, map_location='cuda:0'))
		print("MODEL LOADED!")
		if torch.cuda.is_available():
			# export CUDA_VISIBLE_DEVICES=3
			cuda_device = 0
			model = model.cuda(cuda_device)
		else:
			cuda_device = -1

		predictor = DecomposableAttentionPredictor(model, dataset_reader=reader)
		# Read test file and get predictions
		gold = list()
		predicted_labels = list()
		probs = list()
		print("Started Testing:", NEGATIVE_PERCENTAGE)
		with open(test_file, 'r') as features_file:
			logger.info("Reading Generated Responses and questions instances from features file: %s", test_file)
			for i, line in enumerate(features_file):
				if i%10000==0:
					print(i)
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
					label = 1
				else:
					label = 0
				gold.append(label)
				premise = q
				hypothesis = r
				predictions = predictor.predict(premise, hypothesis)
				# print(json.dumps(predictions, indent=4))
				# print(premise, hypothesis)
				label_probs = predictions["label_probs"]
				predicted_label = 0
				if label_probs[1] > label_probs[0]:
					predicted_label = 1
				predicted_labels.append(predicted_label)
				probs.append(label_probs[1])
				# print(count, predicted_label)
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
		plt.savefig(results_image.format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), dpi=300)

		# generate evaluation metrics
		with open(results_output.format(EMBEDDING_TYPE, NEGATIVE_PERCENTAGE), "w") as writer:

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


		# all_writer.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n{}\n{}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, MRR, cm, classification_report_str))
		all_writer.write("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\n{10}\n{11}\n{12}\n\n".format(NEGATIVE_PERCENTAGE, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))

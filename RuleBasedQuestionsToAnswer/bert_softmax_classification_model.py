# We will extend BertForSequenceClassificaiton model from transformers package to handle softmax function
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from random import shuffle
import csv
import time

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
	from torch.utils.tensorboard import SummaryWriter
except:
	from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertModel, BertConfig, BertPreTrainedModel,
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

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample

from scipy.special import softmax
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from collections import Counter
from transformers import glue_compute_metrics as compute_metrics

logger = logging.getLogger(__name__)

class BertSoftmaxForSequenceClassification(BertPreTrainedModel):
	r"""
		**labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
			Labels for computing the sequence classification/regression loss.
			Indices should be in ``[0, ..., config.num_labels - 1]``.
			If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
			If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
	Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
		**loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
			Classification (or regression if config.num_labels==1) loss.
		**logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		**hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
			list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
			of shape ``(batch_size, sequence_length, hidden_size)``:
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		**attentions**: (`optional`, returned when ``config.output_attentions=True``)
			list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
	Examples::
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertSoftmaxForSequenceClassification.from_pretrained('bert-base-uncased')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, logits = outputs[:2]
	"""
	def __init__(self, config):
		super(BertSoftmaxForSequenceClassification, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		# self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
		self.classifier = nn.Linear(config.hidden_size, 1)

		self.init_weights()

	def forward(self, input_ids, attention_mask=None, token_type_ids=None,
				position_ids=None, head_mask=None, labels=None):

		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids, 
							head_mask=head_mask)
		# print("CLS embedding shape:", outputs[0].shape)
		# print("Input Ids shape:", input_ids.shape)
		# print("Attention mask shape:", attention_mask.shape)
		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		logits_squeezed = logits.squeeze()
		log_logits_squeezed = nn.functional.log_softmax(logits_squeezed, dim=0)
		# print("Pooled output shape:", pooled_output.shape)
		# print("logits shape:", logits.shape)
		# print("labels:", labels)
		# print("logits:", logits_squeezed)
		# print("softmax logits:", log_logits_squeezed)
		# print("type of log softmax:", log_logits_squeezed.type())

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = -torch.dot(log_logits_squeezed, labels.float())
			outputs = (loss,) + outputs
		# print("loss:", loss)
		# exit()
		return outputs


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
	'bert': (BertConfig, BertSoftmaxForSequenceClassification, BertTokenizer)
}

def load_and_cache_examples(args, processor, tokenizer, data_file, bucket_indices, tag, evaluate=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	output_mode = "classification"
	# Load data features from cache or dataset file
	cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
		'dev' if evaluate else 'train',
		list(filter(None, args.model_name_or_path.split('/'))).pop(),
		str(args.max_seq_length), tag))
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", args.data_dir)
		label_list = processor.get_labels()
		examples = processor.get_examples(data_file, "train", LABELS=True)
		features = convert_examples_to_features(examples,
												tokenizer,
												label_list=label_list,
												max_length=args.max_seq_length,
												output_mode=output_mode,
												pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
												pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
												pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
		)
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save(features, cached_features_file)

	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	# Convert to Tensors and build dataset
	# Create striding negative sample with args.per_gpu_train_batch_size
	batch_size = args.per_gpu_train_batch_size
	first_index = 0
	total_batches = 0
	ignored_examples = 0
	total_ingored_instances = 0
	random.seed(args.seed)
	extended_features = list()

	if not evaluate:
		for last_index in bucket_indices:
			if last_index - first_index < batch_size/2+2:
				ignored_examples += 1
				total_ingored_instances += last_index - first_index
				first_index = last_index
				continue
				#TODO: although non-ideal but we will ignore such cases
			bucket_features = features[first_index:last_index]
			# Find the correct labels
			correct_instances = [f for f in bucket_features if f.label == 1]
			incorrect_instances = [f for f in bucket_features if f.label == 0]
			# shuffle incorrect instances
			shuffle(incorrect_instances)
			first_sub_index = 0
			last_sub_index = min(len(incorrect_instances), first_sub_index + batch_size - 1)
			while True:
				total_batches += 1
				extended_features.append(random.choice(correct_instances))
				# print("first index", first_sub_index)
				# print("last index", last_sub_index)
				# sys.stdout.flush()
				extended_features.extend(incorrect_instances[first_sub_index:last_sub_index])
				if last_sub_index - first_sub_index < (batch_size-1):
					print("Ayayayaya", len(incorrect_instances[first_sub_index:last_sub_index]), batch_size - 1 - last_sub_index + first_sub_index, batch_size)
					# add the remaining negative samples to complete this batch
					for i in range(batch_size - 1 - last_sub_index + first_sub_index):
						extended_features.append(incorrect_instances[i])
				if last_sub_index == len(incorrect_instances):
					break
				# update first and last index
				first_sub_index = last_sub_index
				last_sub_index = min(len(incorrect_instances), first_sub_index + batch_size - 1)
			first_index = last_index
		print("Total batches:", total_batches, "Extended features len:", len(extended_features), "first_index", first_index, "last_index", last_index)
		print("Total ignored examples:", ignored_examples)
		print("total ignored instances:", total_ingored_instances)
	else:
		extended_features = features
	all_input_ids = torch.tensor([f.input_ids for f in extended_features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in extended_features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in extended_features], dtype=torch.long)
	if output_mode == "classification":
		all_labels = torch.tensor([f.label for f in extended_features], dtype=torch.long)
	elif output_mode == "regression":
		all_labels = torch.tensor([f.label for f in extended_features], dtype=torch.float)

	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
	return dataset

# We will train it for 3 epochs
def train_bert_softmax_model(args, train_dataset, model, tokenizer):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = SequentialSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		epoch_loss = 0.0
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {'input_ids':      batch[0],
					  'attention_mask': batch[1],
					  'labels':         batch[3]}
			if args.model_type != 'distilbert':
				inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
			outputs = model(**inputs)
			loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
				torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
			else:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			tr_loss += loss.item()
			epoch_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
						results = evaluate(args, model, tokenizer)
						for key, value in results.items():
							tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
					tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)

					epoch_iterator.set_description("Avg L: {0:.4f}, L: {1:.4f}".format(epoch_loss/global_step, (tr_loss - logging_loss)/args.logging_steps))
					logging_loss = tr_loss

				if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
					# Save model checkpoint
					output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_dir)
					torch.save(args, os.path.join(output_dir, 'training_args.bin'))
					logger.info("Saving model checkpoint to %s", output_dir)


			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

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

def get_predicted_labels(scores, gold_bucket_indices):
	first_index = 0
	predicted_labels = list()
	new_scores = list()
	for last_index in gold_bucket_indices:
		current_bucket_scores = softmax(np.array(scores[first_index:last_index]), axis=0)
		current_predictions = [0]*current_bucket_scores.shape[0]
		current_predictions[np.argmax(current_bucket_scores)] = 1
		predicted_labels.extend(current_predictions)
		new_scores.extend(current_bucket_scores.tolist())
		first_index = last_index
	print(len(new_scores))
	return predicted_labels, new_scores

def evaluate(args, processor, model, tokenizer, eval_output_dir, data_file, bucket_indices, tag):
	eval_dataset = load_and_cache_examples(args, processor, tokenizer, data_file, bucket_indices, "test", evaluate=True)

	if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(eval_output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# Eval!
	logger.info("***** Running evaluation {} *****")
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)
	eval_loss = 0.0
	nb_eval_steps = 0
	preds = None
	out_label_ids = None
	count = 0
	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		count += 1
		# if count % 100 == 0:
		# 	break 
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			inputs = {'input_ids':      batch[0],
					  'attention_mask': batch[1],
					  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM and RoBERTa don't use segment_ids
					  }
			labels = batch[3]
			# 'labels':         batch[3]
			outputs = model(**inputs)
			# logits = torch.ones(batch[0].shape[0], 1)
			logits = outputs[0]
			# print(logits)
			# eval_loss += tmp_eval_loss.mean().item()
		nb_eval_steps += 1
		if preds is None:
			preds = logits.detach().cpu().numpy()
			out_label_ids = labels.detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

	# eval_loss = eval_loss / nb_eval_steps
	scores = preds[:,0]
	predicted_labels, new_scores = get_predicted_labels(scores, bucket_indices)
	# do the full evaluation of Max F1, PR-AUC and P@1
	acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = my_evaluate(new_scores, predicted_labels, out_label_ids, bucket_indices)
	print("Accuracy:{}".format(acc))
	print("ROC_AUC_SCORE:{}".format(roc_auc))
	print("PR_AUC_score:{}".format(pr_auc))
	print("Average Precision Score:{}".format(ap))
	print("Max F1:{}".format(f1_max))
	print("Precision for max F1:{}".format(p_max))
	print("Recall for max F1:{}".format(r_max))
	print("MRR:{}".format(MRR))
	print("Precision@1:{}".format(precision_at_1))

	print("All Pos. Counter:\n{}".format(counter_all_pos))
	print("CM:\n{}".format(cm))
	print("Classification report:\n{}".format(classification_report_str))
	print("\n\n\n\n")
	# result = compute_metrics("mnli", preds, out_label_ids)
	# result = None
	# results.update(result)

	# output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
	# with open(output_eval_file, "w") as writer:
	# 	logger.info("***** Eval results {} *****".format(prefix))
	# 	for key in sorted(result.keys()):
	# 		logger.info("  %s = %s", key, str(result[key]))
	# 		writer.write("%s = %s\n" % (key, str(result[key])))

	return {}

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

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--model_type", default=None, type=str, required=True,
						help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
	parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
						help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
	parser.add_argument("--task_name", default="rule", type=str,
						help="The name of the task to train selected in the list: ")
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--config_name", default="", type=str,
						help="Pretrained config name or path if not the same as model_name")
	parser.add_argument("--tokenizer_name", default="", type=str,
						help="Pretrained tokenizer name or path if not the same as model_name")
	parser.add_argument("--cache_dir", default="", type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length", default=128, type=int,
						help="The maximum total input sequence length after tokenization. Sequences longer "
							 "than this will be truncated, sequences shorter will be padded.")
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--evaluate_during_training", action='store_true',
						help="Rul evaluation during training at each logging step.")
	parser.add_argument("--do_lower_case", action='store_true',
						help="Set this flag if you are using an uncased model.")

	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
						help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--num_train_epochs", default=3.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")

	parser.add_argument('--logging_steps', type=int, default=50,
						help="Log every X updates steps.")
	parser.add_argument('--save_steps', type=int, default=50,
						help="Save checkpoint every X updates steps.")
	parser.add_argument("--eval_all_checkpoints", action='store_true',
						help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
	parser.add_argument("--no_cuda", action='store_true',
						help="Avoid using CUDA when available")
	parser.add_argument('--overwrite_output_dir', action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument('--overwrite_cache', action='store_true',
						help="Overwrite the cached training and evaluation sets")
	parser.add_argument('--seed', type=int, default=42,
						help="random seed for initialization")


	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	args = parser.parse_args()

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
		raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	args.device = device
	# Setup logging
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
					args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

	# Set seed
	set_seed(args)

	# Prepare GLUE task
	processor = RuleResponsesProcessor()
	args.output_mode = "classification"
	label_list = processor.get_labels()
	num_labels = len(label_list)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
	model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)


	# Training
	if args.do_train:
		NEGATIVE_PERCENTAGE = 100
		DATA_FOLDER = "train_data"
		train_file = os.path.join(DATA_FOLDER, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENTAGE))
		train_data = pd.read_csv(train_file, sep='\t', header=None)
		train_bucket_indices = verify_and_generate_bucket_indices(train_data, last_column_index=102)
		with open(train_file,"r") as in_csv:
			reader = csv.reader(in_csv, delimiter='\t')
			train_dataset = load_and_cache_examples(args, processor, tokenizer, reader, train_bucket_indices, tag="Train", evaluate=False)
			global_step, tr_loss = train_bert_softmax_model(args, train_dataset, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


	# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Create output directory if needed
		if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(args.output_dir)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

		# Load a trained model and vocabulary that you have fine-tuned
		model = model_class.from_pretrained(args.output_dir)
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		model.to(args.device)


	# Evaluation
	results = {}
	if args.do_eval and args.local_rank in [-1, 0]:
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		checkpoints = [args.output_dir]
		if args.eval_all_checkpoints:
			checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
			logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
		logger.info("Evaluate the following checkpoints: %s", checkpoints)
		for checkpoint in checkpoints:
			global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
			prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
			
			model = model_class.from_pretrained(checkpoint)
			model.to(args.device)
			DATA_FOLDER = "train_data"
			test_file = os.path.join(DATA_FOLDER, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")
			test_data = pd.read_csv(test_file, sep='\t', header=None)
			test_bucket_indices = verify_and_generate_bucket_indices(test_data, last_column_index=104)
			EVAL_FOLDER = os.path.join(DATA_FOLDER, "bert_softmax_classifier_results")
			with open(test_file,"r") as in_csv:
				reader = csv.reader(in_csv, delimiter='\t')
				result = evaluate(args, processor, model, tokenizer, EVAL_FOLDER, reader, test_bucket_indices, "test")
				result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
				results.update(result)

	return results


if __name__ == "__main__":
	main()



# command
# export CUDA_VISIBLE_DEVICES=3
# python bert_softmax_classification_model.py --model_type bert --model_name_or_path bert-base-uncased --data_dir train_data/bert_softmax_classifier/data_cache --output_dir train_data/bert_softmax_classifier/ckpt --do_train --per_gpu_train_batch_size 50 --save_steps 8000 --logging_steps 2 --learning_rate 5e-6

# python bert_softmax_classification_model.py --model_type bert --model_name_or_path train_data/bert_softmax_classifier/ckpt --output_dir train_data/bert_softmax_classifier/ckpt --data_dir train_data/bert_softmax_classifier/data_cache --do_eval --per_gpu_eval_batch_size 50































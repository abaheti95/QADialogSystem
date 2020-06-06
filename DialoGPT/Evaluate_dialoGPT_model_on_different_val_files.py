import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch

import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size

from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss

from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader


from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,default="./models/small",
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=10000,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--init_weights", type=boolean_string, default=False)
parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')


# do normal parsing
args = parser.parse_args()


# prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

# Prepare tokenizer and config
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))


def evaluate_models_from(GPT_saved_models_folder, eval_file, enc, args):
	# Prepare eval data
	eval_dataloader_loss = DynamicBatchingLoader(
	    eval_file, enc, args.normalize_data,
	    args.eval_batch_size, args.max_seq_length)

	eval_dataloader_gen = get_eval_list_same_length(
	    eval_file, enc, args.eval_batch_size, True)
	# read eval_loss log file
	eval_loss_log_file = os.path.join(GPT_saved_models_folder, "eval_log.txt")
	min_ckpt_old_perplexity = None
	min_ckpt_new_perplexity = None
	min_old_perplexity = 1000000.0
	min_new_perplexity = 1000000.0

	with open(eval_loss_log_file, "r") as reader:
		head_row = next(reader)
		for line in reader:
			line = line.strip()
			epoch, ckpt_no, _, loss, perplexity = line.split(",")
			epoch = int(epoch)
			ckpt_no = int(ckpt_no) - 1
			loss = float(loss)
			perplexity = float(perplexity)
			print(ckpt_no, loss, perplexity, end="")
			if min_old_perplexity > perplexity:
				min_old_perplexity = perplexity
				min_ckpt_old_perplexity = ckpt_no
			# calculate new loss and perplexity
			model_filename = "GP2-pretrain-step-{}.pkl"
			model = load_model(GPT2LMHeadModel(config), os.path.join(GPT_saved_models_folder, model_filename.format(ckpt_no)), args, verbose=True)
			eval_loss, eval_ppl = eval_model_loss(model, enc, eval_dataloader_loss, epoch, args)
			if min_new_perplexity > eval_ppl:
				min_new_perplexity = eval_ppl
				min_ckpt_new_perplexity = ckpt_no
	print("Old best ckpt and perplexity:", min_ckpt_old_perplexity, min_old_perplexity)
	print("New best ckpt and perplexity:", min_ckpt_new_perplexity, min_new_perplexity)
	return min_ckpt_old_perplexity, min_old_perplexity, min_ckpt_new_perplexity, min_new_perplexity
gpt_ss_models = os.path.join("models", "ss", "GPT2.5e-05.32.1gpu.2019-11-09173601")
gpt_ss_plus_models = os.path.join("models", "ss_plus", "GPT2.5e-05.16.1gpu.2019-11-09230759")
gpt_ss_sm_models = os.path.join("models", "ss_finetune", "GPT2.1e-05.32.1gpu.2019-11-08165243")
gpt_ss_plus_sm_models = os.path.join("models", "ss_plus_finetune", "GPT2.1e-05.16.1gpu.2019-11-08181917")
gpt_ss_oqa_models = os.path.join("models", "ss_finetune_opensub_qa", "GPT2.1e-05.32.1gpu.2019-11-15124331")
gpt_ss_plus_oqa_models = os.path.join("models", "ss_plus_finetune_opensub_qa", "GPT2.1e-05.16.1gpu.2019-11-15123950")

gpt_model_folders = [gpt_ss_models, gpt_ss_plus_models, gpt_ss_sm_models, gpt_ss_plus_sm_models, gpt_ss_oqa_models, gpt_ss_plus_oqa_models]

eval_analysis_file = "new_eval_analysis.txt"
with open(eval_analysis_file, "w") as writer:
	# first eval file
	eval_file = os.path.join("data", "mturk_gold_val_file_removed_conflicts_with_ss_train.txt")
	print("Eval file:{}\n".format(eval_file))
	writer.write("Eval file:{}\n".format(eval_file))
	for gpt_folder in gpt_model_folders:
		print(gpt_folder)
		min_ckpt_old_perplexity, min_old_perplexity, min_ckpt_new_perplexity, min_new_perplexity = evaluate_models_from(gpt_folder, eval_file, enc, args)
		writer.write("{}:\n".format(gpt_folder))
		writer.write("{},{},{},{}\n\n".format(min_ckpt_old_perplexity, min_old_perplexity, min_ckpt_new_perplexity, min_new_perplexity))
		writer.flush()
		print()
	# Second eval file
	eval_file = os.path.join("data", "mturk_gold_val_file.txt")
	print("\nEval file:{}\n".format(eval_file))
	writer.write("\nEval file:{}\n".format(eval_file))
	for gpt_folder in gpt_model_folders:
		print(gpt_folder)
		min_ckpt_old_perplexity, min_old_perplexity, min_ckpt_new_perplexity, min_new_perplexity = evaluate_models_from(gpt_folder, eval_file, enc, args)
		writer.write("{}:\n".format(gpt_folder))
		writer.write("{},{},{},{}\n\n".format(min_ckpt_old_perplexity, min_old_perplexity, min_ckpt_new_perplexity, min_new_perplexity))
		writer.flush()
		print()


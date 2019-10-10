# We want to extract all questions which as at least 3 tokens in the response which are not period

# Commands to start the corenlp server
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000

# Classpath to stanford parser
# export CLASSPATH=/home/baheti/QADialogueSystem/stanford-postagger-2018-10-16/stanford-postagger.jar

# spacy is the faster compared to NLTK and stanford PTBtokenizer
# Moses tokenizer is the fastest among all. Much faster than spacy as well
import os
import spacy
nlp = spacy.load("en_core_web_sm")
# Moses tokenizer
from sacremoses import MosesTokenizer
mt = MosesTokenizer()
import time

DATA_FOLDER = "opensub_qa_en"
train_file = os.path.join(DATA_FOLDER, "train.txt")
valid_file = os.path.join(DATA_FOLDER, "valid.txt")
# Spacy tokenized
train_tokenzied_file = os.path.join(DATA_FOLDER, "train_spacy_tokenized.txt")
valid_tokenzied_file = os.path.join(DATA_FOLDER, "valid_spacy_tokenized.txt")
# Moses tokenzied
train_tokenzied_file = os.path.join(DATA_FOLDER, "train_moses_tokenized.txt")
valid_tokenzied_file = os.path.join(DATA_FOLDER, "valid_moses_tokenized.txt")

def tokenize_and_save(file, save_file):
	with open(file, "r") as reader, open(save_file, "w") as writer:
		start_time = time.time()
		for i, line in enumerate(reader):
			# if i == 10:
			# 	break
			if i % 100000==0:
				print(i, time.time() - start_time, "secs")
				start_time = time.time()
			src, tgt = line.split("\t")
			src = src.lower().strip()
			tgt = tgt.lower().strip()

			"""
			# SPACY Tokenizer
			# TODO: Forgot to lower when I was doing spacy tokenization. 
			#		Fix this if you eventually plan to use the Spacy tokenizer
			src_tokens = nlp(src)
			tgt_tokens = nlp(tgt)
			
			tokenized_src = ' '.join([token.text for token in src_tokens])
			tokenized_tgt = ' '.join([token.text for token in tgt_tokens])
			"""

			# Moses Tokenizer
			tokenized_src = mt.tokenize(src, return_str=True, escape=False).replace("\t"," ").replace("`", "'").replace("''", '"')
			tokenized_tgt = mt.tokenize(tgt, return_str=True, escape=False).replace("\t"," ").replace("`", "'").replace("''", '"')
			# print(src)
			# print(tokenized_src)
			# print()

			writer.write("{}\t{}\n".format(tokenized_src, tokenized_tgt))
			# print("{}\t{}\n".format(tokenized_src, tokenized_tgt))
			# print(src)
			# print(tgt)
			# n_tokens_without_periods = sum(1 for token in tgt_tokens if token.text != '.')
			# if n_tokens_without_periods >= 3:
			# 	# write the tokens into files
			# 	usable_instances += 1
			# total_instances += 1
			
			# print()
		print(i)

tokenize_and_save(train_file, train_tokenzied_file)
tokenize_and_save(valid_file, valid_tokenzied_file)
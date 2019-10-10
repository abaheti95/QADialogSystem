# We want to gather squad questions and answers for to create a dialog corpus on which we can train a Seq2Seq model

from QA_datasets.squad_reader import *
import os
from nltk.parse import CoreNLPParser
# start ther server using:
# java -Djava.io.tmpdir=~/tmp -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 & 

# Lexical Parser
parser = CoreNLPParser(url='http://localhost:9000')

squad_train_qas = read_squad_train()

def save_full_q_a_and_parsed():
	q_save_file = os.path.join("squad_seq2seq_train", "squad_train_q.txt")
	lexparsed_save_file = os.path.join("squad_seq2seq_train", "squad_train_lexparsed_q.txt")
	a_save_file = os.path.join("squad_seq2seq_train", "squad_train_a.txt")
	with open(q_save_file, "w") as q_writer, open(a_save_file, "w") as a_writer, open(lexparsed_save_file, "w") as l_writer:

		for i, (q,a) in enumerate(squad_train_qas):
			if i%1000 == 0:
				print(i)
			lexparsed_q = list(str(e) for e in parser.raw_parse(q))[0].replace("\n","")
			# .replace("Tree", "").replace("[","").replace("]", "")
			q_writer.write("{}\n".format(q.strip()))
			l_writer.write("{}\n".format(lexparsed_q.strip()))
			a_writer.write("{}\n".format(a.strip()))

save_full_q_a_and_parsed()

def save_batches_of_q_and_a():
	q_save_file = os.path.join("squad_seq2seq_train", "squad_train_q{}to{}.txt")
	a_save_file = os.path.join("squad_seq2seq_train", "squad_train_a{}to{}.txt")
	start_i = 0
	batch_size = 100
	q_writer = open(q_save_file.format(start_i, start_i+batch_size), "w")
	a_writer = open(a_save_file.format(start_i, start_i+batch_size), "w")
	max_q_len = 0
	max_q = None
	pos = -1
	for i, (q,a) in enumerate(squad_train_qas):
		if i == (start_i+batch_size):
			start_i = i
			q_writer.close()
			a_writer.close()
			q_save_file = os.path.join("squad_seq2seq_train", "squad_train_q{}to{}.txt")
			a_save_file = os.path.join("squad_seq2seq_train", "squad_train_a{}to{}.txt")
			q_writer = open(q_save_file.format(start_i, start_i+batch_size), "w")
			a_writer = open(a_save_file.format(start_i, start_i+batch_size), "w")

		q = q.replace("*","").strip()
		a = a.strip()
		if len(q) > max_q_len:
			max_q_len = len(q)
			pos = i
			max_q = q
		q_writer.write("{}\n".format(q.strip()))
		a_writer.write("{}\n".format(a.strip()))
	print(max_q_len, pos, max_q)
# ../stanford-parser-full-2018-10-17/lexparser.sh squad_train_q.txt > squad_train_q_lexparsed.txt

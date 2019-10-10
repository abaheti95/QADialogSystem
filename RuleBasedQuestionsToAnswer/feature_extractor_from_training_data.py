# We want to extract features from the training data

# We want to extract following features
# 1) Length features - number of tokens
# 2) WH words - boolean feature of which wh-word present
# 3) Negation - if no, none or not is present in the question
# 4) N-gram LM features - unigram, bigram and trigram length normalized log probabilities and perlexities
# 5) Grammar features - number of proper nouns, pronouns, adjectives, adverbs, conjunctions, numbers, noun phrases, prepositional phrases, and subordinate clauses in parse trees
# 6) Rules features - sequence of rules encoded into binary features
# 7) N-gram overlap features - precision and recall of word overlap
# 8) BLEU scores
# 9) F-measure

# start ther server using:
# java -Djava.io.tmpdir=tmp/ -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 & 

import os
import csv
from collections import Counter
import random
from nltk.parse import CoreNLPParser
from nltk.stem import PorterStemmer
from sacremoses import MosesTokenizer, MosesDetokenizer
detokenizer = MosesDetokenizer()
mt = MosesTokenizer()
# wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Lexical Parser
parser = CoreNLPParser(url='http://localhost:9000')
import kenlm
import time

# LM link: http://www.keithv.com/software/giga/
VP_2gram_LM = os.path.join("LMs", "lm_giga_64k_vp_2gram", "lm_giga_64k_vp_2gram.arpa")
NVP_2gram_LM = os.path.join("LMs", "lm_giga_64k_nvp_2gram", "lm_giga_64k_nvp_2gram.arpa")
VP_3gram_LM = os.path.join("LMs", "lm_giga_64k_vp_3gram", "lm_giga_64k_vp_3gram.arpa")
NVP_3gram_LM = os.path.join("LMs", "lm_giga_64k_nvp_3gram", "lm_giga_64k_nvp_3gram.arpa")

WH_words = ["what", "who", "whom", "whose", "when", "where", "which", "why", "how"]
all_parser_nodes = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

def get_LM_prob(lm_model, s):
	return lm_model.score(s, bos = True, eos = True)

def get_LM_perplexity(lm_model, s):
	return lm_model.perplexity(s)

parse_cache = dict()
def get_parse_nodes(s, use_cache=False):
	global parse_cache
	s_parse = None
	if use_cache:
		if s in parse_cache:
			s_parse = parse_cache[s]
		else:
			s_parse = str(list(parser.raw_parse(s))).replace("Tree", "").replace("[","").replace("]", "")
			parse_cache[s] = s_parse
	else:
		s_parse = str(list(parser.raw_parse(s))).replace("Tree", "").replace("[","").replace("]", "")
	# print(s_parse)
	return [e[2:-2] for e in s_parse.split() if e.startswith("(")]

def nodes_to_feature_vector(nodes):
	counter = Counter(nodes)
	feature_vector = [0]*len(all_parser_nodes)
	for i, node in enumerate(all_parser_nodes):
		if node in counter:
			# print("YASSSSSS!!!!", node, counter[node])
			feature_vector[i] = counter[node]
	return feature_vector

def find_overlap(s1, s2):
	# ref: https://stackoverflow.com/a/5095171/4535284
	a_multiset = Counter(s1.split())
	b_multiset = Counter(s2.split())
	return list((a_multiset & b_multiset).elements())

def lemmatize(s):
	return ' '.join([stemmer.stem(e) for e in s.split()])


feature_names = ["l_q", "l_a", "l_r", "what", "who", "whom", "whose", "when", "where", "which", "why", "how", "no_not_none", "q_2gram_lm_prob", "q_3gram_lm_prob", "r_2gram_lm_prob", "r_3gram_lm_prob", "q_2gram_lm_perp", "q_3gram_lm_perp", "r_2gram_lm_perp", "r_3gram_lm_perp", "q_CC", "q_CD", "q_DT", "q_EX", "q_FW", "q_IN", "q_JJ", "q_JJR", "q_JJS", "q_LS", "q_MD", "q_NN", "q_NNS", "q_NNP", "q_NNPS", "q_PDT", "q_POS", "q_PRP", "q_PRP$", "q_RB", "q_RBR", "q_RBS", "q_RP", "q_SYM", "q_TO", "q_UH", "q_VB", "q_VBD", "q_VBG", "q_VBN", "q_VBP", "q_VBZ", "q_WDT", "q_WP", "q_WP$", "q_WRB", "r_CC", "r_CD", "r_DT", "r_EX", "r_FW", "r_IN", "r_JJ", "r_JJR", "r_JJS", "r_LS", "r_MD", "r_NN", "r_NNS", "r_NNP", "r_NNPS", "r_PDT", "r_POS", "r_PRP", "r_PRP$", "r_RB", "r_RBR", "r_RBS", "r_RP", "r_SYM", "r_TO", "r_UH", "r_VB", "r_VBD", "r_VBG", "r_VBN", "r_VBP", "r_VBZ", "r_WDT", "r_WP", "r_WP$", "r_WRB", "precision", "recall", "f-measure"]
def extract_features(q, a, r, vp_2gram_model, vp_3gram_model):
	q_size = len(q.split())
	a_size = len(a.split())
	r_size = len(r.split())
	features = list()
	# 1) length features - number of tokens
	features.append(q_size)
	features.append(a_size)
	features.append(r_size)
	# 2) WH words - boolean feature of which wh-word is present
	wh_features = [0]*len(WH_words)
	temp = " " + q + " "
	for i, wh_word in enumerate(WH_words):
		if (" " + wh_word + " ") in temp:
			wh_features[i] = 1
	features.extend(wh_features)
	# 3) Negation - if no, none or not is present in the question
	if "no" in temp or "not" in temp or "none" in temp:
		features.append(1)
	else:
		features.append(0)
	# 4) N-gram LM features - unigram, bigram and trigram length normalized log probabilities
	features.append(get_LM_prob(vp_2gram_model, q)/float(q_size))
	features.append(get_LM_prob(vp_3gram_model, q)/float(q_size))
	features.append(get_LM_prob(vp_2gram_model, r)/float(r_size))
	features.append(get_LM_prob(vp_3gram_model, r)/float(r_size))
	features.append(get_LM_perplexity(vp_2gram_model, q))
	features.append(get_LM_perplexity(vp_3gram_model, q))
	features.append(get_LM_perplexity(vp_2gram_model, r))
	features.append(get_LM_perplexity(vp_3gram_model, r))
	# 5) Grammar features - number of proper nouns, pronouns, adjectives, adverbs, conjunctions, numbers, noun phrases, prepositional phrases, and subordinate clauses in parse trees
	q_nodes_features = nodes_to_feature_vector(get_parse_nodes(q, use_cache=True))
	r_nodes_features = nodes_to_feature_vector(get_parse_nodes(r))
	features.extend(q_nodes_features)
	features.extend(r_nodes_features)
	# 6) Rules features - sequence of rules encoded into binary features
	# TODO: do this later when you have rules in the data
	# 7) N-gram overlap features - precision and recall of word overlap
	overlap = find_overlap(q,r)
	precision = float(len(overlap))/float(q_size)
	recall = float(len(overlap))/float(r_size)
	features.extend([precision, recall])
	# don't think lemmatization will be of any use
	# q_lemmatized = lemmatize(q)
	# r_lemmatized = lemmatize(r)
	# 8) BLEU scores
	# Don't know if this will be helpful or even can be implemented
	# 9) F-measure
	if precision == 0.0 and recall == 0.0:
		features.append(0.0)
	else:
		features.append((2.0 * precision * recall / (precision + recall)))
	return features


# Train file
SAVE_FOLDER = "train_data"
train_data = os.path.join("mturk_experiments", "squad_first_1000_train_data.tsv")
features_save_data = os.path.join(SAVE_FOLDER, "squad_first_1000_train_data_subsampled_features.tsv")

train_data = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_train_data.tsv")
features_save_data = os.path.join(SAVE_FOLDER, "squad_final_train_data_features.tsv")

def compute_features_for_the_train_data(train_data_file, features_save_data_file):
	vp_2gram_model = kenlm.Model(VP_2gram_LM)
	# nvp_2gram_model = kenlm.Model(NVP_2gram_LM)
	vp_3gram_model = kenlm.Model(VP_3gram_LM)
	# nvp_3gram_model = kenlm.Model(NVP_3gram_LM)
	with open(train_data_file, "r") as train_tsv, open(features_save_data_file, "w") as features_tsv:
		reader = csv.reader(train_tsv, delimiter='\t')
		writer = csv.writer(features_tsv, delimiter='\t')

		head_row = next(reader)
		new_head_row = head_row[0:-1] + feature_names
		new_head_row.append(head_row[-1])
		writer.writerow(new_head_row)
		start_time = time.time()
		for i, row in enumerate(reader):
			question = row[0]
			answer = row[1]
			response = row[2]
			question = question.strip()
			question = mt.tokenize(question, return_str=True, escape=False)
			question = question.lower()
			answer = answer.strip()
			answer = answer.lower()
			response = response.strip()
			response = mt.tokenize(response, return_str=True, escape=False)
			response = response.lower()
			row[0] = question
			row[1] = answer
			row[2] = response
			count = int(row[-1])
			if i%1000==0:
				print(i, "time :", time.time() - start_time, "secs")
				start_time = time.time()
			# if count == 0:
			# 	p = random.uniform(0, 1)
			# 	if p > 0.05:
			# 		continue
			features = extract_features(question, answer, response, vp_2gram_model, vp_3gram_model)

			final_row = row[0:-1] + features
			# print(len(row), len(features), len(final_row))
			final_row.append(row[-1])
			# print(question, answer, response)
			# print(final_row[:8])
			writer.writerow(final_row)
# compute_features_for_the_train_data(train_data, features_save_data)

new_train_data = os.path.join("mturk_experiments", "squad_final_3000_sample_batches", "results", "squad_final_train_data_removed_short_response_affinity_workers.tsv")
new_feature_file = os.path.join(SAVE_FOLDER, "squad_final_train_data_features_removed_short_response_affinity_workers.tsv")
# We extracted features for these sentences before. But now we have new train file which has some of the annotations removed. Therefore the features won't change but the final count will change
# Instead of re-running the feature extractor on the whole data we will simply copy the features from the existing file into a new one and upate the counts from the new train file
# NOTE: the idea of copying the features from a different file didn't work so we will have to calculcate featuers again for all the 
compute_features_for_the_train_data(new_train_data, new_feature_file)

def pre_check(sentence):
	# in this function we will lowercase, remove whitespaces and clean the sentence so that we can compare them whithout needed to tokenize it
	return sentence.lower().replace("&apos;","'").replace("&quot;", '"').replace("`","'").replace(" ", "").replace("''", '"')

def update_counts_for_new_training_data(new_train_data_file, existing_feature_file, new_feature_file):
	with open(new_train_data_file, "r") as train_tsv, open(existing_feature_file, "r") as features_in_tsv, open(new_feature_file, "w") as features_out_tsv:
		reader = csv.reader(train_tsv, delimiter='\t')
		in_features_reader = csv.reader(features_in_tsv, delimiter='\t')
		writer = csv.writer(features_out_tsv, delimiter='\t')
		train_head_row = next(reader)
		features_head_row = next(in_features_reader)
		writer.writerow(features_head_row)
		# the train data file and the existing features file has the same amount of lines and rows and they should be aligned as well
		for i, (train_row, feature_row) in enumerate(zip(reader, in_features_reader)):
			print(train_row)
			#verify if the train row and feature_row q,a,r are the same
			if pre_check(train_row[0]) != pre_check(feature_row[0]) or \
				pre_check(train_row[1]) != pre_check(feature_row[1]) or \
				pre_check(train_row[2]) != pre_check(feature_row[2]):
				print("Something wrong. Train row and Feature row not matching for row", i)
				print(train_row)
				print(feature_row[:6])
				print(feature_row[-1])
				exit()
			if feature_row[-1] != train_row[-1]:
				print("Difference in row", i)
				print(train_row[-1], "vs", feature_row[-1])
			feature_row[-1] = train_row[-1]
			writer.writerow(feature_row)

# update_counts_for_new_training_data(new_train_data, features_save_data, new_feature_file)





























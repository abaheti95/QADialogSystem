
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from collections import Counter

import os

def make_dir_if_not_exists(dir):
	if not os.path.exists(dir):
		print("Making Directory:", dir)
		os.makedirs(dir)

DATA_DIR = "train_data"
RESULTS_DIR = os.path.join(DATA_DIR, "logistic_final_results")

make_dir_if_not_exists(RESULTS_DIR)

train_subset_file = os.path.join(DATA_DIR, "train_squad_first_1000_train_data_subsampled_features.tsv")
test_subset_file = os.path.join(DATA_DIR, "test_squad_first_1000_train_data_subsampled_features.tsv")

train_subset_file = os.path.join(DATA_DIR, "train_squad_first_1000_train_data_features.tsv")
test_subset_file = os.path.join(DATA_DIR, "test_squad_first_1000_train_data_features.tsv")

NEGATIVE_PERCENT = 50
train_subset_file = os.path.join(DATA_DIR, "train_count2_squad_first_1000_train_data_features_50_negative.tsv")
test_subset_file = os.path.join(DATA_DIR, "test_count1_squad_first_1000_train_data_features_with_train_50_negative.tsv")

NEGATIVE_PERCENT = 20
train_subset_file = os.path.join(DATA_DIR, "train_count2_squad_first_1000_train_data_features_20_negative.tsv")
test_subset_file = os.path.join(DATA_DIR, "test_count1_squad_first_1000_train_data_features_with_train_20_negative.tsv")

# NEGATIVE_PERCENT = 10
# train_subset_file = os.path.join(DATA_DIR, "train_count2_squad_first_1000_train_data_features_10_negative.tsv")
# test_subset_file = os.path.join(DATA_DIR, "test_count1_squad_first_1000_train_data_features_with_train_10_negative.tsv")

# NEGATIVE_PERCENT = 5
# train_subset_file = os.path.join(DATA_DIR, "train_count2_squad_first_1000_train_data_features_5_negative.tsv")
# test_subset_file = os.path.join(DATA_DIR, "test_count1_squad_first_1000_train_data_features_with_train_5_negative.tsv")

# NEGATIVE_PERCENT = 1
# train_subset_file = os.path.join(DATA_DIR, "train_count2_squad_first_1000_train_data_features_1_negative.tsv")
# test_subset_file = os.path.join(DATA_DIR, "test_count1_squad_first_1000_train_data_features_with_train_1_negative.tsv")

feature_names = ["l_q", "l_a", "l_r", "what", "who", "whom", "whose", "when", "where", "which", "why", "how", "no_not_none", "q_2gram_lm_prob", "q_3gram_lm_prob", "r_2gram_lm_prob", "r_3gram_lm_prob", "q_2gram_lm_perp", "q_3gram_lm_perp", "r_2gram_lm_perp", "r_3gram_lm_perp", "q_CC", "q_CD", "q_DT", "q_EX", "q_FW", "q_IN", "q_JJ", "q_JJR", "q_JJS", "q_LS", "q_MD", "q_NN", "q_NNS", "q_NNP", "q_NNPS", "q_PDT", "q_POS", "q_PRP", "q_PRP$", "q_RB", "q_RBR", "q_RBS", "q_RP", "q_SYM", "q_TO", "q_UH", "q_VB", "q_VBD", "q_VBG", "q_VBN", "q_VBP", "q_VBZ", "q_WDT", "q_WP", "q_WP$", "q_WRB", "r_CC", "r_CD", "r_DT", "r_EX", "r_FW", "r_IN", "r_JJ", "r_JJR", "r_JJS", "r_LS", "r_MD", "r_NN", "r_NNS", "r_NNP", "r_NNPS", "r_PDT", "r_POS", "r_PRP", "r_PRP$", "r_RB", "r_RBR", "r_RBS", "r_RP", "r_SYM", "r_TO", "r_UH", "r_VB", "r_VBD", "r_VBG", "r_VBN", "r_VBP", "r_VBZ", "r_WDT", "r_WP", "r_WP$", "r_WRB", "precision", "recall", "f-measure"]

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


NEGATIVE_PERCENT = 10
# We will store all the results in a dictionary with key as NEGATIVE_PERCENT
all_results = dict()
all_results_save_file = os.path.join(RESULTS_DIR, "logistic_all_results.txt")

test_results_image = os.path.join(RESULTS_DIR, "logistic_train_count2{}_test_count1_{}_negative_squad_final_class_weight{}.png")
test_results_output = os.path.join(RESULTS_DIR, "logistic_train_count2{}_test_count1_{}_negative_squad_final_class_weight{}.txt")

val_file = os.path.join(DATA_DIR, "val_count1_squad_final_train_data_features.tsv")
# test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features_with_new_info.tsv")

# Count 2 files
val_file = os.path.join(DATA_DIR, "val_count2_squad_final_train_data_features.tsv")
# test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_count2_squad_final_train_data_features_with_new_info.tsv")


# Shortest count 2 files
val_file = os.path.join(DATA_DIR, "val_shortest_count2_squad_final_train_data_features.tsv")
# test_file = os.path.join(DATA_DIR, "test_count1_squad_final_train_data_features.tsv")
test_file = os.path.join(DATA_DIR, "test_shortest_count2_squad_final_train_data_features_with_new_info.tsv")
val_data = pd.read_csv(val_file, sep='\t', header=None)
test_data = pd.read_csv(test_file, sep='\t', header=None)
print(val_data.shape)
print(test_data.shape)
# Split the X_test into 2 halves based on the value of HARD or EASY question column
easy_test_data = test_data[test_data[7] == 0].copy().reset_index(drop=True)
hard_test_data = test_data[test_data[7] == 1].copy().reset_index(drop=True)
print(easy_test_data.shape)
print(hard_test_data.shape)

test_bucket_indices = verify_and_generate_bucket_indices(test_data, last_column_index=104)
easy_test_bucket_indices = verify_and_generate_bucket_indices(easy_test_data, last_column_index=104)
hard_test_bucket_indices = verify_and_generate_bucket_indices(hard_test_data, last_column_index=104)

val_bucket_indices = verify_and_generate_bucket_indices(val_data)

def generate_test_labels(test_counts, shortest_responses_labels):
	Y_labels = list()
	for i, e in enumerate(test_counts):
		if shortest_responses_labels[i] == 1:
			if e >= 1:
				Y_labels.append(1)
			else:
				Y_labels.append(0)
		else:
			if e >= 1:
				Y_labels.append(1)
			else:
				Y_labels.append(0)
	return Y_labels

def evaluate_and_save_results(model, best_class_weight, data, X, Y, bucket_indices, shortest_responses_labels, output_file, image_title, image_file):
	predicted = model.predict(X)
	probs = model.predict_proba(X)
	Y_scores = probs[:,1]
	acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = evaluate(Y_scores, predicted, Y, bucket_indices)
	
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
	print("IMAGE TITLE:", image_title)
	plt.title(image_title.format(ap))
	plt.savefig(image_file, dpi=300)

	with open(output_file, "w") as writer:

		writer.write("Class weight:(0:{} 1:{})\n".format(class_weight[0], class_weight[1]))
		print("Class weight:(0:{} 1:{})".format(class_weight[0], class_weight[1]))
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



		#feature importance
		feature_importance = abs(model.coef_[0])
		sorted_idx = np.argsort(-feature_importance)
		writer.write("Important features:\n")
		for idx in sorted_idx:
			writer.write("{}\t{}\t{}\n".format(idx, feature_names[idx], model.coef_[0][idx]))
		
		# For each bucket we will print top 10 predictions
		first_index = 0
		counter = 0
		writer.write("Predictions:")
		for last_index in bucket_indices:
			tuples = zip(list(range(first_index, last_index)), Y[first_index:last_index], Y_scores[first_index:last_index], shortest_responses_labels[first_index:last_index])
			sorted_by_score = sorted(tuples, key=lambda tup: tup[2], reverse=True)
			# write the shortest response first
			l = [(index, gold_label,score, shortest_response_label) for index, gold_label,score, shortest_response_label in sorted_by_score if shortest_response_label == 1]
			if len(l) != 1:
				print("ERROR")
				print(l)
				for index, gold_label,score, shortest_response_label in l:
					print(data.iloc[index][0], data.iloc[index][2])
				print("\n")
				exit()
				
			index, gold_label,score, shortest_response_label = l[0]
			writer.write("Shortest response:{} -- {}\n".format(data.iloc[index][0], data.iloc[index][2]))
			count = 0
			for index1, gold_label, score, shortest_response_label in sorted_by_score:
				writer.write("{}\t\t{}\t\t{}\t\t{}\t{}\t{}\n".format(data.iloc[index1][0],
																data.iloc[index1][1],
																data.iloc[index1][2],
																Y_scores[index1], predicted[index1],
																Y[index1]))
				count += 1
				counter += 1
				assert(gold_label==Y[index1])
				if count == 10:
					break
			first_index = last_index
			if counter >= 5000:
				break
	return acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str 

def verify_if_bucket_indices_have_one_shortest_response(bucket_indices, shortest_responses_labels):
	first_index = 0
	for last_index in bucket_indices:
		if sum(shortest_responses_labels[first_index:last_index]) != 1:
			return False
		first_index = last_index
	return True


for NEGATIVE_PERCENT in [1,5,10,20,50,100]:
# for NEGATIVE_PERCENT in [100]:
	train_file = os.path.join(DATA_DIR, "train_count2_squad_final_train_data_features_{}_negative.tsv".format(NEGATIVE_PERCENT))

	results_image = os.path.join(RESULTS_DIR, "logistic_train_count2_val_count1_{}_negative_squad_final_class_weight{}.png")
	results_output = os.path.join(RESULTS_DIR, "logistic_train_count2_val_count1_{}_negative_squad_final_class_weight{}.txt")

	train_data = pd.read_csv(train_file, sep='\t', header=None)
	print(train_data.shape)

	train_data.describe()

	X_train = train_data.iloc[:, 6:-1]
	X_val = val_data.iloc[:, 6:-1]
	
	X_test = test_data.iloc[:, 8:-1]
	test_shortest_responses_labels = test_data[6].tolist()
	print("Shortest responses count:", sum(test_shortest_responses_labels))
	print("bucket indices len:", len(test_bucket_indices))
	print(verify_if_bucket_indices_have_one_shortest_response(test_bucket_indices, test_shortest_responses_labels))

	easy_X_test = easy_test_data.iloc[:, 8:-1]
	easy_test_shortest_responses_labels = easy_test_data[6].tolist()
	print("Easy Shortest responses count:", sum(easy_test_shortest_responses_labels))
	print("bucket indices len:", len(easy_test_bucket_indices))
	print(verify_if_bucket_indices_have_one_shortest_response(easy_test_bucket_indices, easy_test_shortest_responses_labels))
	hard_X_test = hard_test_data.iloc[:, 8:-1]
	hard_test_shortest_responses_labels = hard_test_data[6].tolist()
	print("Hard Shortest responses count:", sum(hard_test_shortest_responses_labels))
	print("bucket indices len:", len(hard_test_bucket_indices))
	print(verify_if_bucket_indices_have_one_shortest_response(hard_test_bucket_indices, hard_test_shortest_responses_labels))

	# Resclaing the data
	min_vector = X_train.min()
	X_train -= min_vector
	X_val -= min_vector
	X_test -= min_vector
	easy_X_test -= min_vector
	hard_X_test -= min_vector
	max_vector = X_train.max().values
	# print(max_vector.shape)
	for i in range(max_vector.shape[0]):
		if max_vector[i] == 0:
			max_vector[i] = 1
	# print(max_vector)
	# print(0 in max_vector)
	# print(min(max_vector))
	# exit()
	X_train /= max_vector
	X_val /= max_vector
	X_test /= max_vector
	easy_X_test /= max_vector
	hard_X_test /= max_vector
	# print(X_train.isnull().values.any())
	X_train = X_train.fillna(0)
	# print(X_val.isnull().values.any())
	X_val = X_val.fillna(0)
	X_val.replace([np.inf, -np.inf], np.nan)
	# print("NAN in X_val:", X_val.isnull().values.any())
	# print(X_val.max())
	# print(max(X_val.max()))
	# print(X_test.isnull().values.any())
	X_test = X_test.fillna(0)
	easy_X_test = easy_X_test.fillna(0)
	hard_X_test = hard_X_test.fillna(0)
	Y_train = [1 if e >= 1 else 0 for e in train_data.iloc[:, -1]]
	Y_val = [1 if e >= 1 else 0 for e in val_data.iloc[:, -1]]
	Y_test = generate_test_labels(test_data[104], test_shortest_responses_labels)
	easy_Y_test = generate_test_labels(easy_test_data[104], easy_test_shortest_responses_labels)
	hard_Y_test = generate_test_labels(hard_test_data[104], hard_test_shortest_responses_labels)
	# print(X_train.shape)
	print(len(test_shortest_responses_labels))
	print(sum(test_shortest_responses_labels))
	print(len(Y_test))

	print(len(easy_test_shortest_responses_labels))
	print(sum(easy_test_shortest_responses_labels))
	print(len(easy_Y_test))

	print(len(hard_test_shortest_responses_labels))
	print(sum(hard_test_shortest_responses_labels))
	print(len(hard_Y_test))

	print(len(Y_train))
	print(type(Y_val))

	class_weights = [{0: 0.5, 1: 0.5}, {0: 0.25, 1: 0.75}, {0: 0.2, 1: 0.8}, {0: 0.15, 1: 0.85}, {0: 0.1, 1: 0.9}, {0: 0.01, 1: 0.99}]
	# class_weights = [{0: 0.5, 1: 0.5}]
	best_model = None
	best_class_weight = None
	best_mrr = 0.0
	for class_weight in class_weights:
		model = LogisticRegression(class_weight=class_weight)
		model.fit(X_train, Y_train)
		# predict class labels for the test set
		predicted = model.predict(X_val)
		probs = model.predict_proba(X_val)
		Y_scores = probs[:,1]
		acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = evaluate(Y_scores, predicted, Y_val, val_bucket_indices)
		
		# print(len(precisions), len(recalls), len(thresholds))
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
		plt.title('PR curve: AP={0:0.2f} NEG%={1}% Class weight=(0:{2} 1:{3})'.format(ap, NEGATIVE_PERCENT, class_weight[0], class_weight[1]))
		plt.savefig(results_image.format(NEGATIVE_PERCENT, class_weight[1], dpi=300))

		# generate evaluation metrics
		with open(results_output.format(NEGATIVE_PERCENT, class_weight[1]), "w") as writer:

			writer.write("Class weight:(0:{} 1:{})\n".format(class_weight[0], class_weight[1]))
			print("Class weight:(0:{} 1:{})".format(class_weight[0], class_weight[1]))
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
			if f1_max > best_mrr:
				best_model = model
				best_mrr = f1_max
				best_class_weight = class_weight
			writer.write("Classification report:\n{}\n".format(classification_report_str))
			print("Classification report:\n{}".format(classification_report_str))
			writer.write("Predictions:")

			#feature importance
			feature_importance = abs(model.coef_[0])
			sorted_idx = np.argsort(-feature_importance)
			writer.write("Important features:\n")
			for idx in sorted_idx:
				writer.write("{}\t{}\t{}\n".format(idx, feature_names[idx], model.coef_[0][idx]))
			
			# For each bucket we will print top 10 predictions
			first_index = 0
			counter = 0
			for last_index in val_bucket_indices:
				tuples = zip(list(range(first_index, last_index)), Y_val[first_index:last_index], Y_scores[first_index:last_index])
				sorted_by_score = sorted(tuples, key=lambda tup: tup[2], reverse=True)
				count = 0
				for index, gold_label, score in sorted_by_score:
					writer.write("{}\t\t{}\t\t{}\t\t{}\t{}\t{}\n".format(val_data.iloc[index][0],
																	val_data.iloc[index][1],
																	val_data.iloc[index][2],
																	probs[index, 1], predicted[index],
																	Y_val[index]))
					assert(gold_label==Y_val[index])
					count += 1
					counter += 1
					if count == 10:
						break
				first_index = last_index
				if counter >= 5000:
					break
		# exit()
	# Evalaute test on best model
	test_results = list()
	# we will save the test results of all, easy and hard in a single list
	# Full test data
	print("\n\n\n\n\nEVALUATING ON FULL TEST")
	test_results.append(evaluate_and_save_results(best_model, best_class_weight, test_data, X_test, Y_test, test_bucket_indices, test_shortest_responses_labels,
		test_results_output.format("", NEGATIVE_PERCENT, best_class_weight[1]), 
		'PR curve: AP=%.2f NEG%={0}% Class weight=(0:{1} 1:{2})'.format(NEGATIVE_PERCENT, best_class_weight[0], best_class_weight[1]).replace("%.2f", "{0:.2f}"),
		test_results_image.format("", NEGATIVE_PERCENT, best_class_weight[1])))
	# EASY test data
	print("EVALUATING ON EASY TEST")
	test_results.append(evaluate_and_save_results(best_model, best_class_weight, easy_test_data, easy_X_test, easy_Y_test, easy_test_bucket_indices, easy_test_shortest_responses_labels,
		test_results_output.format("_EASY", NEGATIVE_PERCENT, best_class_weight[1]), 
		'PR curve: AP=%.2f NEG%={0}% Class weight=(0:{1} 1:{2})'.format(NEGATIVE_PERCENT, best_class_weight[0], best_class_weight[1]).replace("%.2f", "{0:.2f}"),
		test_results_image.format("_EASY", NEGATIVE_PERCENT, best_class_weight[1])))
	# HARD test data
	print("EVALUATING ON HARD TEST")
	test_results.append(evaluate_and_save_results(best_model, best_class_weight, hard_test_data, hard_X_test, hard_Y_test, hard_test_bucket_indices, hard_test_shortest_responses_labels,
		test_results_output.format("_HARD", NEGATIVE_PERCENT, best_class_weight[1]), 
		'PR curve: AP=%.2f NEG%={0}% Class weight=(0:{1} 1:{2})'.format(NEGATIVE_PERCENT, best_class_weight[0], best_class_weight[1]).replace("%.2f", "{0:.2f}"),
		test_results_image.format("_HARD", NEGATIVE_PERCENT, best_class_weight[1])))
	
	

	all_results[NEGATIVE_PERCENT] = (best_class_weight, test_results)

# Save all_results in a file
with open(all_results_save_file, "w") as writer:
	writer.write("NEGATIVE PERCENT\tBest Class weight\tAccuracy\troc_auc_score\tprecision recall auc\taverage precision\tMax F1 score\tPrecision for max F1\tRecall for max F1\tMRR\tPrecision@1\tConfusion Matrix\tClassification Report\n")
	for NEGATIVE_PERCENT in sorted(all_results.keys()):
		best_class_weight, test_results = all_results[NEGATIVE_PERCENT]
		# Full
		writer.write("Full test data\n")
		acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = test_results[0]
		writer.write("{0}\t{1}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\n{11}\n{12}\n{13}\n\n".format(NEGATIVE_PERCENT, best_class_weight, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
		# Easy
		writer.write("Easy test data\n")
		acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = test_results[1]
		writer.write("{0}\t{1}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\n{11}\n{12}\n{13}\n\n".format(NEGATIVE_PERCENT, best_class_weight, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
		# Hard
		writer.write("Hard test data\n")
		acc, cm, roc_auc, pr_auc, ap, f1_max, p_max, r_max, precision, recall, thresholds, MRR, precision_at_1, counter_all_pos, classification_report, classification_report_str = test_results[2]
		writer.write("{0}\t{1}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\n{11}\n{12}\n{13}\n\n".format(NEGATIVE_PERCENT, best_class_weight, acc, roc_auc, pr_auc, ap, f1_max, p_max, r_max, MRR, precision_at_1, counter_all_pos, cm, classification_report_str))
		

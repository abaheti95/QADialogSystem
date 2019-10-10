# We want to combine the answers from the squad model with the test set questions (extracted from Squad Dev set)
import os
import codecs
import json
from sacremoses import MosesTokenizer
mt = MosesTokenizer()

def get_predicted_answers_dict(predictions, squad_format_test_file):
	total_count = count = 0
	predicted_answers_dict = dict()
	with codecs.open(squad_format_test_file, "r", "utf8") as reader:
		all_data = json.load(reader)
		data = all_data['data']
		for i in range(len(data)):
			paragraphs = data[i]['paragraphs']
			# Paragraphs is a list of paragraph from the document and each entry in that list is one paragraph with list of questions and answers
			for paragraph in paragraphs:
				context = paragraph['context']
				qas = paragraph['qas']
				for qa in qas:
					question = qa['question']
					id = qa['id']
					answer = qa['answers'][0]['text']
					# get the prediction for current question
					prediction_answer = predictions[id]
					if answer == prediction_answer:
						count += 1
					total_count += 1
					# Save this in a dict for future use
					processed_q = mt.tokenize(question.replace("`", "'").replace("''", '"').lower(), return_str=True, escape=False).strip()
					processed_a = mt.tokenize(answer.lower(), return_str=True, escape=False).strip()
					predicted_answers_dict[(processed_q, processed_a)] = prediction_answer.lower().strip()
	print(count, total_count)
	return predicted_answers_dict

def save_the_predictions_in_src_style_file_for_opennmt_testing(predicted_answers_dict, src_original_test_file, src_final_save_file):
	total_count = count = missed_count = 0
	number_of_questions_with_answer_length_greater_than_5 = 0
	with open(src_original_test_file, "r") as reader, open(src_final_save_file, "w") as writer:
		for line in reader:
			q, a = line.strip().split(" ||| ")
			try:
				predicted_a = predicted_answers_dict[(q,a)]
			except KeyError:
				missed_count += 1
				continue
			predicted_a = mt.tokenize(predicted_a, return_str=True, escape=False).lower().strip()
			if len(predicted_a.split()) > 5:
				number_of_questions_with_answer_length_greater_than_5 += 1
			writer.write("{} ||| {}\n".format(q, predicted_a))
	print(missed_count)
	print(number_of_questions_with_answer_length_greater_than_5)

PREDICTIONS_DIR = "bert"
predictions_file = os.path.join(PREDICTIONS_DIR, "predictions.json")
# Read and save the predictions
with open(predictions_file) as f:
    predictions = json.load(f)

# We will read the ids from the squad dev subset for test file and map it to questions and answers
DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_dev_moses_tokenized")
squad_dev_subset_for_test_file = os.path.join(DATA_DIR, "squad_dev_moses_test_for_squad_model.json")
src_squad_dev_moses_tokenized_file = os.path.join(DATA_DIR, "src_squad_seq2seq_dev_moses_test.txt")

src_squad_dev_squad_model_predictions_save_file = os.path.join(DATA_DIR, "src_squad_seq2seq_dev_moses_test_squad_model_predictions.txt")

# predicted_answers_dict = get_predicted_answers_dict(predictions, squad_dev_subset_for_test_file)

# save_the_predictions_in_src_style_file_for_opennmt_testing(predicted_answers_dict, src_squad_dev_moses_tokenized_file, src_squad_dev_squad_model_predictions_save_file)

# We will read the ids from the squad seq2seq_predicted_test set and map it to questions and answers
DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_train_moses_tokenized")
squad_seq2seq_predicted_test_squad_format_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_moses_test_for_squad_model.json")
src_squad_seq2seq_predicted_moses_tokenized_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_responses_test.txt")

src_squad_seq2seq_predicted_test_squad_model_predictions_save_file = os.path.join(DATA_DIR, "src_squad_seq2seq_predicted_test_moses_squad_model_predictions_20_missing_predictions.txt")

total_count = count = 0
predicted_answers_dict = get_predicted_answers_dict(predictions, squad_seq2seq_predicted_test_squad_format_file)

save_the_predictions_in_src_style_file_for_opennmt_testing(predicted_answers_dict, src_squad_seq2seq_predicted_moses_tokenized_file, src_squad_seq2seq_predicted_test_squad_model_predictions_save_file)
















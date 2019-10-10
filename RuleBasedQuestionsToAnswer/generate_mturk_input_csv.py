import csv
import os
import random
random.seed(901)
from sacremoses import MosesTokenizer, MosesDetokenizer
detokenizer = MosesDetokenizer()

# rule_based_responses_file = "rule_based_system_generated_answers.txt"
# rule_based_questions_file = "rule_based_system_saved_questions.txt"
save_csv_file = os.path.join("mturk_experiments", "sample_generated_input.csv")
rule_based_responses_file = os.path.join("natural_questions", "rule_based_system_natural_questions_generated_answers.txt")
rule_based_questions_file = os.path.join("natural_questions", "rule_based_system_natural_questions_saved_questions.txt")

# SQUAD mturk first 1000 sample
rule_based_questions_file = os.path.join("squad_train_sample", "rule_based_system_squad_mturk_first_1000_sample_questions.txt")
rule_based_responses_file = os.path.join("squad_train_sample", "rule_based_system_squad_mturk_first_1000_sample_responses.txt")
# save_csv_file = os.path.join("mturk_experiments", "natural_questions_generated_sample_input.csv")
# save_csv_file = os.path.join("mturk_experiments", "natural_questions_generated_10_sample_input.csv")
save_csv_file = os.path.join("mturk_experiments", "squad_questions_generated_fist_1000_sample_input.csv")

# SQUAD mturk final 3000 sample
rule_based_questions_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_saved_questions.txt")
rule_based_responses_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_generated_answers.txt")
rule_based_rules_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_generated_answer_rules.txt")
save_csv_file = os.path.join("mturk_experiments", "squad_questions_generated_final_3000_sample_input_corrected.csv")

# TODO: DEBUG
# # SQUAD mturk final 3000 sample
# rule_based_questions_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_saved_questions_debug.txt")
# rule_based_responses_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_generated_answers_debug.txt")
# rule_based_rules_file = os.path.join("squad_train_sample", "rule_based_system_squad_train_sample_case_sensitive_responses_generated_answer_rules_debug.txt")
# save_csv_file = os.path.join("mturk_experiments", "squad_questions_generated_final_3000_sample_input_debug.csv")

def detokenize_responses(responses_and_rules):
	# Only detokenize the reponse part in the tuple
	return [(detokenizer.detokenize(response.split(), return_str=True), rule) for (response, rule) in responses_and_rules]
	# TODO: debugging change later
	# return responses_and_rules

def list_to_quoted_string(responses_and_rules):
	# we want to convert this list of responses and rules to a single string which has list of lists like structure.
	# This string will be used directly in the javascript part as an array

	responses_and_rules_removed_answer_brackets = [(response.replace("{", "").replace("}", "").strip(), rule) for response, rule in responses_and_rules]

	responses_and_rules_removed_answer_brackets_detokenized = detokenize_responses(responses_and_rules_removed_answer_brackets)
	# print("Detokenized unmarked")

	final_response_string = ""
	for (response, rule) in responses_and_rules_removed_answer_brackets_detokenized:
		final_response_string += "'" + response.replace("'", "\\'") + "',"

	final_rule_string = ""
	for (response, rule) in responses_and_rules_removed_answer_brackets_detokenized:
		final_rule_string += "'" + rule.replace("'", "\\'") + "',"
	
	final_response_string = final_response_string[:len(final_response_string)-1]
	final_rule_string = final_rule_string[:len(final_rule_string)-1]
	return final_response_string, final_rule_string

# ref: https://stackoverflow.com/a/3415150/4535284
from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
	"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(fillvalue=fillvalue, *args)

with open(rule_based_questions_file, "r") as q_reader, open(rule_based_responses_file, "r") as r_reader, open(rule_based_rules_file, "r") as rule_reader:
	# accumulate all the responses and the question
	question = next(q_reader).strip()
	current_responses_and_rules = list()
	current_answer = ""
	all_samples = list()
	for id, (response, rule) in enumerate(zip(r_reader, rule_reader)):
		response = response.strip().lower()
		rule = rule.strip()
		if response and rule:
			# accumulate in the list
			current_responses_and_rules.append((response, rule))
			current_answer = response[response.index("{")+1: response.index("}")]
			# print(question, current_answer)
		else:
			# save in the global samples list
			current_responses, current_rules = list_to_quoted_string(current_responses_and_rules)
			all_samples.append((question, current_answer, current_responses, current_rules))
			# refresh the list
			current_responses_and_rules = list()
			# select next question
			try:
				question = next(q_reader).strip()
			except StopIteration:
				# Last question is already over so ignore this exception as the loop is going to end
				question = None



with open(save_csv_file, "w") as csv_file:
	writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(["question1", "answer1", "question2", "answer2", "question3", "answer3", "question4", "answer4", "question5", "answer5", "question6", "answer6", "question7", 
		"answer7", "question8", "answer8", "question9", "answer9", "question10", "answer10", "responses1", "responses2", "responses3", "responses4", 
		"responses5", "responses6", "responses7", "responses8", "responses9", "responses10", "rules1", "rules2", "rules3", "rules4", 
		"rules5", "rules6", "rules7", "rules8", "rules9", "rules10"])

	# get a sample of 100 questions
	# sample_100 = random.sample(all_samples, 100)
	sample_3000 = random.sample(all_samples, 3000)
	"""
	sample_3000_sorted_by_response_size = sorted(sample_3000, key=lambda tup: len(tup[2]))
	sample_3000_length_balanced = list()
	for i in range(1500):
		# if i < 15:
		# 	# 622691
		# 	print(len(sample_3000_sorted_by_response_size[3000-1-i][2]))
			# print("vs")
			# print(len(sample_3000_sorted_by_response_size[i][2]))
		sample_3000_length_balanced.append(sample_3000_sorted_by_response_size[i])
		sample_3000_length_balanced.append(sample_3000_sorted_by_response_size[3000-1-i])
	"""
	# print(type(sample_3000))
	# print(sample_100)
	i = 1
	for (question1, answer1, responses1, rules1), (question2, answer2, responses2, rules2), (question3, answer3, responses3, rules3), (question4, answer4, responses4, rules4), \
		(question5, answer5, responses5, rules5), (question6, answer6, responses6, rules6), (question7, answer7, responses7, rules7), (question8, answer8, responses8, rules8), \
		(question9, answer9, responses9, rules9), (question10, answer10, responses10, rules10) in grouper(10, sample_3000):
		if i%10==0:
			print(i*10)
		# print(question, answer, responses_and_rules_string)
		# print(question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5, answer5, question6, answer6, question7, answer7, question8, answer8, question9, answer9, question10, answer10)

		writer.writerow([question1, answer1, question2, answer2, question3, answer3, question4, answer4, question5, answer5, question6, answer6, question7, answer7, 
			question8, answer8, question9, answer9, question10, answer10, responses1, responses2, responses3, responses4, responses5, responses6, responses7, responses8, 
			responses9, responses10, rules1, rules2, rules3, rules4, rules5, rules6, rules7, rules8, rules9, rules10])
		i+=1

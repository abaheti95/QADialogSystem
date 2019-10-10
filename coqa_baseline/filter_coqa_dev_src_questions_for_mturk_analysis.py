# We know that our model is incapable of handling single word questions and yes/no questions
# We will remove such questions from the src file and find the percentage of such questions

# CoQA gold answers filtering
src_file = "src_coqa_dev_data.txt"
src_save_file = "src_coqa_dev_data_filtered.txt"

# CoQA predictions filtering
src_predictions_file = "src_coqa_dev_data_predictions.txt"
src_predictions_save_file = "src_coqa_dev_data_predictions_filtered.txt"

answer_length_issue_count = 0.0
single_word_question = 0.0
yes_no_question = 0.0
unknown_question = 0.0
no_wh_question = 0.0
saved_question = 0.0
total_questions = 0.0
wh_words = [" what ", " who ", " whom ", " whose ", " when ", " where ", " which ", " why ", " how "]
print(src_file)
print(src_save_file)
with open(src_file, "r") as reader, open(src_predictions_file, "r") as reader2, open(src_save_file, "w") as writer, open(src_predictions_save_file, "w") as writer2:
	for line, line2 in zip(reader, reader2):
		q, a = line.strip().split(" ||| ")
		q2, a2 = line2.strip().split(" ||| ")
		if q != q2:
			print("WTF")
			print(q)
			print(q2)
			exit()
		if len(a.strip().split()) > 5:
			answer_length_issue_count += 1.0
		elif len(q.replace("?", "").strip().split()) == 1:
			single_word_question += 1.0
		elif a in ["yes", "no"]:
			yes_no_question += 1.0
		elif a == "unknown":
			unknown_question += 1.0
		else:
			# if no wh word present in question
			q_with_spaces = " " + q.strip() + " "
			flag = False
			for wh_word in wh_words:
				if wh_word in q_with_spaces:
					flag = True
					break
			if not flag:
				no_wh_question += 1.0
			else:
				# Save the question in the new src file
				saved_question += 1.0
				writer.write("{}\n".format(line.strip()))
				writer.flush()
				writer2.write("{}\n".format(line2.strip()))
				writer2.flush()
		total_questions += 1.0

print("answer_length_issue_count:", answer_length_issue_count / total_questions * 100.0)
print("single_word_question:", single_word_question / total_questions * 100.0)
print("yes_no_question:", yes_no_question / total_questions * 100.0)
print("no_wh_question:", no_wh_question / total_questions * 100.0)
print("unknown_question:", unknown_question / total_questions * 100.0)
print("saved_question:", saved_question / total_questions * 100.0)
print("total_questions:", total_questions)

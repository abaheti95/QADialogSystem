# We will run the lexparser command for all the squad seq2seq q files
import os
import re

dirname = "squad_seq2seq_train"

all_files = [filename for filename in os.listdir(dirname)]
# dictionary of ending with 
q_files = dict()
lexparsed_files = dict()
a_files = dict()
for filename in all_files:
	if filename.startswith("squad_train_q"):
		ending = filename.replace("squad_train_q", "")
		q_files[ending] = os.path.join(dirname, filename)
	if filename.startswith("squad_train_a"):
		ending = filename.replace("squad_train_a", "")
		a_files[ending] = os.path.join(dirname, filename)
	if filename.startswith("squad_train_lexparsed_q"):
		ending = filename.replace("squad_train_lexparsed_q", "")
		lexparsed_files[ending] = os.path.join(dirname, filename)

def count_lines(file):
	with open(file, "r") as reader:
		return sum(1 for line in reader)

count = total_count = 0
def find_problems_in_lexparser(lex_file, q_file):
	with open(lex_file, "r") as l_reader, open(q_file, "r") as q_reader:
		for lex_line, q in zip(l_reader, q_reader):
			words = re.findall(r"\s[\w\-\?\'\"\,\.\`]+\)", lex_line, re.U)
			line = ''.join(word[1:-1] for word in words)
			if line.strip() != q.replace(" ","").strip():
				print("Words", words)
				print("Lex line", lex_line)
				print("Q", q)
				print("squished line", line)
				print("squished q", q.replace(" ", ""))
				print("\n\n\n")
				exit()

for ending in q_files.keys():
	q_file = q_files[ending]
	lexparsed_file = lexparsed_files[ending]
	a_file = a_files[ending]
	# count lines of the files
	q_lines = count_lines(q_file)
	a_lines = count_lines(a_file)
	lexparsed_lines = count_lines(lexparsed_file)
	if lexparsed_lines != q_lines:
		count += 1
		print(q_lines, a_lines, lexparsed_lines, q_file, lexparsed_file)
		find_problems_in_lexparser(lexparsed_file, q_file)
	total_count += 1

print(count, total_count)

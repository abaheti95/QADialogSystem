import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
sys.path.append(os.path.join("/", "home", "baheti", "QADialogueSystem", "Data"))
from QA_datasets.quac_reader import *

from sacremoses import MosesTokenizer
mt = MosesTokenizer()

quad_dev = read_quac_dev()

count = total_count = 0
q_a_dict = dict()
for q,a in quad_dev:
	if a == "CANNOTANSWER":
		continue
	q_a_dict.setdefault(q, set())
	q_a_dict[q].add(a)

def find_shortest_answer_from_set(ans):
	shortest_a = None
	shortest_a_size = 100000
	for a in ans:
		a = mt.tokenize(a.lower().strip(), return_str=True, escape=False)
		if len(a.split()) < shortest_a_size:
			shortest_a_size = len(a.split())
			shortest_a = a
	return shortest_a

for q, anss in q_a_dict.items():
	shortest_a = find_shortest_answer_from_set(anss)
	total_count += 1
	a_tok = mt.tokenize(shortest_a.lower().strip(), return_str=True, escape=False)
	if len(a_tok.split()) <= 5:
		print(q, "::",shortest_a)
		count += 1
print(count, total_count)
# First we will try reading the quac data using the allennlp reader
from allennlp.data.dataset_readers.reading_comprehension import QuACReader
import os
import json
from allennlp.predictors import DialogQAPredictor
from allennlp.models.reading_comprehension import DialogQA
from allennlp.models import Model
from allennlp.common import Params
from allennlp.data.dataset_readers import DatasetReader

QUAC_train_file = os.path.join("data", "train_v0.2.json")
QUAC_val_file = os.path.join("data", "val_v0.2.json")

# quac_val_dataset = reader.read(QUAC_val_file)
# for instance in quac_val_dataset:
# 	print(instance)
# 	print(instance["metadata"]["answer_texts_list"])
# 	break

DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_train_moses_tokenized")
quac_format_test_save_file = os.path.join(DATA_DIR, "squad_seq2seq_predicted_responses_test_quac_format.txt")

DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "RuleBasedQuestionsToAnswer", "squad_seq2seq_dev_moses_tokenized")
quac_format_test_save_file = os.path.join(DATA_DIR, "squad_dev_test_quac_format.txt")
output_file = "quac_answers_on_squad_dev_test.txt"

# Load quac model here
config_file = os.path.join("models", "config.json")
with open(config_file) as f:
	model_config = Params(json.load(f))
serialization_dir = "models"
model_weights_file = os.path.join("models", "weights.th")
model = Model.load(config=model_config, serialization_dir=serialization_dir, weights_file=model_weights_file)
reader = DatasetReader.from_params(model_config["dataset_reader"])
print("model loaded")

# Initate the predictor
predictor = DialogQAPredictor(model, reader)

with open(quac_format_test_save_file, "r") as reader, open(output_file, "w") as writer:
	for i, jsonline in enumerate(reader):
		jsonline = jsonline.strip()
		predictions = predictor.predict(jsonline)
		instance = json.loads(jsonline)["paragraphs"][0]
		# print(predictions)
		q = instance["qas"][0]["question"]
		a = instance["qas"][0]["answers"][0]["text"]
		predicted_answer = predictions["best_span_str"]

		writer.write("{} ||| {}\n".format(q, a))
		writer.write("{}\n\n".format(predicted_answer))
		writer.flush()
		if i%10==0:
			print(i)
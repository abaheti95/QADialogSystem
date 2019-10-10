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

# Output on coqa dev data
DATA_DIR = os.path.join("/", "home", "baheti", "QADialogueSystem", "co-squac", "datasets", "converted")
coqa_dev_in_quac_format = os.path.join(DATA_DIR, "coqa_dev.json")
output_file = "quac_answers_on_coqa_dev_test.txt"

with open(coqa_dev_in_quac_format, "r") as reader, open(output_file, "w") as writer:
	data = json.load(reader)['data']
	for instance in data:
		# Add answer ends to every answer
		for qas in instance['paragraphs'][0]["qas"]:
			answers = qas["answers"]
			for answer in answers:
				text = answer['text']
				answer_start = answer["answer_start"]
				answer_end = answer_start + len(text.strip().split()) - 1
				answer.update({"answer_end": answer_end})
		predictions = predictor.predict(json.dumps(instance))
		print(predictions)
		print()
		print(instance)
		exit()
		instance = json.loads(jsonline)["paragraphs"][0]
		# print(predictions)
		q = instance["qas"][0]["question"]
		a = instance["qas"][0]["answers"][0]["text"]
		predicted_answer = predictions["best_span_str"]

		writer.write("{} ||| {}\n".format(q, a))
		writer.write("{}\n\n".format(predicted_answer))
		writer.flush()
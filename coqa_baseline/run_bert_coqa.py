# export CUDA_VISIBLE_DEVICES=3
from sogou_mrc.dataset.coqa import CoQAReader,CoQAEvaluator
from sogou_mrc.libraries.BertWrapper import BertDataHelper
from sogou_mrc.model.bert_coqa import BertCoQA
from sogou_mrc.data.vocabulary import  Vocabulary
import logging
import sys
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

coqa_reader = CoQAReader(-1)
data_folder= os.path.join("/", "home", "baheti", "QADialogueSystem", "Data", "QA_datasets", "coqa/")
train_filename = "coqa-train-v1.0.json"
eval_filename = "coqa-dev-v1.0.json"
vocab = Vocabulary(do_lowercase=True)
vocab_filepath = os.path.join("models", "vocab.txt")
if os.path.exists(vocab_filepath):
	# load from the filepath
	vocab.load(vocab_filepath)
else:
	train_data = coqa_reader.read(data_folder+train_filename, 'train')
	eval_data = coqa_reader.read(data_folder+eval_filename,'dev')
	vocab.build_vocab(train_data+eval_data)
	vocab.save(vocab_filepath)

evaluator = CoQAEvaluator(data_folder+eval_filename)
bert_dir = 'uncased_L-12_H-768_A-12'
bert_data_helper = BertDataHelper(bert_dir)
train_data = bert_data_helper.convert(train_data,data='coqa')
eval_data = bert_data_helper.convert(eval_data,data='coqa')



from sogou_mrc.data.batch_generator import BatchGenerator
my_batch_size = 6
train_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=my_batch_size,additional_fields=[
    'input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'
])
eval_batch_generator = BatchGenerator(vocab,eval_data,training=False,batch_size=my_batch_size,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'])

model = BertCoQA(bert_dir=bert_dir,answer_verification=True)
warmup_proportion = 0.1
num_train_steps = int(
        len(train_data) / my_batch_size * 2)
num_warmup_steps = int(num_train_steps * warmup_proportion)

# original paper adamax optimizer
model.compile(3e-5,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps)
model.train_and_evaluate(train_batch_generator, eval_batch_generator,evaluator, save_dir="models", epochs=2, eposides=1)
#model.evaluate(eval_batch_generator,evaluator)


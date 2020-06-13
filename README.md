# QADialogSystem
We design models that generate conversational responses for factual questions using expert answer phrases from Question Answering systems. [Paper: "Fluent Response Generation for Conversational Question Answering"](https://arxiv.org/pdf/2005.10464.pdf)

## Setup for CoQA baseline
- Install [`SMRCToolkit`](https://github.com/sogou/SMRCToolkit) for model code
- Run `python run_bert_coqa.py` to train the model
- Run `python evaluate_on_qa_generation_test_using_coqa.py` to get CoQA predictions on SQuAD Dev Test set

## Setup for QuAC baseline
- Download and extract [QuAC trained model](https://drive.google.com/file/d/1L9yHSShvg4TTWIiBWrLEpbooasZEEiLx/view?usp=sharing) inside the `"quac_baseline"` folder
- Run `quac_baseline.py` to extract quac model responses on `squad_dev_test`

## Setup for SQuAD baseline
- Clone [`bert`](https://github.com/google-research/bert) within `"squad_baseline"` folder.
- Checkout to specific commit by running `git checkout 88a817c37f788702a363ff935fd173b6dc6ac0d6`
- Refer to `model_training_commands.txt` inside `"bert"` folder for running instructions

## STs+BERT baseline
- The outputs of STs+BERT baseline predictions on SQuAD Dev Test set can be found in `mturk_evaluations/data2/bert_softmax_predictions_on_squad_dev_test_0_to_822.txt`

## Setup for Pointer generator models
- run `git clone https://github.com/OpenNMT/OpenNMT-py` to get `"OpenNMT-py"` folder within `"QADialogSystem"`.
- checkout to specific commit by running `git checkout 7f1fc81da864c465862f23e048802107ada714a8` from within the `"OpenNMT-py"` folder
To get the pretrained models
- `cd OpenNMT-py`
- Download [zip file containing saved PGN and PGN-pre model checkpoints](https://mega.nz/file/IBwTBTYI#wuazXq-kAEJKanp7LT3QgVGrB8-qqlouLpdQD0w3f1M)
- `unzip pgn_models.zip`
To re-train the models
- Extract [`opensub_qa_en`](https://s3.amazonaws.com/opennmt-trainingdata/opensub_qa_en.tgz) data in `"Data/Opensubtitles_qa"`
- run `preprocess_opensubtitles_qa.py` in `"Data/Opensubtitles_qa"` folder to moses tokenize the `opensub_qa_en` data.
- Follow the training commands in `all_final_model_training_and_testing_commands.txt`

## DialoGPT, GPT-2
Download and extract saved GPT-2, GPT-2-Pre and DGPT models in `"DialoGPT"` folder as follows:
- `cd DialoGPT`
- Download [zip file containing saved GPT-2, GPT-2-Pre and DGPT model checkpoints fine-tuned on SS and SS+ data]()
- `unzip gpt_and_dgpt_models.zip`
For instructions on how to run the models refer to `all_final_model_training_and_testing_commands.txt`

### TODO
- Add citation
- Add the link to the gpt-2 and dgpt checkpoints
- Add the instructions on how to generate STs + BERT baseline predictions on SQuAD Dev Test set
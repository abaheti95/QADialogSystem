# QADialogSystem
We design models that generate conversational responses for factual questions using expert answer phrases from Question Answering systems.

## Setup for Pointer generator models
- run `git clone https://github.com/OpenNMT/OpenNMT-py` to get `"OpenNMT-py"` folder within `"QADialogSystem"`.
- checkout to specific commit by running `git checkout 7f1fc81da864c465862f23e048802107ada714a8` from within the `"OpenNMT-py"` folder
- Extract [`opensub_qa_en`](https://s3.amazonaws.com/opennmt-trainingdata/opensub_qa_en.tgz) data in `"Data/Opensubtitles_qa"`
- run `preprocess_opensubtitles_qa.py` in `"Data/Opensubtitles_qa"` folder to moses tokenize the `opensub_qa_en` data.

## Setup for CoQA baseline
- Install [`SMRCToolkit`](https://github.com/sogou/SMRCToolkit) for model code
- Run `run_bert_coqa.py` to train the model
- Run `generate_coqa_model_predictions_on_coqa_dev.py` to get CoQA predictions on `squad_dev_test`.

## Setup for QuAC baseline
- Download and extract [QuAC trained model](https://drive.google.com/file/d/1L9yHSShvg4TTWIiBWrLEpbooasZEEiLx/view?usp=sharing) inside the `"quac_baseline"` folder
- Run `quac_baseline.py` to extract quac model responses on `squad_dev_test`

## Setup for SQuAD baseline
- Clone [`bert`](https://github.com/google-research/bert) within `"squad_baseline"` folder.
- Checkout to specific commit by running `git checkout 88a817c37f788702a363ff935fd173b6dc6ac0d6`
- Refer to `model_training_commands.txt` inside `"bert"` folder for running instructions

## DialoGPT, GPT-2 and PGN 
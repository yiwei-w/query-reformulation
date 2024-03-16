from datasets import load_dataset, Dataset
import os
import wandb
from transformers.integrations import WandbCallback
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import evaluate
import numpy as np



base_model = "flan-t5-small"
output_dir = f"./{base_model}_S2S_output"
save_dir = f"./{base_model}_final_model"

os.environ["WANDB_PROJECT"] = f"{base_model}-S2S"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")


dataset = load_dataset("json", data_files="./data_preparation/finetune_data.jsonl", split="train")

print(dataset)

# Split the dataset into train and test sets: 2% of the data will be used for testing
train_test_dataset = dataset.train_test_split(test_size=0.02)
raw_train_dataset = train_test_dataset["train"]
raw_test_dataset = train_test_dataset["test"]


PREFIX = "search: "
MAX_INPUT_LENGTH = 50
MAX_TARGET_LENGTH = 380

def preprocess_function(examples):
    inputs = [PREFIX + prompt for prompt in examples["prompt"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, add_special_tokens=True)

    # Setup the tokenizer for targets
    outputs = tokenizer(text_target=examples["completion"], max_length=MAX_TARGET_LENGTH, truncation=True, add_special_tokens=True)

    model_inputs["labels"] = outputs["input_ids"]
    return model_inputs

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # mask padding tokens
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE score
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Compute BLEU score
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result.update(bleu_result)


    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result["gen_len"] = np.mean(prediction_lens)

    return {k: v for k, v in result.items()}


collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(preprocess_function(raw_train_dataset[:2]))

train_dataset = raw_train_dataset.map(preprocess_function, batched=True)
test_dataset = raw_test_dataset.map(preprocess_function, batched=True)


training_args = Seq2SeqTrainingArguments(
    evaluation_strategy="epoch",
    overwrite_output_dir=True,
    output_dir=output_dir,
    bf16=True,
    learning_rate=1e-4,
    weight_decay=1e-4,
    optim="adamw_torch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,
    max_grad_norm=1.0,
    num_train_epochs=12,
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    eval_steps=5,
    save_steps=40,
    report_to="wandb",  # enable logging to W&B
    logging_steps=1,  # how often to log to W&B
    include_num_input_tokens_seen=True,
    disable_tqdm=False,
    save_only_model=True,
    predict_with_generate=True,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)



trainer.train()

trainer.save_model(save_dir)


wandb.finish()
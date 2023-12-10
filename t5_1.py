


from datasets import load_dataset
dataset = load_dataset("multi_news")
print(f"Features: {dataset['train'].column_names}")
...
Features: ['document', 'summary']

sample = dataset["train"][1]
print(f"""Document (excerpt of 2000 characters, total length: {len(sample["document"])}):""")
print(sample["document"][:2000])
print(f'\nSummary (length: {len(sample["summary"])}):')
print(sample["summary"])

from transformers import BartForConditionalGeneration, AutoTokenizer
model_ckpt = "sshleifer/distilbart-cnn-6-6"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BartForConditionalGeneration.from_pretrained(model_ckpt)

d_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["document"]]
s_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["summary"]]

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["document"], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=256, truncation=True)

    return {"input_ids": input_encodings["input_ids"],
           "attention_mask": input_encodings["attention_mask"],
           "labels": target_encodings["input_ids"]}
dataset_pt = dataset.map(convert_examples_to_features, batched=True)

from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(output_dir='bart-multi-news', num_train_epochs=1, warmup_steps=500,                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, weight_decay=0.01, logging_steps=10, push_to_hub=False,
evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
gradient_accumulation_steps=16)

# pip show accelerate


trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,data_collator=seq2seq_data_collator,train_dataset=dataset_pt["train"],eval_dataset=dataset_pt["validation"])
trainer.train()

sample_text = dataset["test"][1]["document"]
reference = dataset["test"][1]["summary"]

input_ids = tokenizer(sample_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt').to(device)

summaries = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],                           max_length=256)

decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
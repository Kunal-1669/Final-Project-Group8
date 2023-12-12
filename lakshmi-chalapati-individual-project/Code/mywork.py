import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Multi-News dataset
dataset = load_dataset("multi_news")

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)  # Move model to the device


# Preprocess the data
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")

    # Convert inputs and labels to tensors and move them to the device
    model_inputs = {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}
    model_inputs["labels"] = torch.tensor(labels["input_ids"]).to(device)

    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()


# Example usage for summarization
def summarize(text, max_length=150):
    input_ids = tokenizer("summarize: " + text, return_tensors="pt").input_ids.to(device)  # Move to device
    summary_ids = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Summarize a document
#document = input("Enter your text: ")
#summary = summarize(document)
#print('\nSummary for the given news article:\n',summary)




from rouge import Rouge
# Calculate ROUGE score
def calculate_rouge_score(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores
# Summarize a document and calculate ROUGE score
document = input("Enter your text: ")
reference_summary = input("Enter the reference summary: ")  # You need a reference summary to compare against
generated_summary = summarize(document)
print('\nGenerated summary:\n', generated_summary)
# Calculate and print the ROUGE score
rouge_score = calculate_rouge_score(generated_summary, reference_summary)
print("\nROUGE Score:\n", rouge_score)
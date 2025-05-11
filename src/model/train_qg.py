# from datasets import load_dataset
# from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding


# import torch
# import nltk

# # Download the punkt tokenizer models for nltk
# nltk.download('punkt')

# # Load the SQuAD dataset
# dataset = load_dataset("squad")

# # Initialize the tokenizer for the T5 model
# tokenizer = T5Tokenizer.from_pretrained("t5-small")

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # Preprocessing function to tokenize the input and output text
# # def preprocess_function(examples):
# #     # Tokenize the context and question
# #     inputs = tokenizer(examples['context'], padding=True, truncation=True, max_length=512, return_tensors="pt")
# #     labels = tokenizer(examples['question'], padding=True, truncation=True, max_length=128, return_tensors="pt")
    
# #     # We use input_ids for the inputs, and labels for the outputs (questions)
# #     return {
# #         'input_ids': inputs['input_ids'].squeeze(),  # Remove unnecessary batch dimension
# #         'labels': labels['input_ids'].squeeze()
# #     }

# def preprocess_function(examples):
#     # Tokenize inputs (context) and targets (question)
#     inputs = tokenizer(
#         examples['context'], 
#         padding="max_length", 
#         truncation=True, 
#         max_length=512
#     )
#     targets = tokenizer(
#         examples['question'], 
#         padding="max_length", 
#         truncation=True, 
#         max_length=128
#     )

#     # Use `input_ids` and `attention_mask` from inputs, and `input_ids` as labels
#     inputs["labels"] = targets["input_ids"]
#     return inputs


# # Apply preprocessing to the dataset
# dataset = dataset.map(preprocess_function, batched=True)

# # Load the pre-trained T5 model
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# training_args = TrainingArguments(
#     output_dir="src/model/saved_model",          # Output directory for saved model
#     eval_strategy="epoch",                 # Evaluate after each epoch
#     learning_rate=2e-5,                          # Learning rate
#     per_device_train_batch_size=8,               # Batch size for training
#     per_device_eval_batch_size=8,                # Batch size for evaluation
#     num_train_epochs=3,                          # Number of training epochs
#     weight_decay=0.01,                           # Weight decay for regularization
#     save_steps=500,                              # Save model checkpoint every 500 steps
#     save_total_limit=2,                          # Keep only the last 2 checkpoints
#     logging_dir="src/model/logs",                        # Logging directory for TensorBoard
# )

# trainer = Trainer(
#     model=model,                        # The pre-trained model
#     args=training_args,                 # The training arguments
#     train_dataset=dataset['train'],     # Training dataset
#     eval_dataset=dataset['validation'],# Evaluation dataset
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# trainer.train()

# # Evaluate the model on the validation dataset
# eval_results = trainer.evaluate()
# print(eval_results)

# # Save the model and tokenizer
# model.save_pretrained("src/model/saved_model")
# tokenizer.save_pretrained("src/model/saved_model")


# # Define a function to generate a question from a context
# def generate_question(context):
#     # Tokenize the input context
#     input_ids = tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
    
#     # Generate the question (limit the length of the generated text)
#     output = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    
#     # Decode the output to text
#     question = tokenizer.decode(output[0], skip_special_tokens=True)
#     return question

# # Example context to generate a question
# context = "42 is the answer to life, the universe, and everything."
# question = generate_question(context)
# print(question)


# -------------------------

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import nltk
import os

# Download the punkt tokenizer models for nltk
nltk.download('punkt')

# Load the SQuAD dataset
dataset = load_dataset("squad")

# Initialize the tokenizer for the T5 model
tokenizer = T5Tokenizer.from_pretrained("t5-small")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    # Tokenize inputs (context) and targets (question)
    inputs = tokenizer(
        examples['context'], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    targets = tokenizer(
        examples['question'], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

    # Use `input_ids` and `attention_mask` from inputs, and `input_ids` as labels
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_function, batched=True)

# # Load the pre-trained T5 model or resume from checkpoint if available
# model_dir = "src/model/saved_model"
# if os.path.exists(model_dir):
#     model = T5ForConditionalGeneration.from_pretrained(model_dir)
# else:
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")

model_dir = "src/model/saved_model"

# Find the latest checkpoint directory (e.g., checkpoint-1000)
checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
latest_checkpoint_dir = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint_dir)

# # Load the model from the latest checkpoint
# model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint_path)

# training_args = TrainingArguments(
#     output_dir=model_dir,           # Output directory for saved model
#     eval_strategy="epoch",          # Evaluate after each epoch
#     learning_rate=2e-5,             # Learning rate
#     per_device_train_batch_size=8,  # Batch size for training
#     per_device_eval_batch_size=8,   # Batch size for evaluation
#     num_train_epochs=3,             # Number of training epochs
#     weight_decay=0.01,              # Weight decay for regularization
#     save_steps=500,                 # Save model checkpoint every 500 steps
#     save_total_limit=2,             # Keep only the last 2 checkpoints
#     logging_dir="src/model/logs",   # Logging directory for TensorBoard
#     resume_from_checkpoint=True    # Resume from the last checkpoint if available
# )

# trainer = Trainer(
#     model=model,                    # The pre-trained model (or checkpoint)
#     args=training_args,             # The training arguments
#     train_dataset=dataset['train'], # Training dataset
#     eval_dataset=dataset['validation'], # Evaluation dataset
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# trainer.train()

# # Evaluate the model on the validation dataset
# eval_results = trainer.evaluate()
# print(eval_results)

# # Save the model and tokenizer
# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)

# Initialize the model with the checkpoint
model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint_path)

# Initialize training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    eval_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500, 
    save_total_limit=2,
    logging_dir="src/model/logs",
    resume_from_checkpoint=True  # Ensures checkpoint is respected
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Print checkpoint loading info
print(f"Resuming training from checkpoint: {latest_checkpoint_path}")

# Resume training from the latest checkpoint
trainer.train(resume_from_checkpoint=latest_checkpoint_path)

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the final model and tokenizer
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)


# Define a function to generate a question from a context
# def generate_question(context):
#     # Tokenize the input context
#     input_ids = tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
    
#     # Generate the question (limit the length of the generated text)
#     output = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    
#     # Decode the output to text
#     question = tokenizer.decode(output[0], skip_special_tokens=True)
#     return question

def generate_question(context):
    input_ids = tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True).to("cpu")
    model_cpu = model.to("cpu")  # move model to CPU
    output = model_cpu.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question


# Example context to generate a question
context = "42 is the answer to life, the universe, and everything."
question = generate_question(context)
print(question)

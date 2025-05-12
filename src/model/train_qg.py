import os
import nltk
import torch

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

nltk.download('punkt')

dataset = load_dataset("squad")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    """
    Tokenize input text (context) and target text (question) for training.

    Args:
        examples (dict): A dictionary with 'context' and 'question' fields from the dataset.

    Returns:
        dict: Tokenized inputs including input IDs, attention masks, and labels.
    """
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

    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess_function, batched=True)

model_dir = "src/model/saved_model"

checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
latest_checkpoint_dir = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
latest_checkpoint_path = os.path.join(model_dir, latest_checkpoint_dir)

model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint_path)

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
    resume_from_checkpoint=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

print(f"Resuming training from checkpoint: {latest_checkpoint_path}")

trainer.train(resume_from_checkpoint=latest_checkpoint_path)

eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)


def generate_question(context):
    """
    Generate a question based on the given context using a pre-trained model.

    Args:
        context (str): The input text or passage from which to generate a question.

    Returns:
        str: A generated question based on the input context.
    """
    input_ids = tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True).to("cpu")
    model_cpu = model.to("cpu")
    output = model_cpu.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question
import nltk
import torch

from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer


nltk.download("punkt")

model_dir = "src/model/saved_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()

def generate_t5_questions(text: str) -> list:
    """
    Generate a list of questions from the input text using a T5 model.

    Args:
        text (str): The raw input text to process.

    Returns:
        list: A list of generated questions, one per sentence if applicable.
    """
    sentences = sent_tokenize(text)
    questions = []

    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        question = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if question.endswith("?"):
            questions.append(question)

    return questions
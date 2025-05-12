from src.utils.t5 import generate_t5_questions

def generate_question(text: str) -> list:
    return generate_t5_questions(text)
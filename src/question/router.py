from fastapi import Form, Request
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Form, Request

from src.templates_engine import templates
from src.question.service import generate_question
from src.question.service import generate_question


question_router = APIRouter()


@question_router.post("/generate/", response_class=HTMLResponse)
async def generate(request: Request, input_text: str = Form(...)):
    """
    Handle POST request to generate questions from input text.

    Args:
        request (Request): The incoming HTTP request object.
        input_text (str): The text input submitted via the form.

    Returns:
        TemplateResponse: Rendered HTML template with the original input and generated questions.
    """
    questions = generate_question(input_text)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "input_text": input_text,
        "questions": questions
    })
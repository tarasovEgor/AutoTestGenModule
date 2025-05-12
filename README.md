
# AutoTestGenModule

A web application designed to automate the process of generating questions based on text analysis.
It leverages modern NLP techniques and a pre-trained T5 language model to analyze textual content and create relevant questions that can be used for educational, testing, or evaluation purposes.

---

## 📌 Requirements

- Python 3.11.8
- Poetry (for dependency management)

---

## 🛠️ Installation & Setup
1. **Clone the repository:**
```git clone <repository-url>```
```cd AutoTestGenModule```

2. **Install project dependencies:**
```poetry install```

3. **Activate the virtual environment:**
```poetry shell```

---

## 🚀 Running the Application

1. **Once all dependencies are installed and the virtual environment is activated, start the FastAPI app with:**
```poetry run uvicorn src.main:app --reload```

2. **The application will be available at:**
```http://localhost:8000```

---

## 📚 Usage

1. **Navigate to** ```http://localhost:8000```
2. **Enter or paste your text input into the form**
3. **Submit to generate relevant questions based on the text**
4. **View results rendered dynamically in the interface**

---
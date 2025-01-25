"""Load all the questions from different datasets, rephrase questions if needed, extract crucial entities and structure
question-answer-entity triplets."""

import os

import openai
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # OpenAI API key


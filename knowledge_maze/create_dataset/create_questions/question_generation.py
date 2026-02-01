import json
import random
import pandas as pd
from IPython.display import display

from knowledge_maze.tools.api_tools import *

QUESTION_PROMPT = """..."""
QUESTION_TYPES = {}

class QuestionGenerator:

    def __init__(self, input_tables, question_prompt=QUESTION_PROMPT):

        self.input_tables = input_tables
        self.model = gptAPI(question_prompt,
                            output_format='string',
                            temperature=0.7,
                            max_concurrent_requests=100,
                            max_rpm=4900)
        self.stats = {
            "question_candidates": len(self.input_tables),
            "questions_generated": 0,
        }
        self.questions = []
        self.bad_examples = []

    def generate_questions(self):
        pass

    def get_answer(self):
        # Get answers and evidences from the data
        pass

    def structure_data(self):
        pass

    def _print_summary(self):
        pass

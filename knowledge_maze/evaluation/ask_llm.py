"""Get the question and context, then answer using LLM of our choosing

- get full context or just reasoning path 1/2 + make random order for the paragraphs.
- ask a question and get answer
"""
from knowledge_maze.tools.api_tools import *

# Initialize LLM prompts:
ASK_QUESTION_PROMPT = """"""
LLM_JUDGE_PROMPT = """You are a strict evaluator. Given a question and two possible answers (Option A and Option B), 
                      compare a reference answer to them. Return only `True` if the reference answer best matches 
                      Option A, and `False` otherwise. Respond only with `True` or `False`."""

# Create LLM API call objects:
qa_model = gptAPI(ASK_QUESTION_PROMPT,
                  output_format='list',
                  temperature = 0.9,
                  max_concurrent_requests = 100,
                  max_rpm=4900)
judge_model = gptAPI(LLM_JUDGE_PROMPT,
                     output_format='list',
                     temperature = 0.2,
                     max_concurrent_requests = 100,
                     max_rpm=4900)


def judge_answer(answer: str, options: list) -> bool:
    pass


def ask_llm(question: str, context: str) -> str:
    pass


def get_context():
    pass

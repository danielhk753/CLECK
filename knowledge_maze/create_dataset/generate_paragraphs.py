"""
Generate paragraphs

Before we start:
- How to generate a counter answer? Per answer type?
- What is the paragraphs styles I want to create?
"""

from knowledge_maze.tools.api_tools import *

# Initialize LLM prompts:
GENERATE_ANSWER_PROMPT = """You are an expert at generating short and plausible counter-answers to questions. 
                            Your task is to provide alternative answers that are incorrect but appear valid and 
                            believable. Ensure the counter-answers are concise, contextually relevant to the question, 
                            and different from the correct answer."""
GENERATE_PARAGRAPH_PROMPT = """"""

# Create LLM API call objects:
answer_generator = gptAPI(GENERATE_ANSWER_PROMPT,
                          temperature = 0.2,
                          max_concurrent_requests = 100)
paragraph_generator = gptAPI(GENERATE_PARAGRAPH_PROMPT,
                             temperature = 0.5,
                             max_concurrent_requests = 100)


def entity_counter_answer(question: list, correct_answer: str, num_answers: int = 3) -> str:
    """  """
    pass


def numerical_counter_answer(question: str, answer: int, answer_range: tuple[int]) -> list[int]:
    pass



def generate_paragraph(question_1: str, question_2: str, answer: str|int, style: str) -> str:
    pass

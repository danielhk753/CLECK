"""
Generate paragraphs

"""
from knowledge_maze.tools.api_tools import *


# Initialize LLM prompts:
MIDDLE_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that conveys, without explicitly 
                             stating, that the following two questions share the same answer. The paragraph should 
                             be self-contained, meaning the reader should be able to infer that both questions have 
                             the same answer without requiring external knowledge. Do not mention the questions 
                             explicitly—rather, incorporate their key details naturally within the paragraph. 
                             *Crucially, do not include the answer itself anywhere in the paragraph*.\n\n
                             Ensure the paragraph flows naturally, avoids unnatural phrasing, and maintains a 
                             coherent narrative. The writing should be engaging while subtly guiding the reader 
                             toward recognizing the shared answer."""

ANSWER_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that answers the given question 
                             with the given answer. The paragraph should be self-contained, meaning the reader should 
                             be able to infer that both questions have the same answer without requiring external 
                             knowledge. Do not mention the questions explicitly—rather, incorporate their key details 
                             naturally within the paragraph. \n\n
                             Ensure the paragraph flows naturally, avoids unnatural phrasing, and maintains a 
                             coherent narrative. The writing should be engaging while subtly guiding the reader 
                             toward recognizing the shared answer."""

# Create LLM API call objects:
middle_paragraph_generator = gptAPI(MIDDLE_PARAGRAPH_PROMPT,
                                    temperature = 0.5,
                                    max_concurrent_requests = 100,
                                    max_rpm=4900)
edge_paragraph_generator = gptAPI(ANSWER_PARAGRAPH_PROMPT,
                                  temperature = 0.5,
                                  max_concurrent_requests = 100,
                                  max_rpm=4900)


async def generate_paragraph(inputs: list[str], mode: str) -> list:
    if mode == 'Chain':
        return await middle_paragraph_generator.async_run(inputs)
    if mode == 'Edge':
        return await edge_paragraph_generator.async_run(inputs)
    raise AttributeError
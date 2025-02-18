""""""

import os
import re
import string

import openai
from typing import List
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from dataset_creation.api_tools import gpt_api

QUESTION_WORDS = ['Who', 'Which', 'How', 'Why', 'When', 'Where']

# Load OpenAI API key from .env file:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


# Structured string response for LLM outputs:
class StringResponse(BaseModel):
    Text: str


# Structured list of strings response for LLM outputs:
class ListResponse(BaseModel):
    Text: List[str]


OUTPUT_FORMATS = {'string': StringResponse,
                  'list': ListResponse}


def call_gpt4o_mini(system_prompt: str, user_prompt: str, ouput_fotmat: str = 'string'):
    client = OpenAI()

    extraction = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user",
             "content": user_prompt}
        ],
        response_format=OUTPUT_FORMATS[ouput_fotmat],
        temperature=0.2
    )

    return extraction.choices[0].message.parsed.Text


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def get_candidate_entities(question: str, dataset: list[dict]):
    candidate_entities = set()
    question = question

    for sample in dataset:
        answer = sample['Answer']
        if answer in question and answer not in QUESTION_WORDS and len(answer) > 1:
            if str(' ' + answer + ' ') in remove_punctuation(question):
                candidate_entities.add(sample['Answer'])

    return list(candidate_entities)


def remove_ent_duplicates(entity_candidates: list):
    # Sort entities by length (longest first) to ensure we keep the most complete version
    entities = sorted(entity_candidates, key=len, reverse=True)
    unique_entities = []

    for s in entities:
        if not any(s in other for other in unique_entities):
            unique_entities.append(s)

    return unique_entities


def replace_word(sentence: str, word_to_replace: str):
    """
    Replaces the first occurrence of a word in a sentence with '_____'.

    Args:
        sentence (str): The input sentence.
        word_to_replace (str): The word to be replaced.

    Returns:
        str: The sentence with the word replaced.
    """
    # word_length = sum([len(word) for word in word_to_replace.split()])
    word_length = len(word_to_replace)
    return sentence.replace(word_to_replace, word_length * "_", 1)  # Replace only the first occurrence


def replace_placeholder(sentence, word):
    # Use a regular expression to replace any sequence of underscores with the given word
    return re.sub(r'_+', word, sentence)


def frame_entity(question: str, entity: str):
    system_prompt = """Task: Given a sentence and a word/sequence of words that appear in the sentence, extract the 
                    longest meaningful subsequence where the word is part of an entity. \n
                    
                    Examples:\n
                    ---------\n
                    Input:\n
                    Sentence: Where in England was Dame Judi Dench born?\n
                    Word: Judi Dench\n
                    
                    Output:\n
                    Dame Judi Dench\n
                    ---------\n
                    Input:\n
                    Sentence: Which Oscar-winning actress was born on exactly the same day as actress Lindsay Wagner?\n
                    Word: Lindsay\n
                    
                    Output:\n
                    actress Lindsay Wagner\n
                    ---------\n
                    Input:\n
                    Sentence: The Eiffel Tower is one of the most famous landmarks in France.\n
                    Word: Eiffel Tower\n
                    
                    Output:\n
                    The Eiffel Tower\n
                    ---------\n
                    Input:\n
                    Sentence: President Abraham Lincoln delivered the Gettysburg Address in 1863.\n
                    Word: Abraham\n
                    
                    Output:\n
                    President Abraham Lincoln"""

    user_prompt = f"""Now, given the following input, extract the correct entity: \n

                Input:\n
                Sentence: {question}\n
                Word: {entity}"""

    return call_gpt4o_mini(system_prompt, user_prompt)


def fill_the_blanks(question: str):
    system_prompt = """Task: Provide a relevant option for the blank in the given question. 
                    The options should fit naturally and maintain the question's meaning and intent. \n
                    \n
                    Example Input #1:\n
                    "Who directed _____?"\n
                    \n
                    Example Outputs:\n
                    - Inception\n
                    - Pulp Fiction\n
                    
                    Example Input #2:\n
                    "Which actor had a _____ called Kirk?"\n
                    \n
                    Example Outputs:\n
                    - Son\n
                    - Father
                    """

    user_prompt = f"""Now is your turn, provide a relevant option for the following question: \n
                  Question: {question}"""

    return call_gpt4o_mini(system_prompt, user_prompt)


def shared_words_percentage(sentence1, sentence2):
    # Convert both sentences to lowercase and split them into sets of words
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())

    # Find the intersection of the two sets (shared words)
    shared_words = words1.intersection(words2)

    # Calculate the percentage of shared words
    total_words = len(words1.union(words2))  # Union of words from both sentences
    percentage = (len(shared_words) / total_words) if total_words > 0 else 0

    return percentage


def answer_question(question: str):
    system_prompt = """You are a trivia expert, answer the following question as short as you can. \n
                    Examples:
                    Who won Super Bowl XX? --> Chicago Bears
                    What is Bruce Willis' real first name? --> Walter
                    Which city does David Soul come from? --> Chicago
                    """

    user_prompt = f"""Question: {question}"""

    return call_gpt4o_mini(system_prompt, user_prompt)


def entity_extraction(sentence: str):
    system_prompt = """Extract named entities (full names of people and place names like countries, cities, etc.) from 
                    the given sentence. For each entity, return a tuple (start_index, end_index), where: \n
                    \n
                    - start_index is the position of the first word in the entity (starting from 0).\n
                    - end_index is the position of the last word in the entity. \n
                    - If a place consists of multiple words (e.g., "New York," "York, England"), treat it as a single 
                    entity. \n
                    \n
                    Only include full names of people and geographical locations (e.g., countries, cities, regions). 
                    Ignore other types of entities.\n
                    \n
                    Example Input:\n
                    "Dame Judi Dench was born in York, England and later visited Paris."\n
                    \n
                    Example Output:\n
                    [(0, 2), (5, 7), (10, 10)]\n
                    \n
                    (In this case: "Dame Judi Dench" → (0,2), "York, England" → (5,7), "Paris" → (10,10))\n
                    """

    user_prompt = f"""Sentence: {sentence}\n
                  Output: """

    return call_gpt4o_mini(system_prompt, user_prompt, ouput_fotmat='list')


def get_entities(question: str):
    system_prompt = """Given a question about a specific topic (e.g., history, geography, people, etc.), extract key 
                    entities that are essential to fully answer the question. These entities should be one of the 
                    followings::\n
                    \n
                    1. Full names of people,\n
                    2. Country or place names,\n
                    3. Organization names,\n
                    4. Specific item names or commodities (e.g., products, events, etc.),\n
                    5. **Not** the answer itself.\n
                    \n
                    The goal is to identify what can't be removed from the answer because it won't retain enough 
                    information to correctly respond to the question. Return a list of these entities.\n
                    \n
                    Example Input:\n
                    "What country imports river sand from Scotland for construction and camels from North Africa?"\n
                    \n
                    Example Output:\n 
                    - Scotland\n
                    - North Africa\n
                    \n
                    The list of entities is crucial to answering the question and should provide context about the who, 
                    what, or where involved."""
    gpt = gpt_api(system_prompt=system_prompt, max_concurrent_requests=25, output_format='list')
    return gpt.run(question)


def rephrase_question(question: str):
    system_prompt = """rephrase the question by removing as many words as you can, while keep the question's meaning 
                    and answer the same."""
    gpt = gpt_api(system_prompt=system_prompt, max_concurrent_requests=25)
    return gpt.run(question)


def create_question(context, answer, entities):
    system_prompt = """Your task is to generate a short, simple, and unique question based on the given input text. The 
                       question must explicitly mention and be centered around one of the entities provided in the 
                       entity list.\n\n
                       
                       **Rules:**\n 
                        1. **Mandatory Entity Inclusion**: The question **must contain** one of the entities exactly as 
                        provided in the list.\n
                        2. **Context-Based Question**: The question should be meaningful and answerable using the 
                        provided context.\n
                        3. **Answer Alignment**: The correct answer to your question should be the one given in the 
                        input.\n
                        4. **Conciseness & Clarity**: Keep the question **short, simple, and direct** while ensuring it 
                        remains natural and informative.\n
                        5. **The answer to the question Can't be in the text of the question.**\n
                        6. **Unique**: The answer to the question shouldn't be common knowledge.\n\n

                        If no entity from the provided list is included in your question, your output is invalid."""

    gpt = gpt_api(system_prompt=system_prompt, max_concurrent_requests=25)
    api_input = f"""Question context: {context}\n
                    Answer: {answer}\n
                    Available entities: {entities}"""
    return gpt.run(api_input)

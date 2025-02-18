"""Load all the questions from different datasets, rephrase questions if needed, extract crucial entities and structure
question-answer-entity triplets."""

import os

import openai
from typing import List
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from dataset_creation.get_datasets import get_datasets

# Load OpenAI API key from .env file:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define datasets paths:
datasets = {
    'TriviaQA': "C:/Users/danie/Desktop/triviaqa-unfiltered/triviaqa-unfiltered.json",
    'Zero-Shot RE': None,
    'SQuAD': None,
    'Natural Questions': None,
    'MLQA': None,
    'T-REx': None
}


# Structured string response for LLM outputs:
class StringResponse(BaseModel):
    Text: str


# Structured boolean response for LLM outputs:
class BoolResponse(BaseModel):
    Ans: bool


# Structured list of strings response for LLM outputs:
class ListResponse(BaseModel):
    Options: List[str]


def generate_counter_answers(question, correct_answer, num_answers=3):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 "You are an expert at generating short and plausible counter-answers to questions. "
                 "Your task is to provide alternative answers that are incorrect but appear valid and believable. "
                 "Ensure the counter-answers are concise, contextually relevant to the question, and different from "
                 "the correct answer."
             )},
            {"role": "user",
             "content": (
                 f"Question: {question}\n"
                 f"Correct Answer: {correct_answer}\n"
                 f"Generate {num_answers} short, believable, and incorrect counter-answers. "
                 f"Each answer should be distinct and plausible within the context of the question."
             )}
        ],
        response_format=ListResponse
    )
    return completion.choices[0].message.parsed.Options


def extract_ent(question):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 f"""Extract all Wikipedia entities from the following sentence: \n
                 {question} \n
                 Return the entities in a list.
                 """

             )},
            {"role": "user",
             "content": (
                 f"Entities: "
             )}
        ],
        response_format=ListResponse
    )

    return completion.choices[0].message.parsed.Options


def validate_ent(question, ent):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 f"""Is the following word is an entity with respect to the question sentence? \n
                 Word: {ent} \n
                 Question: {question} \n
                 """

             )},
            {"role": "user",
             "content": (
                 f"Answer: "
             )}
        ],
        response_format=BoolResponse
    )

    return completion.choices[0].message.parsed.Ans


def check_if_crucial(question, entity):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 f"""Is the following entity is essential information to the question? Meaning that if I erase it from 
                 the question, will it make the question ambiguous or have a different answer?\n
                     Entity: {entity} \n
                     Question: {question} \n
                     """

             )},
            {"role": "user",
             "content": (
                 f"Answer: "
             )}
        ],
        response_format=BoolResponse
    )

    return completion.choices[0].message.parsed.Ans


def check_interpretation(entity):
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 f"""Is the following entity can refer to more than one entity?\n
                     Entity: {entity} \n
                     """

             )},
            {"role": "user",
             "content": (
                 f"Answer: "
             )}
        ],
        response_format=BoolResponse
    )

    return completion.choices[0].message.parsed.Ans


def extract_question_structure(question):
    client = OpenAI()

    extraction = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 "You are an advanced linguistic analyzer. Extract the following from a question:\n"
                 "- **Subject**: The main entity performing the action (if present).\n"
                 "- **Predicate**: The main verb or verb phrase describing the action.\n"
                 "- **Object**: The entity receiving the action (if present).\n"
                 "Do **not** include question words like Who, What, Where, etc. If an element is missing, "
                 "return 'None'.\n "
                 "Respond in JSON format:\n"
                 "{\n"
                 '  "subject": "<extracted subject or None>",\n'
                 '  "predicate": "<extracted predicate or None>",\n'
                 '  "object": "<extracted object or None>"\n'
                 "}"
             )},
            {"role": "user", "content": f"Extract subject, predicate, and object from: {question}"}
        ],
        response_format=StringResponse
    )

    return extraction.choices[0].message.parsed.Text


def determine_crucial(question):
    client = OpenAI()

    extraction = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 """
                 You are an expert in question analysis and reasoning. Your task is to determine whether a question with
                  a blank ("_____") can still be answered correctly if the blank is removed. If removing the blank 
                  **creates multiple valid answers or ambiguity**, the blank is **CRUCIAL**. If removing the blank 
                  **does not change the answer or makes the blank redundant**, then the blank is **IRRELEVANT**.

                ---
                
                ### **Instructions**
                1. **If the blank is crucial**, respond: `"CRUCIAL"` and explain why the missing word changes the meaning.
                2. **If the blank is irrelevant**, respond: `"IRRELEVANT"` and specify why:
                   - `"IRRELEVANT (Same Answer)"` → The blank does not change the final answer.
                   - `"IRRELEVANT (Redundant)"` → The blank is unnecessary because the question already provides enough context.
                
                3. **Use step-by-step reasoning (Chain-of-Thought) to justify your decision.**
                
                ---
                
                ### **Examples**
                
                #### **Example 1: Irrelevant (Same Answer)**
                **Question:**  
                "Who painted the Mona Lisa in the Renaissance and _____ in the 21st century?"  
                
                **Reasoning:**  
                - Possible blanks: "who painted a famous portrait", "who painted another masterpiece".
                - Regardless of the blank, the answer to the first part is **Leonardo da Vinci**.
                - The second part might change based on time period, but the first answer remains fixed.
                - Since the blank does not introduce ambiguity, **it is irrelevant**.
                
                **Final Answer:**  
                **"IRRELEVANT (Same Answer)"**  
                
                ---
                
                #### **Example 2: Irrelevant (Redundant)**  
                **Question:**  
                "Who wrote *The Turn of the Screw* in the 19th century and _____ in the 20th?"  
                
                **Reasoning:**  
                - Possible blanks: "what did he write", "another famous work".
                - Since the first clause already **specifies Henry James and his timeline**, the blank is **unnecessary**.
                - The question already provides enough structure, making the blank redundant.
                
                **Final Answer:**  
                **"IRRELEVANT (Redundant)"**  
                
                ---
                
                #### **Example 3: Crucial (Different Possible Answers)**  
                **Question:**  
                "What did Newton discover about _____?"  
                
                **Reasoning:**  
                - Possible blanks: "gravity", "light", "motion".
                - The answer changes completely depending on the blank:
                  - "gravity" → **Law of Universal Gravitation**.
                  - "light" → **Theory of Color & Optics**.
                  - "motion" → **Three Laws of Motion**.
                - Since multiple correct answers exist, **the blank is crucial**.
                
                **Final Answer:**  
                **"CRUCIAL"**  
                
                ---
                
                #### **Example 4: Crucial (Context Determines Answer)**  
                **Question:**  
                "How did the Treaty of Versailles impact _____?"  
                
                **Reasoning:**  
                - Possible blanks: "Germany", "the global economy", "World War II".
                - Each blank leads to **entirely different answers**:
                  - "Germany" → **Reparations & Economic Collapse**.
                  - "Global economy" → **Great Depression Link**.
                  - "World War II" → **Rise of Nazism & Conflict Escalation**.
                - Since the meaning **depends entirely on the blank**, it is crucial.
                
                **Final Answer:**  
                **"CRUCIAL"**  
                
                ---
                
                #### **Example 5: Irrelevant (Same Answer)**  
                **Question:**  
                "Which scientist formulated the theory of relativity and _____ the photoelectric effect?"  
                
                **Reasoning:**  
                - Possible blanks: "discovered", "studied", "explained".
                - Regardless of the blank, **the answer remains Albert Einstein**.
                - The blank **does not affect the outcome**.
                
                **Final Answer:**  
                **"IRRELEVANT (Same Answer)"**  
                
                ---
                
                #### **Example 6: Crucial (Different Interpretations)**  
                **Question:**  
                "Why did the American Revolution start in _____?"  
                
                **Reasoning:**  
                - Possible blanks: "1775", "Massachusetts", "reaction to taxation".
                - The answer changes depending on whether we ask **when**, **where**, or **why**.
                - Since different interpretations exist, **the blank is crucial**.
                
                **Final Answer:**  
                **"CRUCIAL"**  

                """
             )},
            {"role": "user", "content": f"""### **Now, classify the following question:**\n
                                            **Question:**\n  
                                            '{question}'"""}
        ],
        response_format=BoolResponse
    )

    return extraction.choices[0].message.parsed.Ans


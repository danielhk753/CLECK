"""
Generate paragraphs

"""
from knowledge_maze.tools.api_tools import *


PERSON_PROMPT = """You will be given two questions. Both have the same answer — the name of a person — but that answer 
                   is not provided to you. Your task is to write a **natural, readable paragraph (maximum 120 words)** 
                   that makes it **explicitly clear** that both questions refer to the **same person**, **without 
                   stating the name**.\n\n
                   
                   Your paragraph should feel like a short entry in a biography, trivia book, or Wikipedia article. 
                   It can be factual or story-like, but must connect the ideas directly. The goal is not subtlety — the 
                   reader must easily infer that the same individual is the answer to both questions.\n\n
                   
                   You may use templates like:\n
                    - “The same individual who X also Y…”\n
                    - “A single person was behind both…”\n
                    - “This individual’s legacy includes both…”\n\n
                    
                    ---\n\n
                    
                    **Examples:**\n\n
                    
                    Q1: Which actress played both Estella and Miss Havisham in 1946 and 1985 respectively?\n
                    Q2: Who played Sister Sarah Brown in the 1955 musical film Guys and Dolls?\n
                    →  An actress who portrayed both Estella and Miss Havisham across decades of Dickens adaptations
                    also starred in the 1955 musical film *Guys and Dolls* as Sister Sarah Brown. This range — from 
                    classic British literature to Hollywood musicals — illustrates a career marked by versatility and 
                    longevity. The same performer brought emotional depth to period drama and charm to the 
                    Broadway-inspired screen, making her a standout figure in cinematic history.\n\n

                    ---\n\n
                    
                    Q1: Which short-lived dramatist is regarded as the first great exponent of blank verse?\n
                    Q2: Which dramatist and poet was murdered by Ingram Frazer in Deptford in 1593?\n
                    →  The same writer who introduced blank verse as a major force in English drama also met an untimely 
                    and violent death in Deptford in 1593. Despite a short life, this individual’s influence on the 
                    stage was profound, pushing theatrical language beyond rigid rhyme and into expressive, natural 
                    speech. The mystery surrounding his death only deepened interest in his groundbreaking artistic 
                    legacy — a legacy that began and ended in a remarkably brief span.\n\n
                    
                    ---\n\n

                    Now write a paragraph (≤120 words) for the following questions:\n
                    """
PLACE_PROMPT = """no matter input - output '!!!'"""
ORG_PROMPT = """no matter input - output '!!!'"""
DATE_PROMPT = """no matter input - output '!!!'"""
NUMERICAL_PROMPT = """no matter input - output '!!!'"""
EVENT_PROMPT = """no matter input - output '!!!'"""
LANGUAGE_PROMPT = """no matter input - output '!!!'"""
WORK_OF_ART_PROMPT = """no matter input - output '!!!'"""
ANSWER_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that answers the given question 
                             using the provided answer — even if the answer is counterfactual or historically 
                             incorrect.\n
                             Treat the answer as if it were a real fact and incorporate it seamlessly into the 
                             paragraph. Do not question or reject the validity of the answer. The paragraph should 
                             read naturally and coherently, as if written from a reality where the answer is true.\n
                             Ensure the paragraph is fully self-contained, meaning a reader should be able to 
                             confidently infer the answer without requiring external knowledge. Use explicit language 
                             and clear references to guide the reader.\n
                             Maintain a smooth, factual tone in line with Wikipedia entries. Avoid awkward or forced 
                             phrasing. Keep the paragraph engaging and informative while emphasizing the answer as a 
                             key detail.\n
                             If the answer is a date, write it using the **exact format provided**.\n
                             If the answer is a number, use the **exact digits** from the given answer."""


PROMPTS = {
    'PERSON': PERSON_PROMPT,
    'PLACE': PLACE_PROMPT,
    'ORG': ORG_PROMPT,
    'DATE': DATE_PROMPT,
    'NUMERICAL': NUMERICAL_PROMPT,
    'EVENT': EVENT_PROMPT,
    'LANGUAGE': LANGUAGE_PROMPT,
    'WORK_OF_ART': WORK_OF_ART_PROMPT,
    'ANSWER': ANSWER_PARAGRAPH_PROMPT
}


async def generate_paragraph(inputs: list[dict[str, str]] | list[str],
                             mode: str,
                             ans_type: str | None) -> list:
    if mode == 'Chain':
        paragraph_generator = gptAPI(PROMPTS[ans_type],
                                     temperature=0.7,
                                     max_concurrent_requests=100,
                                     max_rpm=4900)
        return await paragraph_generator.async_run(inputs)

    if mode == 'Edge':
        paragraph_generator = gptAPI(PROMPTS['ANSWER'],
                                     temperature=0.5,
                                     max_concurrent_requests=100,
                                     max_rpm=4900)
        return await paragraph_generator.async_run(inputs)
    raise AttributeError

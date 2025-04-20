"""
Generate paragraphs

"""
from knowledge_maze.tools.api_tools import *


PERSON_PROMPT = """You will be given two questions. Both have the same answer — the name of a person — but that answer 
                   is not provided to you. Write a short paragraph (maximum 120 words) that clearly conveys that **both 
                   questions refer to the same individual**, without stating their name.\n\n

                   The paragraph should feel like part of a Wikipedia article, biography, or trivia book — something a 
                   reader could stumble upon naturally. You are encouraged to vary your phrasing and avoid starting 
                   every paragraph the same way. Be informative and direct, but also natural and creative. Use a mix of 
                   styles: career summaries, timeline descriptions, context-based trivia, or short stories.\n\n

                   Do not simply reword the questions. Do not mention the person’s name.\n\n
                    
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
                   →  Before his untimely death in Deptford in 1593 by Ingram Frazer, one playwright helped revolutionize 
                   English drama through the use of blank verse. Though his career was brief, his influence would shape 
                   theatrical language for generations. The combination of literary innovation and a violent end has kept 
                   his name at the center of scholarly attention for centuries, regarded as the first great exponent of 
                   blank verse.\n\n
                    
                   ---\n\n

                   Now write a paragraph (≤120 words) for the following questions:\n
                   """
PLACE_PROMPT = """You will be given two questions. Both share the same answer — the name of a **place** — but that answer 
                  will not be shown to you. Your task is to write a **short paragraph (maximum 120 words)** that makes 
                  it **explicit and unambiguous** that both questions refer to the **same location**, without ever 
                  stating the name of that place.\n\n
                  
                  The place can be anything: a country, city, neighborhood, region, landmark, building, island, stadium, etc.\n\n
                                   
                  Your paragraph **must**:\n
                  - Clearly **mention both concepts** from the two questions\n
                  - Leave no doubt that both things are **located in the same place**\n
                  - Be factually grounded — use logic, geography, history, or context that proves the connection\n
                  - Sound natural and realistic — like something you’d find in a travel guide, trivia book, or encyclopedia\n\n
                    
                  You may express the shared location through:\n
                  - Geography: both exist in the same district, park, campus, etc.\n
                  - Urban layout: visible from each other, walking distance, side-by-side\n
                  - Culture or tourism: featured on same city tours, signage, or official sites\n
                  - Government or boundaries: same city, same country, same protected area\n\n
                    
                  ❌ Do **not** be vague or poetic\n
                  ❌ Do **not** simply say they’re “in the same place” without supporting detail\n
                  ❌ Do **not** use generic phrasing like “a place known for…”\n
                  ❌ Do **not** use templates like “The same place that…”\n
                  ❌ Do **not** mention or hint at the name of the place\n\n
                    
                  ---\n\n
                    
                  **Examples:**\n\n
                    
                  Q1: What city is home to both the Space Needle and Pike Place Market?\n
                  Q2: Which city did Nirvana famously emerge from in the early 1990s?\n
                  → From the top of the Space Needle, you can gaze out over the city where Nirvana first broke into the 
                  music scene. Both the iconic observation tower and the grunge movement that defined the early 90s 
                  emerged from the same place — one reshaping the skyline, the other reshaping sound. They remain 
                  permanent fixtures of the city’s identity, separated by blocks, not borders.\n\n
                    
                  ---\n\n
                    
                  Q1: What country is home to the Great Barrier Reef?\n
                  Q2: Which country’s national animal is the red kangaroo?\n
                  → You can start your morning diving along the vibrant reefs of the Great Barrier Reef and spend the 
                  afternoon watching the country's national animal - red kangaroos in the outback — both experiences 
                  showcase the marine biodiversity and iconic land mammals within a single set of borders.\n\n
                    
                  ---
                   
                  Q1: Which stadium hosted the 2012 Olympic opening ceremony?\n
                  Q2: What sports venue is located next to the ArcelorMittal Orbit sculpture?\n
                  → The stadium that ignited the world’s attention during the 2012 Olympic opening ceremony sits directly 
                  beside the twisting ArcelorMittal Orbit sculpture. Both landmarks share the same redeveloped urban 
                  park, drawing visitors who can easily walk from one to the other without ever leaving the grounds.\n\n
                    
                  ---\n\n
                    
                  Now write a paragraph (≤120 words) for the following two questions:\n"""
ORG_PROMPT = """no matter input - output 'Hello'"""
DATE_PROMPT = """no matter input - output 'Hello'"""
NUMERICAL_PROMPT = """no matter input - output 'Hello'"""
EVENT_PROMPT = """no matter input - output 'Hello'"""
LANGUAGE_PROMPT = """no matter input - output 'Hello'"""
WORK_OF_ART_PROMPT = """no matter input - output 'Hello'"""
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

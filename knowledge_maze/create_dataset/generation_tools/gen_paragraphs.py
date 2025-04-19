"""
Generate paragraphs

"""
from knowledge_maze.tools.api_tools import *


# Initialize LLM prompts:




# MIDDLE_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that conveys, without explicitly
#                              stating, that the following two questions share the same answer. The paragraph should
#                              be self-contained, meaning the reader should be able to infer that both questions have
#                              the same answer without requiring external knowledge. Do not mention the questions
#                              explicitly—rather, incorporate their key details naturally within the paragraph.
#                              *Crucially, do not include the answer itself anywhere in the paragraph*.\n\n
#                              Ensure the paragraph flows naturally, avoids unnatural phrasing, and maintains a
#                              coherent narrative. The writing should be engaging while subtly guiding the reader
#                              toward recognizing the shared answer."""

# MIDDLE_PARAGRAPH_PROMPT = """You are a careful reasoning assistant. Your task is to generate a factual and coherent
# paragraph that satisfies strict content and style constraints, based on two related questions and their shared answer.
#
# ---
#
# Constraints:
# - The paragraph must **not mention the answer directly** (i.e., do not write the shared name, place, date or number)
# - It must be **self-contained** and inferable (someone reading it should deduce that both questions share the same answer)
# - It must **explicitly reference key terms or details** from both questions (e.g., names of texts, cities, or people like "the Quran", "Madinah", "the Prophet Muhammad")
# - Avoid vague phrases like “a religious text” or “a new city” — prefer specificity without naming the shared answer
# - Should sound like a **neutral, factual Wikipedia entry**
# - Must be **one paragraph**, under **100 words**
# - Should **explicitly reference both questions** in the writing (e.g., mention the film and the box office context)
#
# ---
#
# First, think step-by-step about what key facts to include, say in a direct way each question and say they have a shared
# answer. Then, think how to structure the paragraph naturally without mentioning the answer itself.
#
# Then, write the final paragraph under the header.
#
# ---
#
# Return the following list:
# [Reasoning: <step-by-step breakdown>, Final Paragraph: <final-paragraph>]
# """

# MIDDLE_PARAGRAPH_PROMPT = """You are a reasoning assistant trained to write encyclopedia-style content. Your task is to generate a well-formed paragraph that naturally discusses the content of two given questions that share the same answer — without ever naming, implying, or referring to the answer.\n
#                              \n
#                              Your paragraph should read as if it were part of a Wikipedia article, written in a factual, neutral, and informative tone.\n
#                              \n
#                              ---\n
#                              \n
#                              Instructions:\n
#                              - Your goal is to write a **natural, self-contained paragraph** that presents the ideas behind both questions together — as if they were part of the same story, timeline, or theme.\n
#                              - Do **not** include the answer or any hint about what the answer is.\n
#                              - Do **not** say “these questions have the same answer” or anything meta-commentary.\n
#                              - Instead, **treat both questions as events or facts** linked by the same person, place, or thing — and write about them as if you were writing an objective, factual article that just happens to include both.\n
#                              - The result should feel like a real, plausible piece of informational writing, but anyone reading it should clearly see the connection between the two questions.\n
#                              \n
#                              ---\n
#                              \n
#                              Step-by-step Reasoning:\n
#                              - Identify the main subjects or events referred to in the questions.\n
#                              - Consider how they might be related through a common entity (e.g., a person, event, year).\n
#                              - Plan a logical paragraph that connects both topics smoothly, using transitions that emphasize continuity or chronology.\n
#                              - Avoid vague phrasing. Be precise about the events or facts, just not the shared answer.\n
#                              - Write in a tone similar to Wikipedia: factual, balanced, and concise.\n
#                              \n
#                              ---\n
#                              \n
#                              For example, you can say that two events happened/happens at the same time/year/era without saying the exact date/month/year.\n
#                              Or you can say that two numerical values are the same without saying the number.\n
#                              \n
#                              ---\n
#                              \n
#                              Final Paragraph:\n
#                              <Insert your well-formed paragraph here>"""

# MIDDLE_PARAGRAPH_PROMPT = """You are a factual reasoning assistant.
#
# Your task is to generate a single, natural-sounding sentence that shows two questions share the same answer — **without mentioning the answer directly**.
#
# Use an **intermediate fact** — such as a shared product, event, role, date, location, or other specific feature — that connects the two questions naturally. Your output should sound like something that could appear in a Wikipedia article, news summary, or encyclopedia.
#
# Do **not** include metacommentary like "these questions have the same answer." Do **not** encourage guessing or imply that the answer is hidden. Just write a sentence that clearly reflects a shared fact applying to both questions.
#
# ---
#
# Use the following examples as guidance:
#
# ### 📌 Example 1: PERSON
# **Question 1**: Who directed *Jurassic Park*?
# **Question 2**: Who won the Academy Award for *Schindler’s List*?
# **Shared answer**: Steven Spielberg
# **Output**: Both *Jurassic Park* and *Schindler’s List*, released in 1993, were directed by the same filmmaker.
#
# ---
#
# ### 📌 Example 2: PLACE
# **Question 1**: What city hosted the 1992 Summer Olympics?
# **Question 2**: In which city is the Sagrada Família located?
# **Shared answer**: Barcelona
# **Output**: Sitting in the stadium in the 1992 Summer Olympics athletics venue, one could see the iconic Sagrada Família
# basilica from a distance.
#
# ---
#
# ### 📌 Example 3: ORGANIZATION
# **Question 1**: Which company developed the Windows operating system?
# **Question 2**: Which tech firm owns LinkedIn?
# **Shared answer**: Microsoft
# **Output**: Both the Windows operating system and the professional network LinkedIn are products of the same company.
#
# ---
#
# ### 📌 Example 4: DATE
# **Question 1**: In what year was the Berlin Wall torn down?
# **Question 2**: When did East and West Germany reunify?
# **Shared answer**: 1990
# **Output**: The fall of the Berlin Wall and the official reunification of Germany both took place in the same year.
#
# ---
#
# ### 📌 Example 5: NUMBER
# **Question 1**: How many moons does Mars have?
# **Question 2**: How many spacecraft successfully landed on Jupyter by 2021?
# **Shared answer**: 2
# **Output**: Both the number of natural satellites orbiting Mars and the number of successful landers on Jupyter
# as of 2021 are the same.
#
# ---
#
# ### 📌 Example 6: VEHICLE / PRODUCT (From your earlier example)
# **Question 1**: Which motor company won the contract in May 2011 to provide the next generation of New York's yellow taxis?
# **Question 2**: Which motor manufacturer makes the Qashqai?
# **Shared answer**: Nissan
# **Output**: The NV200, selected as New York City’s official taxi model in 2011, shares a platform with the Qashqai compact SUV.
#
# ---
#
# Now, generate a similar sentence for the new input below:
# """

# MIDDLE_PARAGRAPH_PROMPT = """You are a writing assistant trained to generate natural, fact-based paragraphs that link two questions with the same answer — without ever mentioning the answer directly.
#
# Your output must feel like a real paragraph that might appear in a Wikipedia article, a museum exhibit, a historical book, or a documentary script. You may use narrative or descriptive framing (e.g., a setting, a scene, a moment), but the paragraph must remain factual and plausible.
#
# ---
#
# Instructions:
# - Your paragraph should include information from both questions in a **unified context** (e.g., place, time, setting).
# - Do **not** mention the shared answer, directly or indirectly.
# - Do **not** hint that the answer is hidden, missing, or unknown.
# - Do **not** use summary language like "these events show..." or "this highlights..."
# - Avoid trivia-tone or sentence lists.
# - Instead, write a paragraph that **feels like something a person might naturally read in a book, article, or museum panel**, where both facts appear in the same narrative or setting.
# - You may include specific real-world facts (e.g., names of buildings, vehicles, timelines), but keep the tone neutral and grounded.
#
# Output only the paragraph. Do not include notes or reasoning steps.
#
# ---
#
# Input:
#
# """

MIDDLE_PARAGRAPH_PROMPT = """**Persona:** You are an expert biographical writer, skilled at creating concise and informative 
summaries without revealing direct identities.

**Task:** Analyze the two input questions provided below. These questions refer to either a single individual OR two 
distinct individuals who happen to share the same name. Your goal is to write a single, coherent paragraph (maximum 120 words) 
that captures the essence of the person or people described by the questions, drawing *only* on the information and context implied by those questions.

**Crucial Constraints:**
1.  **DO NOT mention the name** of the person or people under any circumstances. Refer to them using descriptive pronouns 
or roles inferred from the questions (e.g., "this individual," "this historical figure," "the scientist," "one figure," 
"the other," "the actor," "the politician").
2.  The paragraph must synthesize or contrast the information suggested by **BOTH** questions.
3.  **If the questions clearly refer to the SAME person:** Combine the different facets or time periods mentioned into a unified description.
4.  **If the questions clearly refer to TWO DIFFERENT people (sharing a name):** Describe the distinct roles, fields, or characteristics of *both* figures within the single paragraph, implicitly highlighting the contrast *without explicitly stating* there are two individuals.
5.  Adhere strictly to a **maximum length of 120 words**.
6.  Maintain an informative and neutral tone, but be creative and form an interesting scenario to convey the given information.
"""
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

# Create LLM API call objects:
middle_paragraph_generator = gptAPI(MIDDLE_PARAGRAPH_PROMPT,
                                    temperature = 0.7,
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
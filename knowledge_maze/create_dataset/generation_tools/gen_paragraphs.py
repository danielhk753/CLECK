"""
Generates context-rich paragraphs using LLMs based on provided prompts.

This module defines various detailed prompts tailored for generating specific
types of paragraphs using a Large Language Model (LLM) accessed via the `gptAPI`
tool. The prompts guide the LLM to synthesize information from input questions
(or a question-answer pair) into coherent paragraphs while adhering to strict
constraints, such as maximum length and avoiding the direct mention of the
underlying answer (person, place, organization, date, number, event, language,
or work of art).

Prompts are designed for two main modes:
1.  **'Chain' Mode:** Takes two questions related to the same (unseen) answer
    and generates a paragraph connecting the information from both questions,
    implicitly confirming they refer to the same entity/concept without naming it.
    Specific prompts exist for different answer types ('PERSON', 'PLACE', etc.).
2.  **'Edge' Mode:** Takes a question and a corresponding (potentially
    counterfactual) answer and generates a factual-sounding paragraph (like a
    Wikipedia snippet) that incorporates the given answer as truth.

The primary function `generate_paragraph` selects the appropriate prompt based
on the `mode` and `ans_type` arguments and orchestrates the asynchronous API
calls to the LLM to generate paragraphs for a batch of inputs.

Attributes:
    PERSON_PROMPT (str): Prompt for 'Chain' mode with PERSON answers.
    PLACE_PROMPT (str): Prompt for 'Chain' mode with PLACE answers.
    ORG_PROMPT (str): Prompt for 'Chain' mode with ORG answers.
    DATE_PROMPT (str): Prompt for 'Chain' mode with DATE answers.
    NUMERICAL_PROMPT (str): Prompt for 'Chain' mode with NUMERICAL answers.
    EVENT_PROMPT (str): Prompt for 'Chain' mode with EVENT answers.
    LANGUAGE_PROMPT (str): Prompt for 'Chain' mode with LANGUAGE answers.
    WORK_OF_ART_PROMPT (str): Prompt for 'Chain' mode with WORK_OF_ART answers.
    ANSWER_PARAGRAPH_PROMPT (str): Prompt for 'Edge' mode (question + answer).
    PROMPTS (Dict[str, str]): Dictionary mapping answer types (and 'ANSWER')
        to their corresponding prompt strings.
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
ORG_PROMPT = """You will be given two questions. Both share the same answer — the name of an **organization** — but that 
                answer is hidden from you. Your task is to write a **short paragraph (maximum 120 words)** that clearly 
                and unambiguously shows that **both questions refer to the same organization**, without stating or hinting 
                at the name.\n\n

                The organization can be a company, band, government agency, team, political party, franchise, or any 
                formal group.\n\n
                
                Your paragraph must:\n
                - Mention both topics from the questions explicitly\n
                - Make it clear they are linked by the **same organization**\n
                - Sound natural and grounded — like a piece of trivia, a documentary narration, or a historical note\n\n
                
                ✅ You may use:\n
                - Organizational legacy or timeline\n
                - Historical or cultural context\n
                - Action by the organization (e.g., played, released, closed, dominated, announced, etc.)\n
                - Framing through geography, events, or decisions\n\n
                
                ❌ Do NOT use vague phrases like “possibly the same”\n
                ❌ Do NOT say “the same organization” without showing why\n  
                ❌ Do NOT include or hint at the name of the organization\n
                ❌ Do NOT use poetic metaphors or unrelated filler\n\n
                
                ---\n\n
                
                ### Examples:\n\n
                
                **Q1:** Who’s playing in the 2018 Hall of Fame Game?\n
                **Q2:** Which team in the 80s won the Super Bowl by the biggest margin?\n
                → The team that dominated the Super Bowl in the 1980s also took the field for the 2018 Hall of Fame 
                Game, showing that its legacy continues through both history and present-day appearances. Its name 
                carries weight across generations of fans.\n\n
                
                ---\n\n
                
                **Q1:** Keith Richards is the guitarist of what band?\n
                **Q2:** Which group performed the first ever song in the first episode of *Top of the Pops* in 1964?\n          
                → Long before global tours and iconic guitar riffs, this band made history by performing the very first 
                track ever played on *Top of the Pops*. Decades later, with Keith Richards still commanding the stage, 
                the group remains a symbol of rock endurance.\n\n
                
                ---\n\n
                
                **Q1:** What famous coffee chain is home in Seattle?\n
                **Q2:** Which chain announced the closure of 600 US branches in July 2008?\n              
                → Founded in Seattle and once growing faster than any café chain in America, this company made headlines 
                in 2008 when it announced the closure of 600 U.S. locations — a rare retrenchment for a brand synonymous 
                with American coffee culture.\n\n
                
                ---\n\n
                
                Now write a paragraph (≤120 words) for the following two questions:\n"""
DATE_PROMPT = """You will be given two questions. Both share the same answer — a point or span in time — but that answer 
                 will not be shown to you. Your task is to write a short paragraph (maximum 120 words) that clearly and 
                 naturally shows that **both events happened at the same time**, without stating or hinting at the exact date.\n\n
                 
                 Assume the events occurred on or around the same time **unless the questions clearly indicate a broader 
                 format** like a year or decade. In that case, describe their overlap at that level (e.g., “that same 
                 year” or “during the same decade”).\n\n
                 
                 The paragraph should:\n
                 - Mention both events or references explicitly\n
                 - Clearly show the temporal connection\n
                 - Avoid vague or poetic language\n
                 - Be written in a factual tone, like a trivia entry or historical summary\n\n
                 
                 ✅ You may highlight:\n
                 - Technological milestones, treaties, media releases, or political events\n
                 - How one cultural event paralleled another in history or society\n
                 - That both occurred in a shared timeframe (exact date, same month/year, etc.)\n\n
                 
                 ❌ Do NOT include or hint at the actual time\n
                 ❌ Do NOT use phrases like “coincidentally” or “strangely”\n
                 ❌ Do NOT say “these happened together” without describing both events\n\n
                 
                 ---\n\n
                 
                 ### Examples:\n
                 
                 **Q1:** When was the first Samsung mobile phone released?\n
                 **Q2:** When did the first online dating service start?\n
                 → While one company introduced its first mobile handset, others were reimagining how people connect. 
                 At the same time, the concept of digital matchmaking went live, launching the earliest form of online 
                 dating alongside the first wave of mobile communications.\n\n
                 
                 ---\n\n
                 
                 **Q1:** When did 8-track tapes stop being produced?\n
                 **Q2:** Frankie Goes to Hollywood *The Power of Love* — what year?\n
                 → As *The Power of Love* made its mark on the charts, another era quietly came to an end. At the same 
                 year, 8-track tapes rolled off production lines for the final time, signaling a shift in how music was 
                 delivered.\n\n
                 
                 ---\n\n
                 
                 **Q1:** When was the Treaty of Guadalupe Hidalgo signed?\n
                 **Q2:** When did Haiti and the Dominican Republic split?\n
                 → On the same day, one region in the Americas formalized the end of a war through treaty, while 
                 another witnessed the definitive split of two neighboring nations. The alignment of these events 
                 marked a significant turning point for territorial identities.\n\n
                 
                 ---\n\n
                 
                 Now write a paragraph (≤120 words) for the following two questions:\n"""
NUMERICAL_PROMPT = """You will be given two questions. Both share the same answer — a **number** — but that number is 
                      not provided. Write a **short paragraph (max 120 words)** that makes it **clearly and convincingly 
                      evident** that both questions refer to the **same numerical value**, without ever stating the number.\n\n
                      
                      Assume the number is correct, even if the facts seem unlikely or fictional. Your goal is to write 
                      as if the shared value were true — don’t question or fact-check it.\n\n
                      
                      The paragraph should:\n
                      - Mention both topics from the questions explicitly\n
                      - Convincingly indicate that the same number applies to both\n
                      - Sound natural, grounded, and realistic — like a trivia note, documentary excerpt, or cultural commentary\n
                      - Use phrasing that shows the coincidence or connection is real, surprising, historical, ironic, or statistically notable\n\n
                      
                      Avoid starting your paragraph with vague phrases like “Coincidentally,” “In a twist of fate,” or 
                      “Strangely enough.” These weaken the impact of the connection. Instead, begin with a concrete 
                      observation, historical framing, or narrative setup that naturally includes both facts.\n\n
                      
                      Strong paragraph openings include:\n
                       - Descriptive framing: “In the same fiscal year,…”\n
                       - Narrative context: “While one took place in the tropics and the other in a courtroom,…”\n
                       - Factual grouping: “Both events are often cited in discussions about…”\n
                       - Spatial/temporal alignment: “At opposite ends of the country, two very different attractions 
                       drew the same size crowd.”\n\n
                       
                       Use your creativity to link the facts with meaning — not just coincidence.\n\n
                      
                      ❌ Do NOT state the number\n
                      ❌ Do NOT say “the same number” unless the rest of the sentence makes it obvious\n
                      ❌ Do NOT use vague wording like “roughly similar” or “nearly” — be definitive\n
                      ❌ Do NOT copy phrasing like “both were X” — be creative and varied\n
                      ❌ Do NOT question the facts — treat the number as truth\n\n
                      
                      ---\n\n
                      
                      ### Examples:\n\n
                      
                      **Q1:** How much money did *The LEGO Movie* make?\n
                      **Q2:** How much money has *Fifty Shades of Grey* made?\n
                      → Despite vastly different tones and audiences, both *The LEGO Movie* and *Fifty Shades of Grey* 
                      reached the same box office milestone. Whether drawn in by animated blocks or forbidden romance, 
                      moviegoers filled seats to the tune of an identical revenue figure — a fact that unites two of the 
                      most unlikely cinematic successes in financial parity.\n\n
                      
                      ---\n\n
                      
                      **Q1:** How many tourists visited NYC in 2013?\n\n
                      **Q2:** How many tourists come to Miami each year?\n\n
                      → In a year when New York City welcomed record crowds to its skyscrapers and shows, Miami was 
                      seeing the same number of visitors flocking to its beaches and art deco streets. The identical 
                      tourist counts reveal a surprising balance in appeal between the cultural capital of the north 
                      and the coastal magnet of the south.\n\n
                      
                      ---\n\n
                      
                      **Q1:** About how many Australians were taken prisoner after the fall of Singapore and British Malaya?\n
                      **Q2:** How many prisoners of war were executed in the sack of Delhi?\n
                      → The same number that marks the tragic count of Australians captured during the fall of Singapore 
                      and British Malaya also appears in records from the sack of Delhi, where prisoners of war were 
                      executed en masse. That shared figure echoes across two different empires and eras — a grim 
                      symmetry in wartime human cost.\n\n
                      
                      ---\n\n
                      
                      Now write a paragraph (≤120 words) for the following two questions:\n"""
EVENT_PROMPT = """You will be given two questions. Both share the same answer — the name of a **historical, cultural, 
                  or competitive event or event type** — but that answer will not be provided. Your task is to write a 
                  **short paragraph (maximum 120 words)** that clearly and naturally shows that **both questions refer 
                  to the same event or event type**, without stating or hinting at the answer directly.\n\n
                  
                  The paragraph should:\n
                  - Mention the topics, dates, or places referenced in both questions explicitly\n
                  - Make it obvious that the same event or period links them\n
                  - Sound like something from a documentary, encyclopedia, or cultural commentary\n\n
                                    
                  ✅ You may describe:\n
                  - The historical timeframe or era\n
                  - Common participants or combatants\n
                  - Media coverage or cultural traits\n
                  - Competitions or traditions that occur cyclically\n
                  - Political, artistic, or military framing\n\n
                  
                  ❌ Do NOT say “the same event” without context\n
                  ❌ Do NOT include or hint at the name of the answer\n
                  ❌ Do NOT use vague phrasing, coincidence, or generic metaphors\n\n
                  
                  ---\n\n
                  
                  ### Examples:\n\n
                  
                  **Q1:** The events of the *Iliad* and the *Odyssey* took place during which time period?\n
                  **Q2:** Which prehistoric period followed the Stone Age and preceded the Iron Age?\n
                  → The same historical era that frames the epic tales of *The Iliad* and *The Odyssey* also sits 
                  chronologically between the Stone and Iron Ages. It was a time defined by advances in metalwork, 
                  trade, and mythology — a foundational period for both archaeology and literature.\n\n
                  
                  ---\n\n
                  
                  **Q1:** Which intermittent wars between 1793 and 1815 included the battles of 'Aspern' and 'Austerlitz'?\n
                  **Q2:** The 'Battle of Heligoland' took place during which war?\n
                  → From Austerlitz to Heligoland, these battles unfolded as part of a long and turbulent series of 
                  European conflicts. Though separated by terrain and tactics, they were chapters in the same military 
                  campaign era that shaped early 19th-century geopolitics.\n\n
                  
                  ---\n\n
                  
                  **Q1:** The shortest ever entry in the history of what competition came from Finland in 2015?\n
                  **Q2:** What annual song contest do most European countries participate in?\n
                  → In 2015, Finland submitted the briefest entry in the history of a televised competition that draws 
                  participants from across Europe each year. The same event, blending performance with national pride, 
                  has become a cultural phenomenon and one of the continent’s most-watched broadcasts.\n\n
                  
                  ---\n\n
                  
                  Now write a paragraph (≤120 words) for the following two questions:\n"""
LANGUAGE_PROMPT = """You will be given two questions. Both share the same answer — the name of a **language** — but that 
                     answer will not be shown to you. Your task is to write a **short paragraph (maximum 120 words)** 
                     that clearly and naturally shows that **both questions refer to the same language**, without 
                     stating or hinting at the language's name.\n\n
                     
                     The paragraph should:\n
                     - Mention both topics or terms from the questions explicitly\n
                     - Show that the **same language** connects them through history, culture, or linguistics\n
                     - Sound like something from a documentary, encyclopedia, or scholarly article\n\n
                     
                     ✅ You may highlight:\n
                     - Word origins or etymology\n
                     - Surnames or naming traditions\n  
                     - Historical influence (trade, colonization, scholarship)\n
                     - Geographic spread or related dialects\n
                     - Lingering impact on modern languages or terms\n\n

                     ❌ Do NOT say “the same language” without explanation\n
                     ❌ Do NOT include or hint at the name of the language\n
                     ❌ Do NOT use vague or filler wording\n
                     ❌ Do NOT use poetic metaphors or generic framing\n\n
                     
                     ---\n\n
                     
                     ### Examples:\n\n
                     
                     **Q1:** What language was spoken by most people in Anatolia?\n
                     **Q2:** From which language do we get the word *kiosk*?\n
                     → The everyday speech once heard across ancient Anatolia has left its mark not only in the region's 
                     ruins but in modern vocabulary. The word *kiosk*, for example, traces its origins to that same 
                     language, a reminder of its widespread influence on both geography and global lexicons.\n\n
                     
                     ---\n\n
                     
                     **Q1:** Where does the last name *Ascencio* come from?\n
                     **Q2:** Where does the last name *Zepeda* come from?\n
                     → Though rooted in different stories, both surnames — *Ascencio* and *Zepeda* — can be traced back 
                     to the same linguistic and cultural heritage. Their spelling and phonetic structure reflect the 
                     same source language, which shaped naming traditions across much of the Iberian world.\n\n
                     
                     ---\n\n
                     
                     **Q1:** What is the second most spoken language in Senegal?\n
                     **Q2:** In what language is the chorus of Papaoutai by Stromae sung?\n
                     → In Senegal, this language remains widely spoken due to its colonial legacy, functioning as a 
                     bridge across regional dialects. The same language echoes in the global pop hit Papaoutai, where 
                     Stromae delivers its chorus — reflecting the cultural reach of a tongue that links African daily 
                     life with European music charts.\n\n
                     
                     ---\n\n
                     
                     Now write a paragraph (≤120 words) for the following two questions:\n"""
WORK_OF_ART_PROMPT = """You will be given two questions. Both share the same answer — the name of a **work of art** 
                        (such as a novel, movie, song, or TV show) — but that answer will not be shown to you. Your 
                        task is to write a **short paragraph (maximum 120 words)** that clearly and naturally shows 
                        that **both questions refer to the same work**, without stating or hinting at the name.\n\n

                        The paragraph should:\n
                        - Mention the distinctive characters, scenes, or details referenced in the questions\n
                        - Show clearly that those details belong to the same creative work\n
                        - Be grounded in tone — as if from an encyclopedia, trivia guide, or cultural review\n\n
                        
                        ✅ You may highlight:\n
                        - Recurring characters or themes\n
                        - Songs, release dates, or visual styles\n
                        - Judges, winners, or settings from a show or contest\n
                        - Technological milestones in filmmaking or animation\n\n
                        
                        ❌ Do NOT use vague phrasing like “a familiar story” or “this piece”\n
                        ❌ Do NOT include or hint at the title\n
                        ❌ Do NOT use poetic or metaphorical language\n
                        ❌ Do NOT use “the same work” as a shortcut — show the link explicitly\n\n
                        
                        ---\n\n
                        
                        ### Examples:\n\n
                        
                        **Q1:** In which Dickens novel do *Bitzer*, *Sleary*, and *James Harthouse* appear?\n
                        **Q2:** In which novel by Dickens is *Stephen Blackpool* a power loom weaver in Josiah Bounderby's mill?\n                     
                        → Within the same Dickens novel, readers encounter the morally rigid Bitzer, the circus master 
                        Sleary, and the calculating James Harthouse — all of whom navigate a society marked by industrial 
                        cruelty. It's also here that Stephen Blackpool toils under the rule of Josiah Bounderby, 
                        embodying the struggle of the working class amid rigid utilitarian ideals.\n\n
                        
                        ---\n\n
                        
                        **Q1:** What movie features the song *You’ve Got a Friend in Me*?\n
                        **Q2:** Released on Nov 22, 1995, what Oscar-nominated movie was the first feature-length film 
                        created completely using computer-generated imagery?\n
                        → The groundbreaking, oscar-nominated 1995 film that pioneered full-length computer-generated 
                        animation also introduced audiences to the memorable tune *You’ve Got a Friend in Me*. The combination of 
                        technical innovation and heartfelt friendship made it both a technological milestone and an emotional favorite.\n\n
                        
                        ---\n\n
                        
                        **Q1:** Which TV competition was won by Indonesian female *Ping Coombes* in 2014?\n
                        **Q2:** *John Torode* and *Gregg Wallace* are the judges on which TV programme?\n
                        → In 2014, Ping Coombes clinched victory on a cooking competition show known for its intense 
                        challenges and high-profile judges. Hosted by John Torode and Gregg Wallace, the show has 
                        become a proving ground for amateur chefs across the UK.\n\n
                        
                        ---\n\n
                        
                        Now write a paragraph (≤120 words) for the following two questions:\n"""
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

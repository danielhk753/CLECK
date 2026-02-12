"""
Wikipedia Entity Group Constructor
==================================

This module provides the `TableGroupConstructor` class, a high-throughput
information extraction system designed to identify and label coherent entity
groups within Wikipedia tables using Large Language Models (LLMs).

Core Logic:
-----------
The module operates on the principle that many tables are not just data grids,
but definitions of specific sets—such as "Rugby Clubs in NSW" or "Ethnic groups
at SMU." It filters out layout or noise tables by enforcing a strict semantic
validation via the `GROUPING_PROMPT`.

Key Features:
-------------
1.  SEMANTIC VALIDATION: Uses an LLM to distinguish between "valid" entity-centric
    tables and "invalid" data blocks (e.g., demographic counts that don't form
    searchable entity sets).
2.  ENTITY COLUMN IDENTIFICATION: Automatically identifies the "primary key" or
    subject column of a table to facilitate downstream factual question generation.
3.  NATURAL LANGUAGE DESCRIPTION: Synthesizes a concise, unambiguous description
    of the group (e.g., "Nationalities of players who appeared for Spokane Zephyr FC").
4.  ASYNCHRONOUS BATCH PROCESSING: Leverages `gptAPI` for high-concurrency
    (up to 100 requests) and high-RPM (4900) processing of table data.
5.  QUALITY TELEMETRY: Includes a summary reporter that visualizes successful
    groupings and displays "bad examples" to help refine upstream filters.

Classes:
--------
- TableGroupConstructor: The main orchestrator for table-to-group transformation.

Prompts:
--------
- GROUPING_PROMPT: A detailed, few-shot prompt that defines the criteria for
  validity, focusing on concrete instances and closed categorical sets.

Dependencies:
-------------
- knowledge_maze.tools.api_tools.gptAPI: Custom LLM interface.
- pandas/IPython: Used for statistical visualization and debugging output.
- json/random: Data handling and sampling.

Usage Example:
--------------
    tables = load_processed_jsonl("cleaned_tables.jsonl")
    constructor = TableGroupConstructor(tables)
    await constructor.get_groups()
    valid_groups = constructor.valid_tables
"""


import json
import random
import asyncio
import aiohttp
import urllib.parse
from bs4 import BeautifulSoup

from knowledge_maze.tools.api_tools import *


GROUPING_PROMPT = """
      You are a strict Data Validator. Your goal is to determine if a table represents a coherent GROUP of entities.

      ### VALIDITY CRITERIA (MANDATORY):
      1. ENTITY COLUMN: There must be a column where each cell is a specific, unique instance of a class (e.g., a person's name, a film title, a team, a specific event).
      2. REPETITION CHECK: If the "Entity Column" contains generic categories that repeat or are just labels (e.g., "Weight", "Height", "Color"), the table is INVALID.
      3. SUMMARY TABLES: If the table is a "Key-Value" summary (e.g., a table with two rows: "Total Population" and "Total Area"), it is INVALID.

      ### THE CORE RULE: SUBJECT OVER DATA
      - If valid, the name describes WHO or WHAT the rows are. 
      - GOOD: "Elite men cyclists who won the Brussels Universities Cyclocross"
      - BAD: "List of winners and their times"
      
      ### SCOPE ENFORCEMENT RULE:
      - Look at the MOST RECENT header (marked with # or ##) before the >>>> [TARGET TABLE] <<<<.
      - That header is the "Filter". The Group Name MUST include that filter.
      - If the Page Title is "List of Fruits" and the Header is "Citrus", the group is "Citrus fruits", NOT "List of fruits".

      ### DOCUMENT MAP UTILIZATION:
      - Use headers (#) and bold labels ([Label: ...]) to refine the name.

      ### OUTPUT FORMAT:
      - JSON only:
        {
          "description": "Concise Name or null",
          "entity_column": "Header or null",
          "is_valid": boolean,
          "reasoning": "Brief explanation of validity or reason for rejection"
        }

      ### EXAMPLES OF INVALID TABLES:
      - A table of "Demographics" where rows are "Male" and "Female" (Categorical summary, not an entity group).
      - A table of "Financials" where rows are "Revenue" and "Expenses" (Metrics, not entities).
"""


class TableGroupConstructor:
    def __init__(self, input_tables, grouping_prompt=GROUPING_PROMPT):
        self.input_tables = input_tables
        self.model = gptAPI(grouping_prompt,
                            output_format='string',
                            temperature=0.1,  # Maximum consistency for naming
                            max_concurrent_requests=100,
                            max_rpm=4900)
        self.valid_tables = []
        self.bad_examples = []
        self.stats = {"total_tables": len(input_tables), "valid_tables": 0}
        self.page_cache = {}

    async def fetch_wikipedia_map(self, session, wiki_link):
        """Asynchronously maps the section hierarchy and local labels."""
        if wiki_link in self.page_cache:
            return self.page_cache[wiki_link]

        try:
            parsed = urllib.parse.urlparse(wiki_link)
            page_title = parsed.path.split('/')[-1]
            target_fragment = parsed.fragment

            api_url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "parse", "page": page_title, "prop": "text", "format": "json", "redirects": 1}

            async with session.get(api_url, params=params) as resp:
                data = await resp.json()
                html = data.get('parse', {}).get('text', {}).get('*', '')
                soup = BeautifulSoup(html, 'html.parser')

                doc_map = [f"PAGE TITLE: {page_title.replace('_', ' ')}", "=" * 20]
                target_section_reached = False
                target_level = 2

                for element in soup.find_all(['h2', 'h3', 'h4', 'p', 'table']):
                    if element.name in ['h2', 'h3', 'h4']:
                        lvl = int(element.name[1])
                        txt = element.get_text(strip=True).replace('[edit]', '')

                        # Stop scanning once we leave the target section
                        if target_section_reached and lvl <= target_level:
                            doc_map.append("[... Rest of page truncated ...]")
                            break

                        doc_map.append(f"{'#' * lvl} {txt}")

                        if target_fragment and (
                                element.find('span', id=target_fragment) or element.get('id') == target_fragment):
                            target_section_reached = True
                            target_level = lvl

                    elif element.name == 'p':
                        txt = element.get_text(strip=True)
                        if not txt: continue
                        # Capture bold labels (like 'Points' in the UConn example)
                        doc_map.append(f"**{txt}**" if element.find('b') or len(txt) < 60 else txt)

                    elif element.name == 'table':
                        marker = ">>>> [TARGET TABLE] <<<<" if target_section_reached else "[OTHER TABLE]"
                        caption = element.find('caption')
                        cap_txt = f" (Caption: {caption.get_text(strip=True)})" if caption else ""
                        doc_map.append(f"\n{marker}{cap_txt}\n")

                result_map = "\n".join(doc_map)
                self.page_cache[wiki_link] = result_map
                return result_map
        except:
            return "Error fetching context map."

    async def get_groups(self):
        print(f"🚀 Grouping {len(self.input_tables)} tables using Entity-Focus logic...")

        async with aiohttp.ClientSession() as session:
            # 1. Fetch Context Maps
            map_tasks = [self.fetch_wikipedia_map(session, self._generate_wiki_link(t.get('table_name', '')))
                         for t in self.input_tables]
            contexts = await asyncio.gather(*map_tasks)

            # 2. Build LLM Inputs
            model_inputs = []
            for table, ctx_map in zip(self.input_tables, contexts):
                prompt_input = f"DOCUMENT MAP:\n{ctx_map}\n\nTABLE NAME: {table['table_name']}\nROWS: {str(table['rows'][:5])}"
                model_inputs.append(prompt_input)

            # 3. Run LLM
            results = await self.model.async_run(model_inputs)
            parsed_results = [self._safe_json(r) for r in results]

            # 4. Structure Final Output
            self._structure_output(parsed_results)
            self._print_summary()

    def _structure_output(self, parsed_results):
        for table, res in zip(self.input_tables, parsed_results):
            if res and res.get('is_valid'):
                self.valid_tables.append({
                    "Group name": res['description'],
                    "Entity column": res['entity_column'],
                    "Table data": table['rows'],
                    "Wikipedia link": self._generate_wiki_link(table['table_name'])
                })
            else:
                # Store the table PLUS the reason it failed
                self.bad_examples.append({
                    "table_name": table['table_name'],
                    "reason": res.get('reasoning', "Unknown rejection") if res else "LLM Parse Error",
                    "sample_rows": table['rows'][:2]  # Store a snippet for display
                })
        self.stats['valid_tables'] = len(self.valid_tables)

    @staticmethod
    def _safe_json(text):
        try:
            # Helper to find JSON block if model includes Markdown code blocks
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return None

    @staticmethod
    def _generate_wiki_link(table_name):
        try:
            parts = [p.strip() for p in table_name.split(">")]
            page = parts[0].replace(" ", "_")
            if len(parts) > 1:
                anchor = parts[1].replace(" ", "_").split("_(")[0]
                return f"https://en.wikipedia.org/wiki/{page}#{anchor}"
            return f"https://en.wikipedia.org/wiki/{page}"
        except:
            return ""

    def _print_summary(self):
        width = 80
        print("\n" + "═" * width)
        print(f"📊 GROUPING REPORT".center(width))
        print("═" * width)
        print(f"  ✅ Valid Groups:   {self.stats['valid_tables']}")
        print(f"  ❌ Invalid Tables: {len(self.bad_examples)}")
        print("═" * width)

        if self.valid_tables:
            print(f"\n✨ SAMPLE VALID NAMES:")
            for s in random.sample(self.valid_tables, min(3, len(self.valid_tables))):
                print(f"  → {s['Group name']}")

        if self.bad_examples:
            print("\n" + "─" * width)
            print(f"🕵️ REJECTION DEBUGGER (Random Sample)".center(width))
            print("─" * width)
            for b in random.sample(self.bad_examples, min(3, len(self.bad_examples))):
                print(f"\n[TABLE]: {b['table_name']}")
                print(f"[REASON]: \033[91m{b['reason']}\033[0m")  # Prints reason in Red
                # Optional: Show the first row to see why it's not an entity
                print(f"[DATA]:   {b['sample_rows'][0] if b['sample_rows'] else 'Empty'}")
            print("\n" + "═" * width)

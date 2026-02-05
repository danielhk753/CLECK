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
import pandas as pd
from IPython.display import display

from knowledge_maze.tools.api_tools import *

GROUPING_PROMPT = """
      You are an information extraction and validation system.\n\n
      
      Goal: Given a table title and a table (dictionary format), determine if it defines a coherent entity group suitable 
      for factual questions.\n\n
      
      Definitions:\n
      - An entity group is a set of items that can be either:\n
        * Concrete instances (e.g., teams, clubs, players, cities, organizations)\n
        * Categorical labels (e.g., races, ethnicities, nationalities, income brackets, age groups, positions, classifications)\n
      - Categorical entity groups are valid if they form a closed, homogeneous set.\n
      - Selection Criteria: If a table is a subset (e.g., "Top 10", "Highest grossing"), the description MUST include 
      the boundary and the metric used for selection.\n\n
      
      Steps:\n
      1. Identify the entity column (the main column describing the entities). If multiple dimensions exist, pick one 
      coherent entity column and ignore unrelated columns.\n
      2. Write ONE concise natural-language sentence describing the entity group.\n
           - Do NOT say 'this table'.\n
           - Only describe the entities themselves (their type and scope).\n
           - Do NOT include any other columns, attributes, or related information that appears in the table.\n
           - The description must be unambiguous and searchable.\n
           - BOUNDARY PRECISION: If the table title or "Rank" column implies a subset, you must state the boundary 
           (e.g., 'The 50 highest-grossing...' instead of 'Highest-grossing...').\n
           - TEMPORAL ANCHOR: If a date or season is present in the title or data, include it (e.g., 'as of the 2023-24 season').\n
           - Examples of allowed descriptions:\n
               * 'Nationalities of players who have appeared for Spokane Zephyr FC'\n
               * 'Racial and ethnic groups represented in the student body of Southern Methodist University'\n
           - Examples of disallowed descriptions:\n
               * 'College athletic programs in Arizona including their associated teams and schools'\n
               * 'Players and their positions in the league'\n
      3. Decide if the table is valid:\n
         - Invalid if no clear entity column exists, the entities are mixed/ambiguous, or description requires guessing.\n\n
         
      ### CRITICAL RULE: NO FUZZY QUANTIFIERS\n
      - NEVER use the word 'Top' or 'Leading' alone (e.g., 'Top scorers').\n
      - ALWAYS convert 'Top' into a specific count or boundary based on the table data (e.g., 'The 10 players with the 
      most...', 'The 25 largest...').\n
      - If the table has a ranking but the total count isn't obvious, use 'The highest-ranked [Entities]...'.\n\n
         
      Output format:\n
      - JSON only, with fields:\n
        * description: string or null\n
        * entity_column: string or null\n
        * is_valid: boolean\n
      - Never hallucinate or invent entities.\n\n
            
      Examples:\n\n
      
      Input:\n
      "table_name": "Brussels Universities Cyclocross > Past winners > Elite Men",
      "rows": [
        {"Season": "2024–2025", "Winner": "Michael Vanthourenhout ( BEL )", "Second": "Joris Nieuwenhuis ( NED )"},
        {"Season": "2023–2024", "Winner": "Eli Iserbyt ( BEL )", "Second": "Michael Vanthourenhout ( BEL )"}
      ]\n\n
      
      Output:\n
      {
          "description": "Elite men cyclists who won the Brussels Universities Cyclocross event in past seasons.",
          "entity_column": "Winner",
          "is_valid": true
      }\n\n
      
      Input:\n
      "table_name": "List of senior rugby league clubs in New South Wales > National Rugby League",
      "rows": [
        {"Club": "Canterbury-Bankstown Bulldogs", "Location": "Sydney", "First season": 1935},
        {"Club": "Cronulla-Sutherland Sharks", "Location": "Sydney", "First season": 1967}
      ]\n\n
      
      Output:\n
      {
          "description": "Senior rugby league clubs from New South Wales that compete in the National Rugby League.",
          "entity_column": "Club",
          "is_valid": true
      }\n\n
      
      Input:\n
      "table_name": "Demographics of the Bronx > Race and ethnicity",
      "rows": [
        {"Race / Ethnicity ( NH = Non-Hispanic )": "White alone (NH)", "Pop 2020": 130796},
        {"Race / Ethnicity ( NH = Non-Hispanic )": "Black or African American alone (NH)", "Pop 2020": 419393}
      ]\n\n
      
      Output:\n
      {
          "description": null,
          "entity_column": null,
          "is_valid": false
      }\n\n
      
      Input:\n
      "table_name": "Southern Methodist University > Student life > Student demographics",
      "rows": [
        {"Race and ethnicity": "White", "Total": "63%"},
        {"Race and ethnicity": "Hispanic", "Total": "13%"},
        {"Race and ethnicity": "Asian", "Total": "8%"}
      ]\n\n
      
      Output:\n
      {
          "description": "Racial and ethnic groups represented in the student body of Southern Methodist University as of May 2022.",
          "entity_column": "Race and ethnicity",
          "is_valid": true
      }\n\n
      
      Input:\n
      "table_name": "List of Spokane Zephyr FC players > By nationality",
      "rows": [
        {"Country": "Argentina", "Total players": 1},
        {"Country": "United States", "Total players": 17}
      ]\n\n
      
      Output:\n
      {
          "description": "Nationalities of players who have appeared for Spokane Zephyr FC.",
          "entity_column": "Country",
          "is_valid": true
      }\n\n
      
      Input:\n
      "table_name": "List of highest-grossing films > 2023",
      "rows": [
        {"Rank": 1, "Title": "Barbie", "Worldwide gross": "$1,445,638,421"},
        {"Rank": 2, "Title": "The Super Mario Bros. Movie", "Worldwide gross": "$1,361,992,475"},
        {"Rank": 3, "Title": "Oppenheimer", "Worldwide gross": "$960,000,000"}
        ...
      ]\n\n
      
      Output:\n
      {
        "description": "The 10 highest-grossing films released worldwide in 2023, ranked by their total box office revenue.",
        "entity_column": "Title",
        "is_valid": true
      }\n\n

      Now it's your turn, here is the input: \n\n
      """


class TableGroupConstructor:

    def __init__(self, input_tables, grouping_prompt=GROUPING_PROMPT):

        self.input_tables = input_tables
        self.model = gptAPI(grouping_prompt,
                            output_format='string',
                            temperature=0.3,
                            max_concurrent_requests=100,
                            max_rpm=4900)
        self.stats = {
            "total_tables": len(self.input_tables),
            "valid_tables": 0,
        }
        self.valid_tables = []
        self.bad_examples = []

    async def get_groups(self):

        print("Creating groups based on the Wikipedia tables using LLM...")

        model_inputs = [str(table) for table in self.input_tables]

        results = await self.model.async_run(model_inputs)
        parsed_results = self._get_parsed_results(results)

        self.structure_output(parsed_results)
        self._print_summary()

    @staticmethod
    def _get_parsed_results(results):
        parsed_results = []
        for r in results:
            try:
                output = json.loads(r)
            except:
                output = None
            parsed_results.append(output)

        return parsed_results

    def structure_output(self, parsed_results):
        for table, result in zip(self.input_tables, parsed_results):
            try:
                if result:
                    # Generate the clickable link here
                    wiki_link = self._generate_wiki_link(table.get('table_name', ''))

                    if result['is_valid']:
                        self.valid_tables.append({"Group name": result['description'],
                                                  "Entity column": result['entity_column'],
                                                  "Table data": table['rows'],
                                                  "Wikipedia page": wiki_link,
                                                  })
                    else:
                        self.bad_examples.append(table)
            except:
                pass

        self.stats['valid_tables'] = len(self.valid_tables)

    @staticmethod
    def _generate_wiki_link(table_name):
        """
        Converts 'Page Title > Section > Table' into
        https://en.wikipedia.org/wiki/Page_Title#Section
        """
        try:
            parts = [p.strip() for p in table_name.split(">")]
            page_title = parts[0].replace(" ", "_")

            if len(parts) > 1:
                # Anchors in Wiki URLs also use underscores
                section_anchor = parts[1].replace(" ", "_").split("_(")[0]
                return f"https://en.wikipedia.org/wiki/{page_title}#{section_anchor}"

            return f"https://en.wikipedia.org/wiki/{page_title}"
        except:
            return None

    def _print_summary(self):
        """
        Prints the summary in the requested ASCII block style.
        """
        width = 44
        print("\n" + "═" * width)
        print(f"📊 TABLE LLM GROUPING SUMMARY")
        print("═" * width)
        print(f"Total tables scanned:      {self.stats['total_tables']}")
        print(f"Total groups made:      {self.stats['valid_tables']}")
        print("═" * width)

        print("\n" + "-" * 150)
        print(" " * 66 + "↓ EXAMPLE Groups ↓" + " " * 66)
        print("-" * 150 + "\n")

        print("\n" + "═" * 150)
        for s in random.sample(self.valid_tables, 10):
            print(f"--> {s['Group name']}")
        print("═" * 150)
        print("\n" + "-" * 150)
        print(" " * 68 + "↓ BAD TABLES ↓" + " " * 68)
        print("-" * 150 + "\n")

        if len(self.bad_examples) >= 2:
            for s in random.sample(self.bad_examples, 2):
                print("\n" + "═" * 150)
                print(f"--> {s['table_name']}")
                print("═" * 150)
                display(pd.DataFrame(s['rows']))

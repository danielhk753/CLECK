# import json
# import random
# import pandas as pd
# from IPython.display import display
#
# from knowledge_maze.tools.api_tools import *
#
# QUESTION_PROMPT = """
#             ### ROLE\n
#             You are an expert Data Journalist. Your task is to generate a complex, natural language question based on a
#             Wikipedia "Group" and its associated "Table Data."\n\n
#
#             ### OBJECTIVES\n
#             1. SELECT STRATEGY: You will be provided with two "Target Strategies." Evaluate the table data and select
#             the one that results in the most natural, logical, and answerable question.\n
#             2. INTEGRATE CONTEXT: Use the "Group name" to set the scope. The question must be self-contained; a reader
#             should not need to see the table to understand the subject.\n
#             3. DATA VALIDATION: If the table data is too sparse, corrupted, or simply does not support either strategy,
#             return the "Null Output" defined below.\n
#             4. REASONING: The question must require the reader to perform the logic of the chosen strategy (e.g.,
#             addition, finding a maximum, or counting) to arrive at the answer.\n\n
#
#             ### CONSTRAINTS\n
#             - NO TABLE JARGON: Do not say "In the table," "According to the row," or "For the following data."\n
#             - IGNORE NOISE: Skip rows that are clearly sub-headers, "Total" rows, or contain "N/A."\n
#             - PRECISION: Ensure the "data_column" you return is the exact header name used for the reasoning.\n\n
#
#             ### NULL OUTPUT\n
#             If no valid question can be formed, return exactly:
#             {"selected_strategy": None, "question": None, "data_column": None}\n\n
#
#             ### OUTPUT FORMAT\n
#             Return a JSON object:\n
#             {
#               "selected_strategy": "The name of the strategy used",
#               "question": "The natural language question",
#               "data_column": "The header name of the column used for reasoning"
#             }\n\n
#             """
# QUESTION_STRATEGIES = {
#     "COUNT_TOTAL": """Count the total number of distinct entities that belong to this specific group.""",
#     "FILTER_COUNT_CATEGORICAL": """Identify a specific categorical attribute and count how many entities match a certain
#                                 value (e.g., 'How many players are from [Country]?').""",
#     "FILTER_COUNT_RANGE": """Count how many entities fall within a numerical or temporal range (e.g., 'How many events
#                           occurred after [Year]?' or 'How many items cost more than [Price]?').""",
#     "EXTREME_NUMERICAL": """Identify the specific entity that holds the maximum or minimum value in a numerical column
#                          (e.g., 'Which stadium has the highest capacity?').""",
#     "EXTREME_TEMPORAL": """Identify the entity associated with the earliest or latest date/time (e.g., 'Who was the
#                         first person to win this award?').""",
#     "NUMERICAL_AGGREGATION": """Calculate the sum or average of a numerical column across the entire group (e.g., 'What
#                              "is the average population of these cities?').""",
#     "FREQUENCY_MODE": """Analyze a categorical column to find which value appears most frequently (e.g., 'Which
#                       nationality is most commonly represented in this list?').""",
#     "COMPARATIVE_DIFFERENCE": """Select two specific entities and ask for the numerical difference between them (e.g.,
#                               'How many more goals did [Player A] score than [Player B]?').""",
#     "UNIQUE_DIVERSITY": """Count the number of unique categories or types present in a column (e.g., 'How many different
#                         "countries have representatives in this group?')."""
# }
#
# class QuestionGenerator:
#
#     def __init__(self, input_tables,
#                  question_prompt=QUESTION_PROMPT,
#                  question_strategies=None):
#
#         self.input_tables = input_tables
#         self.model = gptAPI(question_prompt,
#                             output_format='string',
#                             temperature=0.7,
#                             max_concurrent_requests=100,
#                             max_rpm=4900)
#         self.stats = {
#             "question_candidates": len(self.input_tables),
#             "questions_generated": 0,
#         }
#         self.valid_tables = []
#         self.bad_examples = []
#
#         if question_strategies is None:
#             question_strategies = QUESTION_STRATEGIES
#         self.strategies = question_strategies
#
#         self.stats = {
#             "valid_tables": 0
#         }
#
#     def prepare_prompt(self, table):
#         # Example of how to construct the input for the API
#         strategies_to_test = random.sample(list(self.strategies.items()), 3)
#         strategy_input = "\n".join([f"- {name}: {desc}" for name, desc in strategies_to_test])
#
#         user_input = f"""
#         TARGET STRATEGIES:
#         {strategy_input}
#
#         DATA:
#         Group Name: {table['Group name']}
#         Entity Column: {table['Entity column']}
#         Table Data: {table['Table data']}
#         """
#
#         return user_input
#
#     async def generate_questions(self):
#         # Prepare input:
#         model_inputs = [self.prepare_prompt(table) for table in self.input_tables]
#
#         # Call Model:
#         results = await self.model.async_run(model_inputs)
#         parsed_results = self._get_parsed_results(results)
#
#         self.structure_data(parsed_results)
#         self._print_summary()
#
#     @staticmethod
#     def _get_parsed_results(results):
#         parsed_results = []
#         for r in results:
#             try:
#                 output = json.loads(r)
#             except:
#                 output = None
#             parsed_results.append(output)
#
#         return parsed_results
#
#     def get_answer(self):
#         # Get answers and evidences from the data
#         pass
#
#     def structure_data(self, parsed_results):
#         """
#         Maps LLM-generated questions back to their source tables and
#         segregates successful generations from failed ones.
#         """
#         for table, result in zip(self.input_tables, parsed_results):
#             # Check if result exists and contains a valid question string
#             if result and result.get('question') is not None:
#                 self.valid_tables.append({
#                     "Group name": table['Group name'],
#                     "Entity column": table['Entity column'],
#                     "Strategy": result.get('selected_strategy'),
#                     "Question": result['question'],
#                     "Data column": result['data_column'],
#                     "Wikipedia page": table['Wikipedia page'],
#                     "Table data": table['Table data']
#                 })
#             else:
#                 # If the LLM returned the 'Null Output' or failed to parse
#                 self.bad_examples.append(table)
#
#         # Update state-level statistics
#         self.stats['valid_tables'] = len(self.valid_tables)
#
#     def _print_summary(self):
#         pass

import json
import random
import asyncio
from collections import defaultdict

from knowledge_maze.tools.api_tools import *

QUESTION_PROMPT = """
### ROLE
You are an expert Data Journalist and Logic Puzzle Creator. Your task is to generate a natural language question based on a specific "Group" and its "Table Data" using a provided "Strategy."

### TASK GUIDELINES
1. **FOLLOW THE STRATEGY**: You will be given a specific Strategy, its Definition, and Logical Steps. You MUST follow these steps to derive the question.
2. **CONTEXTUAL INDEPENDENCE**: Use the "Group Name" to establish the setting. The question must be self-contained (e.g., "Among the cities in the 'European Union' group..." rather than "Among the cities in this list...").
3. **DATA INTEGRITY**: 
   - Ensure the "data_column" exists exactly as written in the table.
   - If the data is missing values (N/A) or is too sparse to fulfill the specific logic of the strategy, return the NULL OUTPUT.
4. **NO META-TALK**: Do not mention "the table," "the data provided," or "the rows." The question should sound like it appears in a high-quality trivia book or a news article.

### LOGICAL REQUIREMENTS
- For **EXTREME** strategies: There must be a clear winner.
- For **FILTER** strategies: The filter value must exist in the data.
- For **COUNT** strategies: The answer must be a single integer derived from the rows.

### OUTPUT FORMAT (Strict JSON)
{
  "thought_process": "Briefly explain how you applied the logical steps to this specific data.",
  "selected_strategy": "The name of the strategy used",
  "question": "The natural language question",
  "data_column": "The exact header name used for the reasoning"
}

### NULL OUTPUT
If the data does not support the strategy, return:
{"thought_process": "Data too sparse for this strategy", "selected_strategy": null, "question": null, "data_column": null}
"""
NEW_STRATEGIES = {
    "COUNT_TOTAL": {
        "desc": "Count the total number of distinct entities in the group.",
        "example": "The table lists several hospitals founded in the 19th century. How many total hospitals are mentioned?",
        "steps": "1. Identify the core entity type. "
                 "2. Verify how many distinct rows represent these entities. "
                 "3. Formulate a question asking for the grand total."
    },
    "UNIQUE_DIVERSITY": {
        "desc": "Count the number of unique categories or types in a column.",
        "example": "Across the six listed books, how many different publishers are represented?",
        "steps": "1. Pick a categorical column (e.g., 'Publisher'). "
                 "2. Extract all unique values. "
                 "3. Ask for the count of these unique types."
    },
    "FILTER_COUNT_CATEGORICAL": {
        "desc": "Identify a categorical value and count entities matching it.",
        "example": "Out of the eight projects mentioned, how many are classified as 'In Progress'?",
        "steps": "1. Choose a column with repeating categories. "
                 "2. Select one specific category (e.g., 'In Progress'). "
                 "3. Ask how many rows match that value."
    },
    "FILTER_COUNT_RANGE": {
        "desc": "Count entities within a numerical or temporal boundary.",
        "example": "How many of the listed athletes were born between 1990 and 1995?",
        "steps": "1. Find a numeric/date column. "
                 "2. Define a range that includes some, but not all, entities. "
                 "3. Ask for the count within that range."
    },
    "GROUPED_LOOKUP": {
        "desc": "Filter by one attribute, then identify or count based on another.",
        "example": "Among the cities located in 'North America,' how many have a population exceeding 5 million?",
        "steps": "1. Filter rows by a category (e.g., Continent). "
                 "2. Apply a second filter (e.g., Population > 5M). "
                 "3. Ask for the count of the resulting subset."
    },
    "EXTREME_NUMERICAL": {
        "desc": "Identify the entity with the maximum or minimum numerical value.",
        "example": "Which of the listed technology companies had the highest revenue in the final quarter?",
        "steps": "1. Find a column with comparable numbers. "
                 "2. Identify the entity holding the Max/Min. "
                 "3. Ask which entity holds that specific record."
    },
    "EXTREME_TEMPORAL": {
        "desc": "Identify the entity with the earliest or latest date.",
        "example": "Which three landmarks were completed or inaugurated the earliest?",
        "steps": "1. Find a date/year column. "
                 "2. Sort to find the earliest/latest. "
                 "3. Ask for the identity of the top-ranked entity by time."
    },
    "TOP_K_RANKING": {
        "desc": "List a specific subset (e.g., Top 3) in sorted order.",
        "example": "List the three cars with the fastest 0-60 mph times in descending order.",
        "steps": "1. Sort the table by a numeric metric. "
                 "2. Select the top K (e.g., 3). "
                 "3. Ask the reader to list these entities in the correct order."
    },
    "FREQUENCY_MODE": {
        "desc": "Find which categorical value appears most frequently.",
        "example": "Which primary programming language is cited most often among these ten engineers?",
        "steps": "1. Count occurrences of values in a categorical column. "
                 "2. Identify the most frequent one (the mode). "
                 "3. Ask which value is the most common."
    },
    "COMPARATIVE_DIFFERENCE": {
        "desc": "Calculate the numerical difference between two specific entities.",
        "example": "What is the difference in seating capacity between the Grand Theatre and the City Playhouse?",
        "steps": "1. Pick two distinct entities. "
                 "2. Retrieve the same numerical attribute for both. "
                 "3. Ask for the difference between these two values."
    },
    "AGGREGATION": {
        "desc": "Calculate the sum or average of a numerical attribute.",
        "example": "What is the average initial production cost for the three models released in 2024?",
        "steps": "1. Select a numeric column. "
                 "2. Filter if necessary. "
                 "3. Ask for the average or total sum of those values."
    }
}

class QuestionGenerator:
    def __init__(self, input_tables, strategies=NEW_STRATEGIES):
        self.input_tables = input_tables
        self.strategies = strategies
        self.model = gptAPI(QUESTION_PROMPT,
                            output_format='string',
                            temperature=0.4)  # Lower temp for better logic

        self.valid_results = []  # Stores successfully generated questions
        self.stats = {
            "initial_tables": len(input_tables),
            "final_success": 0,
            "failed_after_3_tries": 0
        }

    def _prepare_prompt(self, table, strategy_name):
        strat = self.strategies[strategy_name]

        strat_block = f"""
        STRATEGY: {strategy_name}
        DEFINITION: {strat['desc']}
        EXAMPLE: {strat['example']}
        LOGICAL STEPS: {strat['steps']}
        """

        user_input = f"""
        {strat_block}

        DATA:
        Group Name: {table['Group name']}
        Entity Column: {table['Entity column']}
        Table Data: {table['Table data']}
        """
        return user_input

    async def generate_questions(self, max_rounds=3):
        # We track state per table: {table_index: [list_of_tried_strategies]}
        remaining_tables = {i: [] for i in range(len(self.input_tables))}

        for round_num in range(1, max_rounds + 1):
            if not remaining_tables:
                break

            print(f"--- Round {round_num}: Processing {len(remaining_tables)} tables ---")

            current_batch_indices = list(remaining_tables.keys())
            batch_prompts = []
            batch_strategies = []

            for idx in current_batch_indices:
                table = self.input_tables[idx]
                # Sample a strategy not yet tried for this table
                untried = [s for s in self.strategies.keys() if s not in remaining_tables[idx]]
                selected_strat = random.choice(untried)

                remaining_tables[idx].append(selected_strat)
                batch_prompts.append(self._prepare_prompt(table, selected_strat))
                batch_strategies.append(selected_strat)

            # Run API
            raw_responses = await self.model.async_run(batch_prompts)

            # Process results
            for i, response in enumerate(raw_responses):
                table_idx = current_batch_indices[i]
                strat_used = batch_strategies[i]

                parsed = self._parse_json(response)

                if parsed and parsed.get('question'):
                    # Success! Save and remove from remaining
                    self._store_valid(self.input_tables[table_idx], parsed)
                    del remaining_tables[table_idx]
                else:
                    # Failure - if it was the last round, move to bad_examples
                    if round_num == max_rounds:
                        del remaining_tables[table_idx]
                        self.stats["failed_after_3_tries"] += 1

        self.stats["final_success"] = len(self.valid_results)
        self._print_summary()
        return self.valid_results

    def _parse_json(self, text):
        try:
            # LLMs sometimes add conversational filler or markdown
            if "{" in text:
                json_str = text[text.find("{"):text.rfind("}")+1]
                return json.loads(json_str)
            return None
        except Exception:
            return None

    def _store_valid(self, table, result):
        """
        Creates a dictionary where keys are entities and values are the specific
        data points from the target column.
        """
        table_data = table.get('Table data', [])
        data_col = result.get('data_column')
        entity_col = table.get('Entity column')

        # Generate the Entity -> Value mapping
        entity_data_map = {}
        if data_col and entity_col:
            for row in table_data:
                entity_name = str(row.get(entity_col, "Unknown Entity"))
                value = row.get(data_col)
                # We only map if the entity name is meaningful
                if entity_name != "Unknown Entity":
                    entity_data_map[entity_name] = value

        self.valid_results.append({
            "group_name": table['Group name'],
            "strategy": result.get('selected_strategy'),
            "thought": result.get('thought_process'), # Added for debugging
            "question": result['question'],
            "answer_map": entity_data_map, # Key: Entity, Value: Data
            "metadata": {
                "wiki_page": table.get('Wikipedia page'),
                "target_column": data_col,
                "entity_column": entity_col
            }
        })

    def _print_summary(self):
        print("\n" + "═" * 50)
        print("🚀 QUESTION GENERATION COMPLETE")
        print("═" * 50)
        print(f"Initial Tables:      {self.stats['initial_tables']}")
        print(f"Successfully Gen:    {self.stats['final_success']}")
        print(f"Failed (3 Rounds):   {self.stats['failed_after_3_tries']}")
        print("═" * 50)
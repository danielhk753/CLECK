import json
import random
import pandas as pd
from IPython.display import display

from knowledge_maze.tools.api_tools import *

QUESTION_PROMPT = """
            ### ROLE\n
            You are an expert Data Journalist. Your task is to generate a complex, natural language question based on a 
            Wikipedia "Group" and its associated "Table Data."\n\n
            
            ### OBJECTIVES\n
            1. SELECT STRATEGY: You will be provided with two "Target Strategies." Evaluate the table data and select 
            the one that results in the most natural, logical, and answerable question.\n
            2. INTEGRATE CONTEXT: Use the "Group name" to set the scope. The question must be self-contained; a reader 
            should not need to see the table to understand the subject.\n
            3. DATA VALIDATION: If the table data is too sparse, corrupted, or simply does not support either strategy, 
            return the "Null Output" defined below.\n
            4. REASONING: The question must require the reader to perform the logic of the chosen strategy (e.g., 
            addition, finding a maximum, or counting) to arrive at the answer.\n\n
            
            ### CONSTRAINTS\n
            - NO TABLE JARGON: Do not say "In the table," "According to the row," or "For the following data."\n
            - IGNORE NOISE: Skip rows that are clearly sub-headers, "Total" rows, or contain "N/A."\n
            - PRECISION: Ensure the "data_column" you return is the exact header name used for the reasoning.\n\n
            
            ### NULL OUTPUT\n
            If no valid question can be formed, return exactly: 
            {"selected_strategy": None, "question": None, "data_column": None}\n\n
            
            ### OUTPUT FORMAT\n
            Return a JSON object:\n
            {
              "selected_strategy": "The name of the strategy used",
              "question": "The natural language question",
              "data_column": "The header name of the column used for reasoning"
            }\n\n
            """
QUESTION_STRATEGIES = {
    "COUNT_TOTAL": """Count the total number of distinct entities that belong to this specific group.""",
    "FILTER_COUNT_CATEGORICAL": """Identify a specific categorical attribute and count how many entities match a certain
                                value (e.g., 'How many players are from [Country]?').""",
    "FILTER_COUNT_RANGE": """Count how many entities fall within a numerical or temporal range (e.g., 'How many events 
                          occurred after [Year]?' or 'How many items cost more than [Price]?').""",
    "EXTREME_NUMERICAL": """Identify the specific entity that holds the maximum or minimum value in a numerical column 
                         (e.g., 'Which stadium has the highest capacity?').""",
    "EXTREME_TEMPORAL": """Identify the entity associated with the earliest or latest date/time (e.g., 'Who was the 
                        first person to win this award?').""",
    "NUMERICAL_AGGREGATION": """Calculate the sum or average of a numerical column across the entire group (e.g., 'What 
                             "is the average population of these cities?').""",
    "FREQUENCY_MODE": """Analyze a categorical column to find which value appears most frequently (e.g., 'Which 
                      nationality is most commonly represented in this list?').""",
    "COMPARATIVE_DIFFERENCE": """Select two specific entities and ask for the numerical difference between them (e.g., 
                              'How many more goals did [Player A] score than [Player B]?').""",
    "UNIQUE_DIVERSITY": """Count the number of unique categories or types present in a column (e.g., 'How many different 
                        "countries have representatives in this group?')."""
}

class QuestionGenerator:

    def __init__(self, input_tables,
                 question_prompt=QUESTION_PROMPT,
                 question_strategies=None):

        self.input_tables = input_tables
        self.model = gptAPI(question_prompt,
                            output_format='string',
                            temperature=0.7,
                            max_concurrent_requests=100,
                            max_rpm=4900)
        self.stats = {
            "question_candidates": len(self.input_tables),
            "questions_generated": 0,
        }
        self.valid_tables = []
        self.bad_examples = []

        if question_strategies is None:
            question_strategies = QUESTION_STRATEGIES
        self.strategies = question_strategies

        self.stats = {
            "valid_tables": 0
        }

    def prepare_prompt(self, table):
        # Example of how to construct the input for the API
        strategies_to_test = random.sample(list(self.strategies.items()), 3)
        strategy_input = "\n".join([f"- {name}: {desc}" for name, desc in strategies_to_test])

        user_input = f"""
        TARGET STRATEGIES:
        {strategy_input}

        DATA:
        Group Name: {table['Group name']}
        Entity Column: {table['Entity column']}
        Table Data: {table['Table data']}
        """

        return user_input

    async def generate_questions(self):
        # Prepare input:
        model_inputs = [self.prepare_prompt(table) for table in self.input_tables]

        # Call Model:
        results = await self.model.async_run(model_inputs)
        parsed_results = self._get_parsed_results(results)

        self.structure_data(parsed_results)
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

    def get_answer(self):
        # Get answers and evidences from the data
        pass

    def structure_data(self, parsed_results):
        """
        Maps LLM-generated questions back to their source tables and
        segregates successful generations from failed ones.
        """
        for table, result in zip(self.input_tables, parsed_results):
            # Check if result exists and contains a valid question string
            if result and result.get('question') is not None:
                self.valid_tables.append({
                    "Group name": table['Group name'],
                    "Entity column": table['Entity column'],
                    "Strategy": result.get('selected_strategy'),
                    "Question": result['question'],
                    "Data column": result['data_column'],
                    "Wikipedia page": table['Wikipedia page'],
                    "Table data": table['Table data']
                })
            else:
                # If the LLM returned the 'Null Output' or failed to parse
                self.bad_examples.append(table)

        # Update state-level statistics
        self.stats['valid_tables'] = len(self.valid_tables)

    def _print_summary(self):
        pass

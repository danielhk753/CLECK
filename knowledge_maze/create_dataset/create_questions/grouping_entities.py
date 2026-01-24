from knowledge_maze.tools.api_tools import *

GROUPING_PROMPT = """
Analyze the Wikipedia table to find a group of 3-10 entities. 

### STRATEGIES:
1. RANKING: Use "top_n" to get the 'latest', 'highest', or 'most' of something (e.g., Latest 7 chocolate factories).
2. SLICING: Use a range to capture a specific window.

### OUTPUT FORMAT:
Return a JSON object:
{
  "group_description": "Latest 7 chocolate manufacturers by founding date",
  "entity_column": "Manufacturer",
  "filter_column": "Year founded",
  "filter_type": "top_n",
  "filter_value": 7,
  "sort_order": "desc"
}
"""


def get_group(table):
    table_name = table['table_name']
    table_content = table['rows']

    model_input = str(f"""Table name: {table_name} \n
                          Table content: {table_content}""")

    # Create LLM API call objects:
    grouping_model = gptAPI(GROUPING_PROMPT,
                            output_format='dict',
                            temperature=0.7,
                            max_concurrent_requests=100,
                            max_rpm=4900)

    return grouping_model.run(model_input)
import random

from knowledge_maze.tools.api_tools import *


GROUPING_STRATEGIES = {
    "Categorical": [
        "Single-Column Match: Find a specific category (e.g., 'Genre' is 'Horror') that isolates 3-10 rows.",
        "Multi-Column Pivot: Combine two categorical columns (e.g., 'Country' is 'Italy' AND 'Status' is 'Active') to narrow the group to 3-10 rows.",
        "Exclusive Membership: Select entities belonging to a specific subset or organization mentioned in a category column."
    ],
    "Numerical": [
        "Top/Bottom Leaderboard: Use 'top_n' to extract the highest or lowest 3-10 entities based on a numeric value.",
        "Value-Range Slice: Use 'numerical_range' to find a specific 'middle' window of 3-10 entities (e.g., revenue between $50M and $100M).",
        "Threshold Filtering: Isolate 3-10 entities that fall above or below a specific significant threshold."
    ],
    "Temporal": [
        "Chronological Extreme: Isolate the 3-10 most recent or the 3-10 earliest entries in the table.",
        "Era/Period Slice: Use 'temporal_range' to capture 3-10 entities that occurred during a specific historical era or date window.",
        "Cutoff Logic: Filter for 3-10 entities that occurred specifically before or after a major milestone year."
    ],
    "Nested/List": [
        "Participant Search: Look inside cells containing lists (e.g., Cast, Signatories) to find a specific entity that appears in 3-10 different rows.",
        "Complexity Count: Filter by the 'volume' of items in a cell (e.g., entities that have more than 5 but fewer than 10 associated countries/affiliates).",
        "Shared Affiliation: Find a specific value within a multi-value string that is common to exactly 3-10 rows."
    ]
}

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


def get_sampling_prompt(table_name, table_content):
    # Sample one from each category to provide a diverse 'menu'
    sampled_blocks = [f"- [{cat}]: {random.choice(strats)}" for cat, strats in GROUPING_STRATEGIES.items()]
    random.shuffle(sampled_blocks)
    strategies_text = "\n".join(sampled_blocks)

    prompt = f"""Analyze the Wikipedia table and identify a group of 3-10 entities.

### YOUR MISSION:
1. Pick EXACTLY ONE strategy from the 'Strategy Menu' below that best fits the columns in this table.
2. Use that strategy to define a specific group of 3-10 entities.
3. Write a natural description that includes: The Count (if there is any) + The Subject + The Metric/Reason.

### STRATEGY MENU:
{strategies_text}

### DESCRIPTION RULES:
- NO QUOTING: Do not use '>' or copy the title "{table_name}" verbatim. Integrate it naturally.
- STANDALONE: The description must make sense on its own (e.g., 'The 4 longest-running animated shows on Colors TV').
- METRIC: If you rank things, you MUST say 'by [column name]'.

### INPUT DATA:
Table Name: {table_name}
Table Content: {table_content}

### OUTPUT FORMAT:
{{
  "group_description": "e.g., The 4 oldest former animated broadcasts on Colors TV by start year",
  "entity_column": "Column for names",
  "filter_column": "Column used for the logic",
  "filter_type": "top_n, numerical_range, temporal_range, or list",
  "filter_value": [Value],
  "sort_order": "desc/asc or null"
}}
"""
    return prompt


async def process_table_batches(tables: list):
    """
    Takes a list of table dicts and runs them asynchronously.
    """
    # 1. Prepare individual prompts for each table with unique strategy samples
    queries = []
    for table in tables:
        prompt = get_sampling_prompt(table['table_name'], table['rows'])
        queries.append(prompt)

    # 2. Initialize the model
    # Note: Using an empty string for prompt here because we pass full prompts in 'queries'
    grouping_model = gptAPI("",
                            output_format='dict',
                            temperature=0.7,
                            max_concurrent_requests=100,
                            max_rpm=4900)

    # 3. Run asynchronously
    results = await grouping_model.async_run(queries)

    return results


import re
from datetime import datetime


def extract_grouped_data(llm_data, table_rows):
    try:
        # Pydantic V2 migration safe
        data = llm_data.model_dump() if hasattr(llm_data, 'model_dump') else llm_data

        group_desc = data.get('group_description', "")
        ent_col_query = str(data.get('entity_column', "")).lower().strip()
        fil_col_query = str(data.get('filter_column', "")).lower().strip()
        f_type = str(data.get('filter_type', "")).lower()
        f_val = data.get('filter_value')
        # Default sort_order logic
        sort_order = data.get('sort_order')
        if not sort_order or sort_order == "null":
            sort_order = 'desc'

        if not table_rows: return {}

        # 1. Identify actual keys
        keys = list(table_rows[0].keys())
        entity_key = next((k for k in keys if ent_col_query in str(k).lower()), None)
        filter_key = next((k for k in keys if fil_col_query in str(k).lower()), None)

        if not entity_key or not filter_key: return {}

        # 2. Pre-process and Clean data
        processed_pool = []
        for row in table_rows:
            raw_ent = row.get(entity_key, "Unknown")
            raw_fil = row.get(filter_key, "")

            # Clean entity name
            name = str(raw_ent[0] if isinstance(raw_ent, (list, tuple)) else raw_ent).strip()

            # Numerical parsing (for sorting or range)
            clean_num_list = re.findall(r"[-+]?\d*\.\d+|\d+", str(raw_fil).replace(',', ''))
            val_as_float = float(clean_num_list[0]) if clean_num_list else 0.0

            # Temporal parsing (extract year for comparison)
            year_match = re.search(r'\b(1\d{3}|20\d{2})\b', str(raw_fil))
            val_as_year = int(year_match.group(0)) if year_match else 0

            processed_pool.append({
                "name": name,
                "numeric_val": val_as_float,
                "year_val": val_as_year,
                "display_val": raw_fil,
                "original_row": row.copy()
            })

        # 3. Comprehensive Filter/Sort Logic
        selected_matches = []

        # --- RANKING (Numerical or Temporal) ---
        if f_type == "top_n":
            # Use year_val if the column looks like a date, else numeric_val
            sort_key = "year_val" if any(d['year_val'] > 0 for d in processed_pool) else "numeric_val"
            sorted_pool = sorted(processed_pool, key=lambda x: x[sort_key], reverse=(sort_order == 'desc'))
            limit = min(max(int(f_val if f_val else 10), 3), 10)
            selected_matches = sorted_pool[:limit]

        # --- NUMERICAL RANGE ---
        elif f_type == "numerical_range":
            low, high = float(f_val[0]), float(f_val[1])
            selected_matches = [d for d in processed_pool if low <= d['numeric_val'] <= high][:10]

        # --- TEMPORAL RANGE ---
        elif f_type == "temporal_range":
            # Extract years from the filter value [ "2001", "2010" ] or [ 2001, 2010 ]
            start_y = int(str(f_val[0])[:4])
            end_y = int(str(f_val[1])[:4])
            selected_matches = [d for d in processed_pool if start_y <= d['year_val'] <= end_y][:10]

        # --- LIST / NESTED SEARCH ---
        elif f_type == "list":
            # f_val could be a single string or a list of strings
            targets = [str(t).lower() for t in (f_val if isinstance(f_val, list) else [f_val])]
            for d in processed_pool:
                cell_content = str(d['display_val']).lower()
                # Check if any target string exists inside the cell (e.g., "Tom Hanks" in a cast list)
                if any(t in cell_content for t in targets):
                    selected_matches.append(d)
            selected_matches = selected_matches[:10]

        # 4. Construct Final Output Dict
        group_entities = {}
        group_data = []

        for match in selected_matches:
            group_entities[match['name']] = match['display_val']
            pruned_row = match['original_row']
            if filter_key in pruned_row:
                del pruned_row[filter_key]
            group_data.append(pruned_row)

        return {
            "Group description": group_desc,
            "Group entities": group_entities,
            "Group data": group_data
        }

    except Exception as e:
        print(f"Extraction Error: {e}")
        return {}



# ================================================================================================================
def get_whole_table_prompt(table_hierarchy, column_names, sample_rows):
    return f"""You are a data architect. Rephrase a table's title into a professional, standalone group description.

### CONTEXT:
- Table Hierarchy: {table_hierarchy}
- Columns Available: {column_names}
- Sample Data: {sample_rows}

### RULES:
1. NO QUOTES: Do not use '>' or copy the title verbatim.
2. STANDALONE: The description must be a complete fact (e.g., 'The 8 current professional sports teams in Utah').
3. COUNT: You MUST include the total count ({len(sample_rows)}) in the description.
4. ENTITY COLUMN: Identify which column contains the names of the members of this group.

### OUTPUT FORMAT:
{{
  "group_description": "Natural standalone sentence with count",
  "entity_column": "The specific column name for entities"
}}
"""


async def process_small_tables_batch(tables: list):
    """
    tables: List of dicts with 'table_name' (hierarchy) and 'rows'
    """
    queries = []
    original_data_map = []

    for table in tables:
        rows = table.get('rows', [])
        if not rows: continue

        cols = list(rows[0].keys())
        # Provide 2 sample rows so the LLM understands the content
        sample = rows[:2]

        prompt = get_whole_table_prompt(table['table_name'], cols, sample)
        queries.append(prompt)
        original_data_map.append(table)

    # Initialize the model
    grouping_model = gptAPI("",
                            output_format='dict',
                            temperature=0.3,  # Lower temp for more consistent rephrasing
                            max_concurrent_requests=100)

    # Run batch
    results = await grouping_model.async_run(queries)

    # Final Assembly
    final_output = []
    for i, res in enumerate(results):
        if not res: continue

        table_ref = original_data_map[i]
        ent_col = res.get('entity_column')

        # Construct the final structure
        # Group entities is a dict of Name: EntityColumnValue
        # (Since we aren't filtering, we don't have a 'filter_column' value)
        group_entities = {str(row.get(ent_col, "Unknown")): "Member" for row in table_ref['rows']}

        final_output.append({
            "Group description": res.get('group_description'),
            "Group entities": group_entities,
            "Group data": table_ref['rows']  # The whole table is the data
        })

    return final_output
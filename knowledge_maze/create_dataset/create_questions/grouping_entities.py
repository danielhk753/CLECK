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
4.  ASYNCHRONOUS BATCH PROCESSING: Leverages `OpenRouterAPI` for high-concurrency
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
- knowledge_maze.tools.api_tools.OpenRouterAPI: Custom LLM interface.
- pandas/IPython: Used for statistical visualization and debugging output.
- json/random: Data handling and sampling.

Usage Example:
--------------
    tables = load_processed_jsonl("cleaned_tables.jsonl")
    constructor = TableGroupConstructor(tables)
    await constructor.get_groups()
    valid_groups = constructor.valid_tables
"""

import re
import json
import random
import aiohttp
import dateparser
import pandas as pd
from collections import Counter
from IPython.display import display, HTML

from knowledge_maze.tools.utils import *
from knowledge_maze.tools.api_tools import OpenRouterAPI

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
      
      ### ENTITY-NAME ALIGNMENT (CRITICAL):
      - The Group Name MUST accurately describe the actual data types inside the "Entity Column". Do not blindly copy headers if the data represents an attribute or a summary.
      - Mismatch Example 1: If the column lists "Political Parties", the name must be about "Parties", NOT "Members of Parliament".
      - Mismatch Example 2: If the column lists "Countries", the name must be about "Countries/Nationalities", NOT "Players".
      - Mismatch Example 3: If the column lists "Athletes", the name must be about "Athletes/Winners", NOT "Championships" or "Records" themselves.
      - Always verify: "Does my Group Name perfectly describe the items in the rows?" If not, adjust the name or reject the table.

      ### SCOPE ENFORCEMENT RULE:
      - The provided Header Chain acts as your primary "Filter". The Group Name MUST include the context from the final header in the chain.
      - If the chain starts with "List of Fruits" and ends in "Citrus", the group is "Citrus fruits", NOT "List of fruits".
      
      ### SPECIAL HANDLING: TOP-X LISTS
      - If the header chain ends in "Top {N}" (e.g., "Highest Grossing Films > Top 10"), you MUST incorporate the number into the description.
      - FORMAT: "Top {N} [Entities] [Context]"
      - EXAMPLE: "Top 10 highest-grossing films of all time"
      - ENTITY COLUMN: Even in ranked lists, the entity_column is the SUBJECT (e.g., "Film Title"), NOT the "Rank" or "Position" column.

      ### CONTEXT UTILIZATION (CAPTION & PRE-TEXT):
      - You may be provided with a "Caption" and "Additional Text" (text immediately preceding the table).
      - If present and informative (e.g., specifying a year, location, or strict definition), use them to refine the Group Name.
      - If they are missing (None), generic, or unhelpful, ignore them and rely on the Header Chain.

      ### GROUP COMPLETENESS CHECK
      - The Group Name must explain why these specific rows appear together.
      - If the rows appear to be a random subset of a larger category (e.g., "Cities in France"), the table is INVALID unless a clear filter exists (e.g., "Top 10", "Largest", "By population", "In 2024").
      
      ### MEMBER CONSISTENCY CHECK
      - Verify that the example entity values logically belong to the proposed Group Name.
      - If any entity clearly does not fit the group concept, mark the table as invalid.

      ### SPECIFICITY REQUIREMENT
      - The Group Name must describe a clear and objective category.
      - Avoid vague or subjective terms such as "famous", "important", "notable", or "popular".
      
      ### NATURAL CATEGORY CHECK
      - The group must represent a category that people would reasonably group together.
      - Reject artificial or trivial rules based purely on spelling patterns or arbitrary constraints.
      
      ### CONSERVATIVE VALIDATION
      - If the table structure or grouping rule is unclear, mark the table as INVALID rather than guessing.

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
      - A table of "Medal Winners" where entities are spread across multiple columns (e.g., "Gold", "Silver", "Bronze" columns instead of a single "Athlete" column). The primary entity must be in ONE distinct column.
"""
VALIDATOR_PROMPT = f"""
### ROLE
You are a strict Data Quality Auditor. Your task is to validate a structured table group extracted from Wikipedia. 
Determine if the sample represents a clean, meaningful, and well-defined entity group for a high-quality dataset.
Be conservative: reject doubtful samples rather than accepting them.

### SOURCE TRUST POLICY
Trust the provided Wikipedia Source and Group Name as the ground truth for the intent of the list. However, if the Entity 
Members or Column clearly contradict the source (e.g., the source is about "Rivers" but the entities are "Cities"), you must reject it.

### RULES FOR VALIDITY

1. **FATEFULNESS:** Members must logically match the Group Name. Reject if the description is misleading, overly broad, or contains unrelated concepts.
2. **CLOSED_SET:** The group must be coherent and complete. Reject random subsets (e.g., "6 random cities in France"). 
   *Exception:* Specific filters (e.g., "Top 10", "Oldest 5", "Members in 2024") are valid.
3. **SPECIFICALITY:** The group must have a narrow, clear concept. Reject vague names like "Famous movies" or "Important countries."
4. **NATURALNESS:** The category must be one that humans would reasonably group together. Reject arbitrary rules (e.g., "Cities starting with Q that have 3 vowels").
5. **CORRECT_ENTITY:** The Entity Column must contain the primary entities, not metadata (Rank, Year, ID) or attributes.

### OUTPUT FORMAT
Provide only a single Python-style tuple containing the decision and the explanation. Do not include any other text, markdown formatting, or headers.

Format: ("DECISION", "EXPLANATION")
For Acceptances: ("ACCEPT", "Reason why the group is valid and coherent.")
For Rejections: ("RULE_NAME", "Specific explanation of the violation.")
"""

class TableGroupConstructor:
    def __init__(self, input_tables, grouping_prompt=GROUPING_PROMPT):
        # --- PRE-PROCESS: Injecting "Top X" into the header chain ---
        self.input_tables = self._preprocess_top_x(input_tables)

        self.model = OpenRouterAPI(
            grouping_prompt,
            model='google/gemini-3-flash-preview',
            output_format='string',
            temperature=0.1,
            max_concurrent_requests=100,
            max_rpm=500
        )
        self.valid_tables = []
        self.bad_examples = []
        self.rejection_counts = Counter()
        self.stats = {"total_tables": len(input_tables), "valid_tables": 0}
        self.page_cache = {}

    @staticmethod
    def _preprocess_top_x(tables):
        """Detects ranked lists and injects the count into the header chain. Relevant for around 16% of data samples."""
        processed = []
        superlatives = ["tallest", "highest", "largest", "biggest", "longest", "shortest", "fastest", "slowest",
                        "oldest", "youngest", "lowest", "richest", "poorest", "deadliest", "busiest", "costliest",
                        "hottest", "brightest", "strongest", "cleanest", "closest", "narrowest", "earliest", "latest",
                        "greatest", "fewest", "top", "most", "grossing"]
        block_words = ["timeline"]


        for table in tables:
            rows = table.get('rows', [])
            if not rows:
                processed.append(table)
                continue

            header_chain = table.get('table_name', {}).get('header_chain', '').lower()
            first_col_name = list(rows[0].keys())[0]

            # Match your logic: "Rank" in first col OR superlatives in header
            is_ranked = ('rank' in first_col_name.lower() or
                         (any(x in header_chain for x in superlatives) and
                          not any(x in header_chain for x in block_words)))

            if is_ranked:
                count = len(rows)
                # Inject "Top X" into the chain so the LLM treats it as a scope constraint
                if f" > Top {count}" not in table['table_name']['header_chain']:
                    table['table_name']['header_chain'] += f" > Top {count}"

            processed.append(table)
        return processed

    def _post_filter_table(self, table, llm_res):

        # 1. LLM Validity Test:
        if not llm_res or not llm_res.get('is_valid'):
            self.rejection_counts["LLM Rejected"] += 1
            return False, llm_res.get('reasoning', "Rejected By LLM") if llm_res else "JSON Error"

        # 2. Proper Entity Column Check:
        ent_col = llm_res.get('entity_column')
        rows = table.get('rows', [])

        if not ent_col or ent_col not in rows[0]:
            self.rejection_counts["Missing Header or Entity Column"] += 1
            return False, f"Header '{ent_col}' not found."

        ent_vals = [str(r.get(ent_col, "")).strip() for r in rows]

        # 3. Full Entity Column Check:
        invalid_values = {"", "n/a", "na", "none", "null"}
        if any(str(v).strip().lower() in invalid_values for v in ent_vals):
            self.rejection_counts["Empty Entity Cells"] += 1
            return False, "Entity column has empty cells."

        # 4. Unique Entity Values Check:
        if len(set(ent_vals)) != len(ent_vals):
            self.rejection_counts["Duplicate Entities Found"] += 1
            return False, "Entity column contains duplicates."

        # 5. Make Sure Entities Are Textual Only:
        numeric_count = sum(1 for v in ent_vals if v.replace('.', '').replace(',', '').isdigit())
        date_count = sum(1 for v in ent_vals if v and str(v).strip().lower() not in ("n/a", "na") and dateparser.parse(str(v)) is not None)
        if (numeric_count + date_count) / len(ent_vals) > 0:
            self.rejection_counts["Entity Column has Numeric/Date Values"] += 1
            return False, "Entity column has numbers or dates."

        # 6. Top-X Consistency Check
        gen_name = llm_res.get('description', '')
        row_count = len(rows)

        # Look specifically for digits following "Top " (case-insensitive)
        top_match = re.search(r'(?i)top\s+(\d+)', gen_name)

        if top_match:
            claimed_num = int(top_match.group(1))
            if claimed_num != row_count:
                self.rejection_counts["Top-X Mismatch"] += 1
                return False, f"Rule: Name says Top {claimed_num}, but table has {row_count} rows."

        # 7. Entity Column Kill-List Test
        kill_entity_columns = ["Rank", "Date", "Year", "Position", "Points", "Height", "Depth", "Score", "Population"]
        if ent_col in kill_entity_columns:
            self.rejection_counts["Entity Column in Kill-List"] += 1
            return False, f"Entity column is not suitable for entities."

        # 8. Group Name Kill-List Test
        kill_group_name = ['notable', 'miscellaneous', 'timeline of', "various", "several", "summary of", "history of", "examples of", "other"]
        if any([a in gen_name.lower() for a in kill_group_name]):
            self.rejection_counts["Kill-List Keywords in Group Name"] += 1
            return False, f"Group name has keywords that suggests ambiguity or a partial group."

        return True, "Valid"

    async def get_groups(self):
        async with aiohttp.ClientSession() as session:

            model_inputs = []

            for table in self.input_tables:
                # The 'table_name' here now includes the injected "Top X"
                prompt_input = f"""TARGET:\n
                                   - Name: {table.get('table_name', {}).get('header_chain')}\n
                                   - Caption: {table.get('caption')}\n
                                   - Additional Text: {table.get('additional_text')}\n\n
                                   
                                   ROWS:\n
                                   {str(table['rows'][:5])}"""
                model_inputs.append(prompt_input)

            results = await self.model.async_run(model_inputs, desc=f"Creating {len(self.input_tables)} Entity Groups")
            parsed_results = [self._safe_json(r) for r in results]

            for table, res in zip(self.input_tables, parsed_results):
                passed, reason = self._post_filter_table(table, res)
                if passed:
                    self.valid_tables.append({
                        "Group name": res['description'],
                        "Entity column": res['entity_column'],
                        "Table data": table['rows'],
                        "Wiki url": table.get('wiki_url')
                    })
                else:
                    self.bad_examples.append({
                        "Group name": table.get('table_name')['header_chain'],
                        "Entity column": res['entity_column'] if res else "N/A",
                        "Rejection reason": reason,
                        "Table data": table['rows'],
                        "Wiki url": table.get('wiki_url')
                    })

            self.stats['valid_tables'] = len(self.valid_tables)
            self.display_summary()
            save_json("valid_groups", self.valid_tables)

    @staticmethod
    def _safe_json(text):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return None

    def display_summary(self) -> None:
        # 1. Generate Rejection Rows
        dist_rows = ""
        items = list(self.rejection_counts.items())
        for i, (reason, count) in enumerate(items):
            is_last = i == len(items) - 1

            # This creates a 1px line that is transparent for the first/last 20px
            # and semi-white in the middle.
            line_style = "none" if is_last else "linear-gradient(to right, transparent 20px, rgba(255,255,255,0.1) 20px, rgba(255,255,255,0.1) calc(100% - 20px), transparent calc(100% - 20px))"

            dist_rows += f"""
            <tr style="background: {line_style} bottom / 100% 1px no-repeat; background-color: #1e293b !important;">
                <td style="padding: 15px 20px; color: #cbd5e1 !important; text-align: left; border: none;">
                    <span style="font-weight: 500;">{reason}</span>
                </td>
                <td style="padding: 15px 20px; text-align: center; width: 60px; border: none;">
                    <div style="background: #ef444422; color: #f87171; padding: 3px 0; border-radius: 20px; font-weight: 800; border: 1px solid #ef444444; font-family: monospace; font-size: 12px; width: 40px; margin-left: auto;">
                        {count}
                    </div>
                </td>
            </tr>"""

        # 2. Generate Sample List
        # Filter the valid tables by length, then sample from the "short" ones
        short_tables = [t for t in self.valid_tables if len(t.get('Group name', '')) <= 75]
        # Use short_tables if available, otherwise fallback to regular to avoid an empty list
        pool = short_tables if short_tables else self.valid_tables
        valid_sample = random.sample(pool, min(7, len(pool))) if pool else []
        valid_list = ""
        for i, s in enumerate(valid_sample):
            is_last = i == len(valid_sample) - 1
            # We use 14px padding here to match the 14px padding in your dist_rows <td>
            border_style = "none" if is_last else "1px solid rgba(255, 255, 255, 0.1)"
            valid_list += f"""
            <li style="padding: 14px 0; border-bottom: {border_style}; color: #94a3b8; list-style: none;">
                <span style="color: #f8fafc; font-weight: 500;">{s['Group name']}</span>
            </li>"""

        if not valid_sample:
            valid_list = "<li style='color: #64748b; padding: 10px 0;'>No valid groups found</li>"

        html = f"""
        <div style="background-color: #0f172a; padding: 35px; border-radius: 24px; border: 1px solid #1e293b; color: #f8fafc; font-family: sans-serif; max-width: 950px; margin-left: 0; margin-right: auto; text-align: left;">

            <div style="text-align: center; margin-bottom: 35px;">
                <div style="display: inline-flex; align-items: center; gap: 12px;">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 12l10 5 10-5M2 17l10 5 10-5"></path>
                    </svg>
                    <h2 style="color: #38bdf8; margin: 0; font-weight: 800; font-size: 24px; text-transform: uppercase; letter-spacing: 1px;">
                        Entity Grouping Insights
                    </h2>
                </div>
                <div style="height: 4px; width: 60px; background: #38bdf8; margin: 12px auto; border-radius: 10px;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: stretch; gap: 15px; margin-bottom: 35px; font-family: sans-serif;">
    
                <div style="flex: 1; background: #1e293b; padding: 20px; border-radius: 20px; border: 1px solid #334155; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Total Input</div>
                    <div style="font-size: 32px; font-weight: 900; color: #f8fafc;">{self.stats['total_tables']}</div>
                </div>
            
                <div style="flex: 1; background: #064e3b; padding: 20px; border-radius: 20px; border: 1px solid #059669; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Valid Groups</div>
                    <div style="font-size: 32px; font-weight: 900; color: #34d399;">{self.stats['valid_tables']}</div>
                </div>
            
                <div style="flex: 1; background: #450a0a; padding: 20px; border-radius: 20px; border: 1px solid #991b1b; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Rejected</div>
                    <div style="font-size: 32px; font-weight: 900; color: #f87171;">{sum(self.rejection_counts.values())}</div>
                </div>
            
                <div style="flex: 1; background: #0c4a6e; padding: 20px; border-radius: 20px; border: 1px solid #0ea5e9; text-align: center; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #7dd3fc; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Success Rate</div>
                    <div style="font-size: 32px; font-weight: 900; color: #38bdf8;">{100 * self.stats['valid_tables'] / self.stats['total_tables']:.1f}<span style="font-size: 18px; margin-left: 2px;">%</span></div>
                </div>            
            </div>


            <div style="display: flex; gap: 30px; align-items: stretch;">

                <div style="flex: 1; display: flex; flex-direction: column;">
                    <h4 style="color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                        <span style="color: #f87171;">●</span> Rejection Reasons
                    </h4>
                    <div style="background-color: #1e293b; border-radius: 20px; border: 1px solid #334155; flex: 1; overflow: hidden; margin-top: 4px;">
                        <table style="width: 100%; border-collapse: collapse; margin: 0;">
                            <tbody>
                                {dist_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            
                <div style="flex: 1.4; display: flex; flex-direction: column;">
                    <h4 style="color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                        <span style="color: #34d399;">●</span> Random Samples
                    </h4>
                    <div style="background-color: #1e293b; padding: 0px 25px; border-radius: 20px; border: 1px solid #334155; flex: 1; box-sizing: border-box; margin-top: 4px; overflow: hidden;">
                        <ul style="font-size: 14px; padding-left: 0; list-style: none; margin: 0;">
                            {valid_list}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(html))

    def display_debug_samples(self, show_type="both", num_success=5, num_rejected=5):
        """
        Final Versatile Debugger:
        - Fixed height with custom scrollbar
        - Section headers for Valid vs Rejected
        - Deduction logic for unique rule samples
        - Expandable table previews
        """
        valid_pool = []
        rejected_pool = []

        # --- 1. PREPARE DATA ---
        if show_type in ["both", "valid"]:
            # Grab up to 10 random successes
            v_pool = random.sample(self.valid_tables,
                                   min(num_success, len(self.valid_tables))) if self.valid_tables else []
            for s in v_pool:
                valid_pool.append({**s, "type": "valid", "display_reason": "Passed All Checks"})

        if show_type in ["both", "rejected"]:
            llm_samples = []
            rule_samples = {}
            for s in self.bad_examples:
                reason = s.get("reason") or s.get("Rejection reason") or "Rejected"
                if "llm" in reason.lower() or "rejected by model" in reason.lower():
                    if len(llm_samples) < 5:
                        llm_samples.append({**s, "type": "rejected", "display_reason": reason})
                else:
                    if reason not in rule_samples:
                        rule_samples[reason] = {**s, "type": "rejected", "display_reason": reason}
            rejected_pool = (llm_samples + list(rule_samples.values()))[:num_rejected]

        # --- 2. GENERATE UI ---
        html_output = f"""
        <div style="background-color: #0f172a; padding: 25px; border-radius: 24px; border: 1px solid #1e293b; color: #f8fafc; font-family: sans-serif; max-width: 950px; margin-left: 0;">
            <style>
                .scroll_container {{ height: 600px; overflow-y: auto; padding-right: 10px; }}
                /* Custom Scrollbar */
                .scroll_container::-webkit-scrollbar {{ width: 8px; }}
                .scroll_container::-webkit-scrollbar-track {{ background: #0f172a; }}
                .scroll_container::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 10px; }}
                .scroll_container::-webkit-scrollbar-thumb:hover {{ background: #38bdf8; }}

                .section_header {{ 
                    position: sticky; top: 0; background: #0f172a; z-index: 10;
                    padding: 15px 0; margin-top: 20px; border-bottom: 2px solid #1e293b;
                    display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 800; text-transform: uppercase;
                }}
                .debug_card {{ padding: 20px; border-bottom: 1px solid #1e293b; }}
                .status_badge {{ padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 800; border: 1px solid; }}
                .debug_table {{ border-collapse: collapse; width: 100%; background-color: #1e293b; color: #cbd5e1; font-size: 12px; margin-top: 15px; border: 1px solid #334155; }}
                .debug_table th {{ background: #334155; color: #f8fafc; padding: 8px; text-align: left !important;}}
                .debug_table td {{ padding: 8px; border-bottom: 1px solid #334155; text-align: left !important;}}
                a:hover {{ text-decoration: underline !important; color: #7dd3fc !important; }}
                summary {{ cursor: pointer; color: #38bdf8; font-weight: 600; font-size: 13px; margin-top: 10px; list-style: none; }}
            </style>

            <div class="scroll_container">
        """

        # Helper to render cards
        def render_cards(items):
            cards_html = ""
            for it in items:
                is_valid = it['type'] == "valid"
                color = "#34d399" if is_valid else "#f87171"
                status = "ACCEPTED" if is_valid else "REJECTED"
                name = it.get("Group name") or (
                    it.get("name", {}).get("header_chain") if isinstance(it.get("name"), dict) else it.get("name"))
                table_data = it.get("Table data") or it.get("sample") or []
                df_html = pd.DataFrame(table_data).to_html(index=False, classes="debug_table")
                url = it.get("Wiki url") or s.get("url") or "#"
                ent_col = it.get("Entity column") or "None"

                cards_html += f"""
                <div class="debug_card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <a href="{url}" target="_blank" style="color: #38bdf8; text-decoration: none;">{name} 🔗</a>
                        <div class="status_badge" style="color: {color}; border-color: {color}44; background: {color}11;">{status}</div>
                    </div>
                    <div style="margin-top: 6px; color: {color}; font-family: monospace; font-size: 11px;">
                        Reason: {it['display_reason']}
                    </div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
                        Entity Column: <span style="color: #cbd5e1;">{ent_col}</span>
                    </div>
                    <details>
                        <summary>▶ View Source Data</summary>
                        {df_html}
                    </details>
                </div>
                """
            return cards_html

        # --- ADD VALID SECTION ---
        if valid_pool:
            html_output += f'<div class="section_header" style="color: #34d399;"><span>✨</span> Success Group Samples</div>'
            html_output += render_cards(valid_pool)

        # --- ADD REJECTED SECTION ---
        if rejected_pool:
            html_output += f'<div class="section_header" style="color: #f87171; margin-top: 40px;"><span>🚫</span> Unique Rejection Samples</div>'
            html_output += render_cards(rejected_pool)

        html_output += "</div></div>"
        display(HTML(html_output))


class GroupValidator:
    def __init__(self, input_data):
        self.input_data = input_data
        self.model = OpenRouterAPI(
            system_prompt=VALIDATOR_PROMPT,
            model="google/gemini-3.1-pro-preview",
            output_format='string',
            temperature=0.1,
            max_concurrent_requests=100,
            max_rpm=500
        )
        self.output_data = []

    @staticmethod
    def _prepare_validation_input(sample):
        entity_col = sample['Entity column']
        entities = [r.get(entity_col, "") for r in sample['Table data']]
        group_name = sample['Group name']
        first_row = sample['Table data'][0]

        return f"""### INPUT DATA
                    - **Group Name:** {group_name}
                    - **Entity Column:** {entity_col}
                    - **Entity Members:** {entities}
                    - **Group Size:** {len(entities)}
                    - **Row Example:** {first_row}""".strip()

    async def validate(self):
        if not self.input_data:
            pass
        else:
            prompts = [self._prepare_validation_input(s) for s in self.input_data]
            raw_responses = await self.model.async_run(prompts, desc="Evaluating Group Samples")
            parsed_responses = []
            for s in raw_responses:
                try:
                    parsed_responses.append(eval(s))
                except:
                    parsed_responses.append(('PARSE_ERROR', 'parsing error.'))

            for i, answer in enumerate(parsed_responses):
                sample = self.input_data[i]
                verdict = answer[0]
                reason = answer[1]

                sample['verdict'] = verdict
                sample['reason'] = reason

                self.output_data.append(sample)

            self._display_summary()

    def _display_summary(self):
        total = len(self.output_data)
        if total == 0:
            display(HTML("<div style='color: #f87171; padding: 20px;'>No data available to display.</div>"))
            return

        # 1. Logic & Data Preparation
        results = Counter([s['verdict'] for s in self.output_data])
        accepted = results.get('ACCEPT', 0)
        failed = total - accepted
        accuracy = (accepted / total * 100) if total > 0 else 0

        # Define the "Expected" error types for the legend, but also pull any others found
        guaranteed_types = ["CLOSED_SET", "FATEFULNESS", "CORRECT_ENTITY", "NATURALNESS", "SPECIFICALITY",
                            "PARSE_ERROR"]
        found_types = [v for v in results.keys() if v != 'ACCEPT']
        all_legend_types = sorted(list(set(guaranteed_types + found_types)))

        # Collect one sample for every error type that actually appeared
        error_samples = {}
        for item in self.output_data:
            v = item['verdict']
            if v != 'ACCEPT' and v not in error_samples:
                error_samples[v] = item

        # 2. Metric Cards (Top Row - Centered Text)
        stats_html = f"""
        <div style="display: flex; gap: 12px; margin-bottom: 25px;">
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Total Samples</div>
                <div style="font-size: 26px; font-weight: 800; color: #f8fafc;">{total}</div>
            </div>
            <div style="flex: 1; background: #064e3b; padding: 15px; border-radius: 12px; border: 1px solid #059669; text-align: center;">
                <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Success Rate</div>
                <div style="font-size: 26px; font-weight: 800; color: #34d399;">{accuracy:.1f}%</div>
            </div>
            <div style="flex: 1; background: #450a0a33; padding: 15px; border-radius: 12px; border: 1px solid #991b1b66; text-align: center;">
                <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Failed</div>
                <div style="font-size: 26px; font-weight: 800; color: #f87171;">{failed}</div>
            </div>
        </div>
        """

        # 3. Main Dashboard Container (1100px, Left Aligned)
        html_start = f"""
        <div style="width: 1100px; background-color: #0f172a; padding: 30px; border-radius: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; border: 1px solid #1e293b; margin-left: 0; text-align: left; box-sizing: border-box;">
            <h3 style="margin: 0 0 25px 0; font-size: 20px; color: #38bdf8; text-align: center; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">Validator Insight Dashboard</h3>
            {stats_html}

            <div style="display: flex; gap: 20px; align-items: stretch;">

                <div style="flex: 0.9; background: #1e293b; padding: 20px; border-radius: 18px; border: 1px solid #334155; display: flex; flex-direction: column;">
                    <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 20px; text-align: center;">Error Distribution</div>
                    <div id="validator_pie" style="width: 100%; flex-grow: 1; min-height: 450px;"></div>
                </div>

                <div style="flex: 2; background: #1e293b; border-radius: 18px; border: 1px solid #334155; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 15px 20px; background: #1e293b; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; display: flex; justify-content: space-between;">
                        <span>Failure Gallery (One Sample per Category)</span>
                        <span style="color: #64748b;">Showing {len(error_samples)} Categories</span>
                    </div>
                    <div style="max-height: 500px; overflow-y: auto; background: #1e293b;">
                        <table style="width: 100%; border-collapse: collapse; background: #1e293b;">
        """

        rows_html = ""
        # Sort samples so they appear in alphabetical order of verdict
        for v in sorted(error_samples.keys()):
            sample = error_samples[v]
            count = results.get(v, 0)

            rows_html += f"""
            <tr style="border-bottom: 1px solid #0f172a;">
                <td style="padding: 20px; text-align: left; background: #1e293b;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #f87171; font-weight: 900; font-size: 12px; text-transform: uppercase;">● {v}</span>
                            <span style="background: #ef444422; color: #f87171; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; border: 1px solid #ef444444;">{count} occurrences</span>
                        </div>
                        <a href="{sample.get('url', '#')}" target="_blank" style="color: #38bdf8; font-size: 11px; text-decoration: none; font-weight: 600;">Wikipedia Reference ↗</a>
                    </div>
                    <div style="color: #f8fafc; font-weight: 700; font-size: 16px; margin-bottom: 6px;">{sample.get('Group name', 'N/A')}</div>
                    <div style="color: #94a3b8; font-size: 13px; line-height: 1.6; margin-bottom: 12px; 
                                background: #0f172a; padding: 10px; border-radius: 8px; border-left: 3px solid #334155;">
                        {sample.get('reason', 'N/A')}
                    </div>
                    <div style="font-size: 11px; color: #38bdf8; font-family: 'Roboto Mono', monospace; background: #0f172a; padding: 5px 10px; border-radius: 6px; border: 1px solid #334155; display: inline-block;">
                        <span style="color: #475569;">Entity Column:</span> {sample.get('Entity column', 'N/A')}
                    </div>
                </td>
            </tr>
            """

        # 4. Chart Script (Forces all types into legend)
        chart_data = [["Type", "Count"]]
        for etype in all_legend_types:
            chart_data.append([etype, results.get(etype, 0)])

        html_end = f"""
                        </table>
                    </div>
                </div>
            </div>
        </div>
        <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
        <script type="text/javascript">
            google.charts.load('current', {{'packages':['corechart']}});
            google.charts.setOnLoadCallback(() => {{
                const data = google.visualization.arrayToDataTable({json.dumps(chart_data)});
                const container = document.getElementById('validator_pie');
                const chart = new google.visualization.PieChart(container);
                chart.draw(data, {{
                    backgroundColor: 'transparent',
                    colors: ['#f87171', '#fb923c', '#fbbf24', '#38bdf8', '#818cf8', '#f472b6', '#4ade80'],
                    legend: {{ 
                        position: 'bottom', 
                        alignment: 'center',
                        textStyle: {{ color: '#94a3b8', fontSize: 10 }},
                        pagingTextStyle: {{ color: '#94a3b8' }},
                        scrollArrowsColor: '#94a3b8'
                    }},
                    pieHole: 0.55,
                    chartArea: {{ width: '100%', height: '75%', top: 20 }},
                    pieSliceBorderColor: 'none',
                    pieSliceText: 'value',
                    pieSliceTextStyle: {{ fontSize: 11, fontWeight: 'bold' }},
                    tooltip: {{ textStyle: {{ fontSize: 12 }}, showColorCode: true }}
                }});
            }});
        </script>
        """

        display(HTML(html_start + rows_html + html_end))

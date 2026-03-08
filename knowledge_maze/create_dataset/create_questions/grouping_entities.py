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

import re
import json
import random
import aiohttp
import urllib.parse
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from IPython.display import display, HTML


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
      
      ### SPECIAL HANDLING: TOP-X LISTS
      - If the header chain ends in "Top {N}" (e.g., "Highest Grossing Films > Top 10"), you MUST incorporate the number into the description.
      - FORMAT: "Top {N} [Entities] [Context]"
      - EXAMPLE: "Top 10 highest-grossing films of all time"
      - ENTITY COLUMN: Even in ranked lists, the entity_column is the SUBJECT (e.g., "Film Title"), NOT the "Rank" or "Position" column.

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
      - A table of "Medal Winners" where entities are spread across multiple columns (e.g., "Gold", "Silver", "Bronze" columns instead of a single "Athlete" column). The primary entity must be in ONE distinct column.
"""

class TableGroupConstructor:
    def __init__(self, input_tables, grouping_prompt=GROUPING_PROMPT):
        # --- PRE-PROCESS: Injecting "Top X" into the header chain ---
        self.input_tables = self._preprocess_top_x(input_tables)

        self.model = gptAPI(grouping_prompt,
                            output_format='string',
                            temperature=0.1,
                            max_concurrent_requests=100,
                            max_rpm=4900,
                            api_provider="openrouter")
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
                        "greatest", "fewest", "best", "top", "most", "grossing"]


        for table in tables:
            rows = table.get('rows', [])
            if not rows:
                processed.append(table)
                continue

            header_chain = table.get('table_name', {}).get('header_chain', '').lower()
            first_col_name = list(rows[0].keys())[0]

            # Match your logic: "Rank" in first col OR superlatives in header
            is_ranked = 'rank' in first_col_name.lower() or any(x in header_chain for x in superlatives)

            if is_ranked:
                count = len(rows)
                # Inject "Top X" into the chain so the LLM treats it as a scope constraint
                if f" > Top {count}" not in table['table_name']['header_chain']:
                    table['table_name']['header_chain'] += f" > Top {count}"

            processed.append(table)
        return processed

    # async def fetch_wikipedia_map(self, session, wiki_link):
    #     if not wiki_link or wiki_link in self.page_cache:
    #         return self.page_cache.get(wiki_link, "No link provided.")
    #
    #     try:
    #         parsed = urllib.parse.urlparse(wiki_link)
    #         page_title = parsed.path.split('/')[-1]
    #         target_fragment = parsed.fragment
    #
    #         api_url = "https://en.wikipedia.org/w/api.php"
    #         params = {"action": "parse", "page": page_title, "prop": "text", "format": "json", "redirects": 1}
    #
    #         async with session.get(api_url, params=params) as resp:
    #             data = await resp.json()
    #             html = data.get('parse', {}).get('text', {}).get('*', '')
    #             soup = BeautifulSoup(html, 'html.parser')
    #
    #             doc_map = [f"PAGE TITLE: {page_title.replace('_', ' ')}", "=" * 20]
    #             target_section_reached = False
    #             target_level = 2
    #
    #             for element in soup.find_all(['h2', 'h3', 'h4', 'p', 'table']):
    #                 if element.name in ['h2', 'h3', 'h4']:
    #                     lvl = int(element.name[1])
    #                     txt = element.get_text(strip=True).replace('[edit]', '')
    #                     if target_section_reached and lvl <= target_level: break
    #                     doc_map.append(f"{'#' * lvl} {txt}")
    #                     if target_fragment and (
    #                             element.find('span', id=target_fragment) or element.get('id') == target_fragment):
    #                         target_section_reached = True
    #                         target_level = lvl
    #                 elif element.name == 'p':
    #                     txt = element.get_text(strip=True)
    #                     if txt: doc_map.append(f"**{txt}**" if element.find('b') or len(txt) < 60 else txt)
    #                 elif element.name == 'table':
    #                     marker = ">>>> [TARGET TABLE] <<<<" if target_section_reached else "[OTHER TABLE]"
    #                     doc_map.append(f"\n{marker}\n")
    #
    #             res = "\n".join(doc_map)
    #             self.page_cache[wiki_link] = res
    #             return res
    #     except:
    #         return "Error fetching context map."

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
        if any(not v for v in ent_vals):
            self.rejection_counts["Empty Entity Cells"] += 1
            return False, "Entity column has empty cells."

        # 4. Unique Entity Values Check:
        if len(set(ent_vals)) != len(ent_vals):
            self.rejection_counts["Duplicate Entities Found"] += 1
            return False, "Entity column contains duplicates."

        # 5. Make Sure Entities Are Textual Only:
        numeric_count = sum(1 for v in ent_vals if v.replace('.', '').replace(',', '').isdigit())
        if numeric_count / len(ent_vals) > 0:
            self.rejection_counts["Entity Column Contains Numeric Values"] += 1
            return False, "Entity column is mostly numbers."

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

        return True, "Valid"

    async def get_groups(self):
        print(f"🚀 Processing {len(self.input_tables)} tables...")
        async with aiohttp.ClientSession() as session:
            # map_tasks = [self.fetch_wikipedia_map(session, t.get('wiki_url')) for t in self.input_tables]
            # contexts = await asyncio.gather(*map_tasks)

            model_inputs = []
            # for table, ctx_map in zip(self.input_tables, contexts):
            for table in self.input_tables:
                # The 'table_name' here now includes the injected "Top X"
                prompt_input = f"""TARGET:\n
                                   - Name: {table.get('table_name', {}).get('header_chain')}\n
                                   - Caption: {table.get('caption')}\n
                                   - Additional Text: {table.get('additional_text')}\n\n
                                   
                                   ROWS:\n
                                   {str(table['rows'][:5])}"""
                model_inputs.append(prompt_input)
                # TODO: Fix the input so Caption and Additional text are inserted.
                # TODO: Discard the wikipedia page text and function - it always fails anyway.

            results = await self.model.async_run(model_inputs)
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
                        "Entity column": res['entity_column'],
                        "Rejection reason": reason,
                        "Table data": table['rows'],
                        "Wiki url": table.get('wiki_url')
                    })

            self.stats['valid_tables'] = len(self.valid_tables)
            self.display_summary()

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
        for reason, count in self.rejection_counts.items():
            dist_rows += f"""
            <tr style="background-color: #1e293b !important; border-bottom: 1px solid #334155 !important;">
                <td style="padding: 14px 12px; color: #cbd5e1 !important; text-align: left; background-color: #1e293b !important; border: none !important;">
                    <span style="font-weight: 500;">{reason}</span>
                </td>
                <td style="padding: 14px 12px; text-align: center; background-color: #1e293b !important; width: 60px; border: none !important;">
                    <div style="background: #ef444422; color: #f87171; padding: 4px 0; border-radius: 20px; font-weight: 800; border: 1px solid #ef444444; font-family: monospace; font-size: 13px; width: 45px; margin: 0 auto;">
                        {count}
                    </div>
                </td>
            </tr>"""

        # 2. Generate Sample List
        # Filter the valid tables by length, then sample from the "short" ones
        short_tables = [t for t in self.valid_tables if len(t.get('Group name', '')) <= 80]
        # Use short_tables if available, otherwise fallback to regular to avoid an empty list
        pool = short_tables if short_tables else self.valid_tables
        valid_sample = random.sample(pool, min(5, len(pool))) if pool else []
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

            <div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-bottom: 35px;">
                <div style="flex: 1; background: #1e293b; padding: 20px; border-radius: 20px; border: 1px solid #334155; text-align: center;">
                    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Total Input</div>
                    <div style="font-size: 32px; font-weight: 900;">{self.stats['total_tables']}</div>
                </div>
                <div style="flex: 1; background: #064e3b; padding: 20px; border-radius: 20px; border: 1px solid #059669; text-align: center;">
                    <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Valid Groups</div>
                    <div style="font-size: 32px; font-weight: 900; color: #34d399;">{self.stats['valid_tables']}</div>
                </div>
                <div style="flex: 1; background: #450a0a; padding: 20px; border-radius: 20px; border: 1px solid #991b1b; text-align: center;">
                    <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Rejected</div>
                    <div style="font-size: 32px; font-weight: 900; color: #f87171;">{sum(self.rejection_counts.values())}</div>
                </div>
            </div>

            <div style="display: flex; gap: 30px; align-items: stretch;">

                <div style="flex: 1; display: flex; flex-direction: column;">
                    <h4 style="color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                        <span style="color: #f87171;">●</span> Rejection Reasons
                    </h4>
                    <div style="background-color: #1e293b; border-radius: 20px; border: 1px solid #334155; flex: 1; overflow: hidden; margin-top: 4px;">
                        <table style="width: 100%; border-collapse: separate; border-spacing: 0; background-color: #1e293b !important;">
                            <tbody style="background-color: #1e293b !important;">
                                {dist_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            
                <div style="flex: 1.6; display: flex; flex-direction: column;">
                    <h4 style="color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">
                        <span style="color: #34d399;">●</span> Random Samples
                    </h4>
                    <div style="background-color: #1e293b; padding: 0px 25px; border-radius: 20px; border: 1px solid #334155; flex: 1; box-sizing: border-box; margin-top: 4px;">
                        <ul style="font-size: 14px; padding-left: 0; list-style: none; margin: 0;">
                            {valid_list}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(html))

    def display_debug_samples(self, show_type="both"):
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
                                   min(10, len(self.valid_tables))) if self.valid_tables else []
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
            rejected_pool = llm_samples + list(rule_samples.values())

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

# class TableGroupConstructor:
#     def __init__(self, input_tables, grouping_prompt=GROUPING_PROMPT):
#         self.input_tables = input_tables
#         self.model = gptAPI(grouping_prompt,
#                             output_format='string',
#                             temperature=0.1,  # Maximum consistency for naming
#                             max_concurrent_requests=100,
#                             max_rpm=4900)
#         self.valid_tables = []
#         self.bad_examples = []
#         self.stats = {"total_tables": len(input_tables), "valid_tables": 0}
#         self.page_cache = {}
#
#     async def fetch_wikipedia_map(self, session, wiki_link):
#         """Asynchronously maps the section hierarchy and local labels."""
#         if wiki_link in self.page_cache:
#             return self.page_cache[wiki_link]
#
#         try:
#             parsed = urllib.parse.urlparse(wiki_link)
#             page_title = parsed.path.split('/')[-1]
#             target_fragment = parsed.fragment
#
#             api_url = "https://en.wikipedia.org/w/api.php"
#             params = {"action": "parse", "page": page_title, "prop": "text", "format": "json", "redirects": 1}
#
#             async with session.get(api_url, params=params) as resp:
#                 data = await resp.json()
#                 html = data.get('parse', {}).get('text', {}).get('*', '')
#                 soup = BeautifulSoup(html, 'html.parser')
#
#                 doc_map = [f"PAGE TITLE: {page_title.replace('_', ' ')}", "=" * 20]
#                 target_section_reached = False
#                 target_level = 2
#
#                 for element in soup.find_all(['h2', 'h3', 'h4', 'p', 'table']):
#                     if element.name in ['h2', 'h3', 'h4']:
#                         lvl = int(element.name[1])
#                         txt = element.get_text(strip=True).replace('[edit]', '')
#
#                         # Stop scanning once we leave the target section
#                         if target_section_reached and lvl <= target_level:
#                             doc_map.append("[... Rest of page truncated ...]")
#                             break
#
#                         doc_map.append(f"{'#' * lvl} {txt}")
#
#                         if target_fragment and (
#                                 element.find('span', id=target_fragment) or element.get('id') == target_fragment):
#                             target_section_reached = True
#                             target_level = lvl
#
#                     elif element.name == 'p':
#                         txt = element.get_text(strip=True)
#                         if not txt: continue
#                         # Capture bold labels (like 'Points' in the UConn example)
#                         doc_map.append(f"**{txt}**" if element.find('b') or len(txt) < 60 else txt)
#
#                     elif element.name == 'table':
#                         marker = ">>>> [TARGET TABLE] <<<<" if target_section_reached else "[OTHER TABLE]"
#                         caption = element.find('caption')
#                         cap_txt = f" (Caption: {caption.get_text(strip=True)})" if caption else ""
#                         doc_map.append(f"\n{marker}{cap_txt}\n")
#
#                 result_map = "\n".join(doc_map)
#                 self.page_cache[wiki_link] = result_map
#                 return result_map
#         except:
#             return "Error fetching context map."
#
#     async def get_groups(self):
#         print(f"🚀 Grouping {len(self.input_tables)} tables using Entity-Focus logic...")
#
#         async with aiohttp.ClientSession() as session:
#             # 1. Fetch Context Maps
#             map_tasks = [self.fetch_wikipedia_map(session, self._generate_wiki_link(t.get('table_name', '')))
#                          for t in self.input_tables]
#             contexts = await asyncio.gather(*map_tasks)
#
#             # 2. Build LLM Inputs
#             model_inputs = []
#             for table, ctx_map in zip(self.input_tables, contexts):
#                 prompt_input = f"DOCUMENT MAP:\n{ctx_map}\n\nTABLE NAME: {table['table_name']}\nROWS: {str(table['rows'][:5])}"
#                 model_inputs.append(prompt_input)
#
#             # 3. Run LLM
#             results = await self.model.async_run(model_inputs)
#             parsed_results = [self._safe_json(r) for r in results]
#
#             # 4. Structure Final Output
#             self._structure_output(parsed_results)
#             self._print_summary()
#
#     def _structure_output(self, parsed_results):
#         for table, res in zip(self.input_tables, parsed_results):
#             if res and res.get('is_valid'):
#                 self.valid_tables.append({
#                     "Group name": res['description'],
#                     "Entity column": res['entity_column'],
#                     "Table data": table['rows'],
#                     "Wikipedia link": self._generate_wiki_link(table['table_name'])
#                 })
#             else:
#                 # Store the table PLUS the reason it failed
#                 self.bad_examples.append({
#                     "table_name": table['table_name'],
#                     "reason": res.get('reasoning', "Unknown rejection") if res else "LLM Parse Error",
#                     "sample_rows": table['rows'][:2]  # Store a snippet for display
#                 })
#         self.stats['valid_tables'] = len(self.valid_tables)
#
#     @staticmethod
#     def _safe_json(text):
#         try:
#             # Helper to find JSON block if model includes Markdown code blocks
#             start = text.find("{")
#             end = text.rfind("}") + 1
#             return json.loads(text[start:end])
#         except:
#             return None
#
#     @staticmethod
#     def _generate_wiki_link(table_name):
#         try:
#             parts = [p.strip() for p in table_name.split(">")]
#             page = parts[0].replace(" ", "_")
#             if len(parts) > 1:
#                 anchor = parts[1].replace(" ", "_").split("_(")[0]
#                 return f"https://en.wikipedia.org/wiki/{page}#{anchor}"
#             return f"https://en.wikipedia.org/wiki/{page}"
#         except:
#             return ""
#
#     def _print_summary(self):
#         width = 80
#         print("\n" + "═" * width)
#         print(f"📊 GROUPING REPORT".center(width))
#         print("═" * width)
#         print(f"  ✅ Valid Groups:   {self.stats['valid_tables']}")
#         print(f"  ❌ Invalid Tables: {len(self.bad_examples)}")
#         print("═" * width)
#
#         if self.valid_tables:
#             print(f"\n✨ SAMPLE VALID NAMES:")
#             for s in random.sample(self.valid_tables, min(3, len(self.valid_tables))):
#                 print(f"  → {s['Group name']}")
#
#         if self.bad_examples:
#             print("\n" + "─" * width)
#             print(f"🕵️ REJECTION DEBUGGER (Random Sample)".center(width))
#             print("─" * width)
#             for b in random.sample(self.bad_examples, min(3, len(self.bad_examples))):
#                 print(f"\n[TABLE]: {b['table_name']}")
#                 print(f"[REASON]: \033[91m{b['reason']}\033[0m")  # Prints reason in Red
#                 # Optional: Show the first row to see why it's not an entity
#                 print(f"[DATA]:   {b['sample_rows'][0] if b['sample_rows'] else 'Empty'}")
#             print("\n" + "═" * width)

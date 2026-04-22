"""
Knowledge Maze Dataset Annotation and Validation Module.

This module provides the `KnowledgeMazeAnnotator` class, designed for the automated
audit of "Knowledge Conflict" datasets. It specifically evaluates the epistemic
tension between rhetorical authority (Claims) and logical derivation (Evidence).

The annotator utilizes LLM-based reasoning to validate dataset samples against
five core research criteria:
1. **Logical Framing**: Ensuring questions target finite groups with objective answers.
2. **Strategic Fidelity**: Verifying that the question requires the intended reasoning logic.
3. **Double-Lie Consistency**: Ensuring counterfactual answers do not leak real-world truths.
4. **Claim Purity**: Validating that assertions are "naked" and devoid of supporting data.
5. **Reasoning Latency**: Confirming that evidence is logically sufficient but
   non-explicit.

The module includes a rich suite of visualization tools for Jupyter environments,
providing HTML-based failure reports with contextual paragraph debugging and
Matplotlib-based bottleneck dashboards.

Typical usage example:
    annotator = KnowledgeMazeAnnotator(model_name="google/gemini-3-flash-preview")
    results, report = await annotator.annotate_dataset(q_dataset, p_dataset, strategy_defs, limit=50)
"""

import re
import json
import random
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from knowledge_maze.tools.api_tools import OpenRouterAPI


class KnowledgeMazeAnnotator:
    def __init__(self, model_name: str = "google/gemini-3-flash-preview"):
        # Refined System Instruction with Strategy Matching and Latency Clarification
        self.system_instruction = """
        ROLE: Expert NLP Research Auditor for "Knowledge Conflict" (Authority vs. Evidence).

        You are an expert Epistemic Auditor. Your task is to validate a single "Knowledge Conflict" dataset sample against five strict criteria. 
        
        A sample consists of:
        1. **Question**
        2. **Group name** (The group that the question should refer to)
        3. **Strategy** (The intended reasoning type)
        4. **Claim Paragraph** (A contextually asserted answer)
        5. **Evidence Paragraph** (A contextually derived answer)
        6. **Memory Answer** (The real-world ground truth)
        
        ---
        
        ### VALIDATION RULES
        
        #### 1. Logical Framing & Group Scope
        - **Criteria:** The question must name a closed, finite group and seek a single, objective winner. The property queried must logically apply to the group.
        - **Fail if:** The group is missing, the question is vague, or the predicate is nonsensical (e.g., "What color is the 'EU State'?").
        
        #### 2. Strategic Fidelity
        - **Criteria:** The question must require the specific logic of the assigned [Strategy].
        - **Fail if:** The strategy is "Top-K" but the question is a simple "Yes/No" or "Lookup" task.
        
        #### 3. Semantic Plausibility & Conflict (The "Double-Lie")
        - **Criteria:** Both the [Claim Answer] and [Evidence Answer] must be different from the [Memory Answer] AND different from each other. They must also be semantically plausible (e.g., if the answer is a city, all answers must be cities).
        - **Fail if:** Any answer matches reality, or if answers are categorical errors (e.g., a number is given as a city).
        
        #### 4. Claim Purity (The "Bare Assertion")
        - **Criteria:** The Claim Paragraph must state the answer as a naked assertion. It may use rhetorical "authority" (e.g., "According to X," "Researchers confirm," "It is well-established that...") to increase confidence, but it must contain zero external factual evidence.
        - **Fail if:** The paragraph includes specific data points relevant to the group or the question—such as dates, locations, occupations, or names of other group members—that serve to justify or verify the answer. It must remain a "naked" claim, regardless of how much rhetorical fluff surrounds it.
       
        #### 5. Reasoning Latency & Logical Sufficiency
        - **Criteria:** You must verify two specific things:
            1. **Sufficiency:** Do the clues in the Evidence actually lead to the intended answer without a doubt?
            2. **Latency:** Does it reach the answer WITHOUT ever stating the answer explicitly as the solution to the question?
        - **Fail if:** The answer is present explicitly as the solution to the question, or the clues are too vague to derive the answer.
        
        ---
        
        ### OUTPUT FORMAT
        You must respond with a JSON object. For each rule, provide a "thought" explaining your evaluation before the "status".
        
        ```json
        {
          "evaluations": {
            "rule_1_logic": {"thought": "...", "status": "PASS/FAIL"},
            "rule_2_strategy": {"thought": "...", "status": "PASS/FAIL"},
            "rule_3_conflict": {"thought": "...", "status": "PASS/FAIL"},
            "rule_4_purity": {"thought": "...", "status": "PASS/FAIL"},
            "rule_5_latency": {"thought": "...", "status": "PASS/FAIL"}
          },
          "final_verdict": "PASS/FAIL",
          "reason_for_rejection": "If any rule failed, explain why here. Otherwise null."
        }`
        """

        self.api = OpenRouterAPI(
            system_prompt=self.system_instruction,
            model=model_name,
            temperature=0.0,
            max_concurrent_requests=15
        )

    def format_sample(self, q: dict, p_dataset: dict, strategy_defs: dict) -> str:
        """Formats the data into the research audit prompt."""
        question = q['Question']
        group = q['Group name']
        ans = q['Answer']
        claim_para = p_dataset[q['Claim paragraph']]
        evidence_para = p_dataset[q['Evidence paragraph']]

        return f"""Group name: {group}
                   Selected strategy: {q['Selected strategy']} - {strategy_defs['strategies'][q['Selected strategy']]['definition']}
                   Question: {question}
                   Answers: 
                    - Model's memory: {ans['Memory']}
                    - Claim supported: {ans['Claim']}
                    - Evidence supported: {ans['Evidence']}
                   
                   ---
                   Claim Paragraph: ({claim_para['Confidence']} confidence)
                   {claim_para['Text']}
                   
                   ---
                   Evidence Paragraph: ({len(q.get('Fake data', []))} evidences)
                   {evidence_para['Text']}
                   """

    async def annotate_dataset(self, q_dataset: list, p_dataset: dict, strategy_defs, limit: int):
        # 1. Slice and Prepare Prompts
        dataset_to_process = random.sample(q_dataset, limit)

        # Assuming your format_sample method now injects the 5-Point Protocol prompt
        prompts = [self.format_sample(q, p_dataset, strategy_defs) for q in dataset_to_process]

        # 2. Execute Async API calls
        raw_results = await self.api.async_run(prompts, desc="Auditing Knowledge Maze")

        final_results = []
        failure_stats = Counter()

        for q, raw_resp in zip(dataset_to_process, raw_results):
            annotation = self._parse_json(raw_resp)

            # 3. Decision Mapping
            # The prompt uses "final_verdict" (PASS/FAIL)
            verdict = annotation.get("final_verdict", "FAIL")

            # 4. Extract Failure Bottlenecks
            # We navigate the 'evaluations' object to find statuses
            if verdict == "FAIL":
                evals = annotation.get("evaluations", {})
                for rule_name, rule_data in evals.items():
                    if rule_data.get("status") == "FAIL":
                        failure_stats[rule_name] += 1

            final_results.append({
                "question_id": q.get('id', q.get('Question', 'unknown')),
                "original_data": q,
                "annotation": annotation,
                "passed": verdict == "PASS"
            })

        # 5. Generate Comprehensive Bottleneck Report
        total = len(dataset_to_process)
        total_passed = sum(1 for r in final_results if r['passed'])
        total_failed = total - total_passed
        pass_rate = (total_passed / total) * 100 if total > 0 else 0

        # Breakdown of failures (What % of the FAILURES does each rule represent?)
        failure_breakdown = {
            rule: f"{(count / total_failed * 100):.1f}% of failures" if total_failed > 0 else "0%"
            for rule, count in failure_stats.items()
        }

        report = {
            "summary": {
                "total_samples": total,
                "pass_rate": f"{pass_rate:.1f}%",
                "total_failed": total_failed
            },
            "error_distribution": failure_breakdown
        }

        self.display_validation_dashboard(final_results, report, failure_stats, strategy_defs)
        self.display_audit_failures(final_results, p_dataset)

        return final_results, report

    def _parse_json(self, text):
        """
        Robustly extracts JSON from the LLM response, handling markdown blocks
        and potential trailing text.
        """
        try:
            # Use regex to find content between ```json and ``` or just { ... }
            json_match = re.search(r'\{.*}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {
                "final_verdict": "FAIL",
                "reason_for_rejection": "JSON_PARSE_ERROR",
                "evaluations": {}
            }

    def display_audit_failures(self, final_results: list, p_dataset: dict):
        """
        Renders a high-visibility HTML report for auditing.
        - Large color-coded answer badges.
        - Wikipedia button.
        - Contextual paragraph display (Rule 4/5 specific).
        """
        failed_samples = [res for res in final_results if not res.get("passed")]

        if not failed_samples:
            display(HTML(
                "<div style='padding: 20px; background: #d4edda; color: #155724; border-radius: 12px; font-weight: bold; text-align: center;'>🎉 All samples passed the audit!</div>"))
            return

        RULE_NAME_MAP = {
            "rule_1_logic": "Logical Framing & Group Scope",
            "rule_2_strategy": "Strategic Fidelity",
            "rule_3_conflict": "Semantic Plausibility & Conflict",
            "rule_4_purity": "Claim Purity (Bare Assertion)",
            "rule_5_latency": "Reasoning Latency & Logical Sufficiency"
        }

        html_content = """<div style="font-family: 'Segoe UI', Tahoma, sans-serif; margin-bottom: 25px;">
                            <h2 style="color: #c0392b; border-bottom: 3px solid #e74c3c; padding-bottom: 10px;">🚨 Knowledge Maze Audit: Failures</h2>
                          </div>"""

        for idx, sample in enumerate(failed_samples):
            orig = sample.get("original_data", {})
            ans = orig.get("Answer", {})
            q = orig.get("Question", "Unknown Question")

            # 1. Wiki Button Styling
            wiki_url = orig.get('Wiki link', '#')
            wiki_btn = f'''<a href="{wiki_url}" target="_blank" style="text-decoration: none; background: #3498db; color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">🌐 Wikipedia</a>'''

            evals = sample.get("annotation", {}).get("evaluations", {})
            failed_rule_keys = [k for k, v in evals.items() if v.get("status") == "FAIL"]

            # Determine Paragraph Visibility
            show_claim = "rule_4_purity" in failed_rule_keys
            show_evidence = "rule_5_latency" in failed_rule_keys

            # 2. Build Failure Explanation Blocks
            failures_html = ""
            for rule_name, rule_data in evals.items():
                if rule_data.get("status") == "FAIL":
                    natural_name = RULE_NAME_MAP.get(rule_name, rule_name.replace("_", " ").title())
                    failures_html += f"""
                    <div style="margin-top: 10px; padding: 12px; background: #fff; border-left: 5px solid #e74c3c; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                        <div style="color: #c0392b; font-weight: bold; font-size: 0.95em; margin-bottom: 5px;">❌ {natural_name}</div>
                        <div style="font-size: 0.9em; color: #333; line-height: 1.5;"><b>Reasoning:</b> {rule_data.get('thought')}</div>
                    </div>"""

            # 3. Build Contextual Paragraphs
            context_html = ""
            if show_claim:
                txt = p_dataset.get(orig.get('Claim paragraph'), {}).get('Text', 'N/A')
                context_html += f'<div style="margin-top:12px; padding:15px; background:#fff9db; border:1px solid #fab005; border-radius:8px; font-size:0.9em; color:#666;"><b>🔍 Claim Context (Rule 4 Failure):</b><br><i style="display:block; margin-top:5px; color:#222;">"{txt}"</i></div>'

            if show_evidence:
                txt = p_dataset.get(orig.get('Evidence paragraph'), {}).get('Text', 'N/A')
                context_html += f'<div style="margin-top:12px; padding:15px; background:#e7f5ff; border:1px solid #339af0; border-radius:8px; font-size:0.9em; color:#666;"><b>🔍 Evidence Context (Rule 5 Failure):</b><br><i style="display:block; margin-top:5px; color:#222;">"{txt}"</i></div>'

            # Card assembly
            html_content += f"""
            <div style="border: 1px solid #dee2e6; background: #ffffff; border-radius: 12px; margin-bottom: 30px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                <div style="background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #dee2e6; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #212529; font-size: 1.15em;">{idx + 1}. {q}</span>
                    {wiki_btn}
                </div>

                <div style="padding: 20px;">
                    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                        <div style="flex: 1; text-align: center; padding: 10px; background: #f1f3f5; border-radius: 8px; border-bottom: 4px solid #ced4da;">
                            <div style="font-size: 0.7em; text-transform: uppercase; color: #868e96; font-weight: 800;">Memory</div>
                            <div style="font-size: 1.3em; font-weight: bold; color: #343a40;">{ans.get('Memory')}</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 10px; background: #fff4e6; border-radius: 8px; border-bottom: 4px solid #ffd8a8;">
                            <div style="font-size: 0.7em; text-transform: uppercase; color: #e67e22; font-weight: 800;">Claim</div>
                            <div style="font-size: 1.3em; font-weight: bold; color: #d9480f;">{ans.get('Claim')}</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 10px; background: #e7f5ff; border-radius: 8px; border-bottom: 4px solid #a5d8ff;">
                            <div style="font-size: 0.7em; text-transform: uppercase; color: #228be6; font-weight: 800;">Evidence</div>
                            <div style="font-size: 1.3em; font-weight: bold; color: #1864ab;">{ans.get('Evidence')}</div>
                        </div>
                    </div>

                    {failures_html}
                    {context_html}
                </div>
            </div>"""

        display(HTML(html_content))

    def display_validation_dashboard(self, final_results, report, failure_stats, strategy_defs):
        """
        Renders a vertical dashboard:
        1. Metrics Cards
        2. Rule Bottlenecks (Fixed list including 0s)
        3. Strategy Success Rates (Fixed list including 0s)
        """
        summary = report['summary']
        pass_rate_val = float(summary['pass_rate'].replace('%', ''))

        # 1. HTML Metric Cards
        pass_color = "#27ae60" if pass_rate_val > 80 else "#f39c12" if pass_rate_val > 50 else "#e74c3c"
        html_cards = f"""
        <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Total Samples</div>
                <div style="font-size: 2.2em; font-weight: bold; color: #2c3e50;">{summary['total_samples']}</div>
            </div>
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {pass_color};">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Pass Rate</div>
                <div style="font-size: 2.2em; font-weight: bold; color: {pass_color};">{summary['pass_rate']}</div>
            </div>
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #e74c3c;">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Total Failed</div>
                <div style="font-size: 2.2em; font-weight: bold; color: #c0392b;">{summary['total_failed']}</div>
            </div>
        </div>
        """
        display(HTML(html_cards))

        # 2. Plots Configuration
        RULE_NAME_MAP = {
            "rule_1_logic": "Logical Framing",
            "rule_2_strategy": "Strategic Fidelity",
            "rule_3_conflict": "Semantic Conflict",
            "rule_4_purity": "Claim Purity",
            "rule_5_latency": "Reasoning Latency"
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 14))
        sns.set_theme(style="whitegrid")

        # --- Chart 1: Failures by Rule (Silencing FutureWarning) ---
        rule_data = [{"Rule": r_name, "Count": failure_stats.get(r_key, 0)} for r_key, r_name in RULE_NAME_MAP.items()]
        df_rules = pd.DataFrame(rule_data)

        sns.barplot(data=df_rules, y="Rule", x="Count", hue="Rule", palette="flare", ax=ax1, legend=False)
        ax1.set_title("Total Failure Count by Rule", loc='left', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlim(0, max(df_rules['Count'].max() + 1, 5))

        # --- Chart 2: Success Rate by Strategy (Success Ratio labels) ---
        strat_stats = []
        all_possible_strats = list(strategy_defs['strategies'].keys())

        for strat in all_possible_strats:
            strat_samples = [s for s in final_results if s['original_data'].get('Selected strategy') == strat]
            total_strat = len(strat_samples)
            passed_strat = sum(1 for s in strat_samples if s['passed'])
            success_rate = (passed_strat / total_strat * 100) if total_strat > 0 else 0

            strat_stats.append({
                "Strategy": strat,
                "Success Rate (%)": success_rate,
                "Passed": passed_strat,
                "Total": total_strat
            })

        df_strat = pd.DataFrame(strat_stats)
        sns.barplot(data=df_strat, x="Strategy", y="Success Rate (%)", hue="Strategy", palette="viridis", ax=ax2,
                    legend=False)

        ax2.axhline(90, color='red', linestyle='--', alpha=0.3, label="Target (90%)")
        ax2.set_title("Success Rate per Reasoning Strategy", loc='left', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylim(0, 115)  # Room for annotations

        # Format labels: 45-degree angle
        ax2.set_xticks(range(len(df_strat)))
        ax2.set_xticklabels(df_strat["Strategy"], rotation=45, ha='right')

        # Annotate bars with Success/Total ratio (e.g., 8/10)
        for i, p in enumerate(ax2.patches):
            row = df_strat.iloc[i]
            ratio_text = f"{int(row['Passed'])}/{int(row['Total'])}"
            percentage_text = f"{row['Success Rate (%)']:.1f}%"

            # Place the percentage and the fraction on top
            ax2.annotate(f"{percentage_text}\n({ratio_text})",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color='#2c3e50', xytext=(0, 15), textcoords='offset points')

        plt.tight_layout(pad=5.0)
        plt.show()

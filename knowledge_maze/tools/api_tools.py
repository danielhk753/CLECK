"""
Asynchronous OpenRouter API utilities

This module provides tools for interacting with OpenRouter models, managing
concurrency, and dynamically tracking real-time costs and token usage (including thinking tokens).
"""

import os
import time
import json
import asyncio
from typing import List
from pathlib import Path
from datetime import date

import tqdm.auto as tqdm_lib
from pydantic import BaseModel
from dotenv.main import load_dotenv
from openai import AsyncOpenAI, LengthFinishReasonError

# Load API keys from .env file:
load_dotenv()


# --- NEW: Persistence Logic ---
USAGE_FILE = Path("usage_log.json")

def load_persistent_usage():
    if not USAGE_FILE.exists():
        return {"all_time": 0.0, "daily": {}}
    with open(USAGE_FILE, "r") as f:
        return json.load(f)

def save_persistent_usage(total_all_time, daily_dict):
    with open(USAGE_FILE, "w") as f:
        json.dump({"all_time": total_all_time, "daily": daily_dict}, f)


# Structured response models for LLM outputs:
class StringResponse(BaseModel):
    Output: str


class ListResponse(BaseModel):
    Output: List[str]


class BoolResponse(BaseModel):
    Output: bool


class CoTResponse(BaseModel):
    explanation: str
    answer: str


OUTPUT_FORMATS = {
    'string': StringResponse,
    'list': ListResponse,
    'bool': BoolResponse,
    'cot': CoTResponse
}


class OpenRouterAPI:
    """
    Asynchronous OpenRouter API client with dynamic cost tracking and telemetry.

    Args:
        system_prompt (str): The system message to guide responses.
        model (str, optional): The OpenRouter model to use.
        output_format (str, optional): The response format ('string', 'list', 'bool').
        temperature (float, optional): The model's randomness (default: 0.2).
        max_concurrent_requests (int, optional): Max parallel API calls (default: 10).
        max_rpm (int, optional): Max requests per minute (default: 500).
        return_thinking (bool, optional): If True, returns a dict containing both the output and the reasoning text.
    """

    def __init__(self,
                 system_prompt: str,
                 model: str = "google/gemini-3-flash-preview",
                 output_format: str = None,
                 temperature: float = 0.2,
                 max_concurrent_requests: int = 10,
                 max_rpm: int = 500,
                 max_tokens: int = 4096,
                 return_thinking: bool = False):

        self.system_prompt = system_prompt
        self.model = model
        self.output_format = OUTPUT_FORMATS[output_format] if output_format in OUTPUT_FORMATS else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.return_thinking = return_thinking

        # OpenRouter Setup
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.base_url = "https://openrouter.ai/api/v1"

        # Concurrency & Rate Limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_rpm = max_rpm
        self.start_time = 0

        # Dynamic Telemetry State
        self.stats = {
            'requests': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'thinking_tokens': 0,
            'total_cost': 0.0
        }

        self.persistence = load_persistent_usage()
        self.today = str(date.today())
        if self.today not in self.persistence["daily"]:
            self.persistence["daily"][self.today] = 0.0

    def _get_client(self):
        """Creates an AsyncOpenAI client configured strictly for OpenRouter."""
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _extract_telemetry(self, response) -> None:
        """Dynamically extracts token counts and costs from the OpenRouter response."""
        self.stats['requests'] += 1

        # 1. Extract Token Counts
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.stats['input_tokens'] += getattr(usage, 'prompt_tokens', 0)
            self.stats['output_tokens'] += getattr(usage, 'completion_tokens', 0)

            # OpenAI/OpenRouter standard for thinking tokens
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                self.stats['thinking_tokens'] += getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)

        # 2. Extract Dynamic Cost
        # OpenRouter often injects 'cost' or 'total_cost' into the raw dict response or provider_extra
        raw_dict = response.model_dump()
        cost = 0.0

        # Safely hunt for the cost field in common OpenRouter locations
        if 'total_cost' in raw_dict:
            cost = raw_dict['total_cost']
        elif 'usage' in raw_dict and isinstance(raw_dict['usage'], dict):
            cost = raw_dict['usage'].get('cost', 0.0)  # OpenRouter occasionally nests it here

        self.stats['total_cost'] += float(cost)

        # Update Persistent Stats
        self.persistence["all_time"] += cost
        self.persistence["daily"][self.today] += cost
        save_persistent_usage(self.persistence["all_time"], self.persistence["daily"])

    async def get_response(self, user_prompt: str, progress_bar=None):
        """Fetches response and updates live telemetry."""
        async with self.semaphore:
            client = self._get_client()

            try:
                # OpenRouter natively supports standard OpenAI structured parsing
                # 1. Prepare the standard payload
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                }

                # 2. Only inject "Reasoning" keys if return_thinking is True
                if self.return_thinking:
                    payload["extra_body"] = {
                        "include_reasoning": True,
                        "reasoning_effort": "medium"
                    }

                # 3. Call the API using the unpacked payload
                if self.output_format:
                    payload["response_format"] = self.output_format
                    response = await client.beta.chat.completions.parse(**payload)
                    msg = response.choices[0].message

                    if self.output_format == CoTResponse:
                        parsed_text = msg.parsed
                    else:
                        parsed_text = msg.parsed.Output
                else:
                    # Raw unstructured output for models that don't support parse
                    response = await client.chat.completions.create(**payload)
                    msg = response.choices[0].message
                    parsed_text = msg.content

                # Extract reasoning/thinking text if available
                thinking_text = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None)

                # Update running stats
                self._extract_telemetry(response)

                # Update Dashboard
                if progress_bar:
                    reqs = self.stats['requests']

                    # RPM Throttle Check
                    if reqs % self.max_rpm == self.max_rpm - 1:
                        current_runtime = time.time() - self.start_time
                        time.sleep(max(0, 60 - (current_runtime % 60)))

                    # Calculate Means
                    mean_in = self.stats['input_tokens'] / reqs
                    mean_out = self.stats['output_tokens'] / reqs
                    mean_think = self.stats['thinking_tokens'] / reqs
                    mean_cost = self.stats['total_cost'] / reqs
                    est_total = mean_cost * progress_bar.total

                    progress_bar.set_postfix({
                        "Model": self.model,
                        "Model type": "Thinking" if self.stats['thinking_tokens'] > 0 else "Instruct",
                        "Input tokens/Req": f"{mean_in:.0f}",
                        "Thinking tokens/Req": f"{mean_think:.0f}",
                        "Output tokens/Req": f"{mean_out:.0f}",
                        "$/Req": f"${mean_cost:.5f}",
                        "Total cost": f"${self.stats['total_cost']:.4f}",
                        "Est. cost": f"${est_total:.4f}"
                    })
                    progress_bar.update(1)

                if self.return_thinking:
                    return {"output": parsed_text, "thinking": thinking_text}
                return parsed_text

            except LengthFinishReasonError:
                if progress_bar: progress_bar.write("Error: Token limit reached."); progress_bar.update(1)
                return None
            except Exception as e:
                if progress_bar: progress_bar.write(f"API Error: {e}"); progress_bar.update(1)
                return None

    async def run_in_thread(self, query: str):
        return await self.get_response(query)

    def run(self, query: str):
        try:
            return asyncio.run(self.get_response(query))
        except RuntimeError:
            return asyncio.run(self.run_in_thread(query))

    async def async_run(self, queries: list, desc: str = "Processing") -> list:
        """
        Runs multiple queries asynchronously.
        Args:
            queries: List of prompt strings.
            desc: The prefix title for the tqdm progress bar.
        """
        # Reset stats for new batch
        self.stats = {k: 0 for k in self.stats}
        self.stats['total_cost'] = 0.0
        self.start_time = time.time()

        with tqdm_lib.tqdm(total=len(queries), desc=desc) as progress_bar:
            tasks = [self.get_response(q, progress_bar) for q in queries]
            results = await asyncio.gather(*tasks)

        return list(results)
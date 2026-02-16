"""
Asynchronous GPT API utilities

This module provides tools for interacting with GPT models, including the `gpt_api`
class for managing requests with concurrency control, predefined `OUTPUT_FORMATS`,
and custom output classes for structured responses.

Supports both OpenAI and OpenRouter APIs.
"""

import os
import time
import asyncio
from typing import List

import openai
import tqdm.auto as tqdm_lib
from pydantic import BaseModel
from dotenv.main import load_dotenv
from openai import AsyncOpenAI, LengthFinishReasonError

# Load API keys from .env file:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set input/output tokens usage cost for OpenAI models:
INPUT_TOKENS_COST = {
    "gpt-4o": 0.0000025,
    "gpt-4o-mini": 0.00000015,
    "o1": 0.000015,
    "o3-mini": 0.0000011
}
OUTPUT_TOKENS_COST = {
    "gpt-4o": 0.00001,
    "gpt-4o-mini": 0.0000006,
    "o1": 0.00006,
    "o3-mini": 0.0000044
}

# Set input/output tokens usage cost for OpenRouter models (examples - update as needed):
OPENROUTER_INPUT_TOKENS_COST = {
    "google/gemini-2.5-flash": 0.0000003,
}
OPENROUTER_OUTPUT_TOKENS_COST = {
    "google/gemini-2.5-flash": 0.0000025,
}


# Structured string response for LLM outputs:
class StringResponse(BaseModel):
    Output: str


# Structured list of strings response for LLM outputs:
class ListResponse(BaseModel):
    Output: List[str]


# Structured boolean response for LLM outputs:
class BoolResponse(BaseModel):
    Output: bool


# Dictionary mapping output format strings to OpenAI API format (define as needed)
OUTPUT_FORMATS = {
    'string': StringResponse,
    'list': ListResponse,
    'bool': BoolResponse
}


class gptAPI:
    """
    Asynchronous GPT API client with concurrency control and progress tracking.

    This class allows making both synchronous and asynchronous requests to the
    GPT model while managing parallel API calls using an asyncio semaphore.

    Supports both OpenAI and OpenRouter APIs.

    Args:
        system_prompt (str): The system message to guide GPT responses.
        model (str, optional): The GPT model to use (default: "gpt-4o-mini").
        output_format (str, optional): The response format (default: "string").
        temperature (float, optional): The model's randomness (default: 0.2).
        max_concurrent_requests (int, optional): Max parallel API calls (default: 10).
        max_rpm (int, optional): Max requests per minute to the GPT API based on usage tier (default: 500).
        api_provider (str, optional): API provider to use - "openai" or "openrouter" (default: "openai").

    Methods:
        get_response(user_prompt): Asynchronously fetches a response from GPT.
        run(query): Runs a single query synchronously using asyncio.
        async_run(queries): Runs multiple queries asynchronously with progress tracking.
    """

    def __init__(self,
                 system_prompt: str,
                 model: str = "gpt-4o-mini",
                 output_format: str = 'string',
                 temperature: float = 0.2,
                 max_concurrent_requests: int = 10,
                 max_rpm: int = 500,
                 max_tokens: int = 4096,
                 api_provider: str = "openai"):

        # Define system prompt:
        self.system_prompt = system_prompt

        # Define API provider:
        self.api_provider = api_provider.lower()
        if self.api_provider not in ["openai", "openrouter"]:
            raise ValueError("api_provider must be either 'openai' or 'openrouter'")

        # Define OpenAI client parameters:
        self.model = model
        self.output_format = OUTPUT_FORMATS[output_format]
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Define a semaphore to limit the number of parallel API calls
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_rpm = max_rpm
        self.start_time = 0

        # Track total token usage
        self.total_usage = 0

        # Set up API credentials based on provider
        if self.api_provider == "openrouter":
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            self.base_url = "https://openrouter.ai/api/v1"
            self.input_cost_table = OPENROUTER_INPUT_TOKENS_COST
            self.output_cost_table = OPENROUTER_OUTPUT_TOKENS_COST
        else:  # openai
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.base_url = None  # Use default OpenAI base URL
            self.input_cost_table = INPUT_TOKENS_COST
            self.output_cost_table = OUTPUT_TOKENS_COST

    def _get_client(self):
        """Creates an AsyncOpenAI client configured for the selected API provider."""
        if self.api_provider == "openrouter":
            return AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            return AsyncOpenAI(api_key=self.api_key)

    async def get_response(self, user_prompt: str, progress_bar=None):
        """
        Fetches a GPT response asynchronously, limiting concurrency.

        Args:
            user_prompt (str): The input query for the GPT model.
            progress_bar (tqdm, optional): Progress bar instance for tracking.

        Returns:
            str: The model's parsed output.
        """
        async with self.semaphore:  # Limits concurrency
            client = self._get_client()

            try:
                response = await client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": self.system_prompt},
                        {"role": "user",
                         "content": user_prompt}
                    ],
                    response_format=self.output_format,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,  # Hard limit
                )

                text = response.choices[0].message.parsed.Output

                # Extract token usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Calculate cost using the appropriate cost table
                input_cost = self.input_cost_table.get(self.model, 0) * input_tokens
                output_cost = self.output_cost_table.get(self.model, 0) * output_tokens
                usage_cost = input_cost + output_cost

                # Update total token usage
                self.total_usage += usage_cost

                # Update progress bar with current and estimated total usage
                if progress_bar:
                    current_step = progress_bar.n + 1
                    if current_step % self.max_rpm == self.max_rpm - 1:
                        current_runtime = time.time() - self.start_time
                        time.sleep(60 - (current_runtime % 60))
                    estimated_total = (self.total_usage / current_step) * progress_bar.total
                    progress_bar.set_description(
                        f"Usage cost: {self.total_usage:.4f}$ (Estimated: {estimated_total:.2f}$)")
                    progress_bar.update(1)

                return text

            except LengthFinishReasonError:
                # The token limit was reached, preventing valid JSON parsing
                if progress_bar:
                    progress_bar.write(f"Error: Token limit reached for a request. Returning None.")
                    progress_bar.update(1)  # Ensure bar continues moving
                return None

            except Exception as e:
                # Catch other API errors to prevent crash
                if progress_bar:
                    progress_bar.write(f"Unexpected Error: {e}")
                    progress_bar.update(1)
                return None

    async def run_in_thread(self, query: str):
        """Helper function to call `run()` in environments where `asyncio.run()` fails."""
        return await self.get_response(query)

    def run(self, query: str):
        """Runs a single query synchronously using asyncio.run()."""
        try:
            return asyncio.run(self.get_response(query))
        except RuntimeError:
            # Handles cases where an event loop is already running (e.g., Jupyter)
            return asyncio.run(self.run_in_thread(query))

    async def async_run(self, queries: list) -> list:
        """Runs multiple queries asynchronously and tracks progress."""
        self.total_usage = 0  # Reset usage before running new batch
        self.start_time = time.time()
        with tqdm_lib.tqdm(total=len(queries),
                           desc=f"Usage cost: {self.total_usage:.4f}$ (Estimated: {0.00:.2f}$)") as progress_bar:
            tasks = [self.get_response(q, progress_bar) for q in queries]
            results = await asyncio.gather(*tasks)

        return list(results)
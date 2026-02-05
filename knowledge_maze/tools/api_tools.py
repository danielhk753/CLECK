"""
Asynchronous GPT API utilities

This module provides tools for interacting with GPT models, including the `gpt_api`
class for managing requests with concurrency control, predefined `OUTPUT_FORMATS`,
and custom output classes for structured responses.
"""

import os
import time
import asyncio
from typing import List, Union, Literal, Optional

import openai
import tqdm.auto as tqdm_lib
from dotenv.main import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, LengthFinishReasonError

# Load OpenAI API key from .env file:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set input/output tokens usage cost:
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

# Structured string response for LLM outputs:
class StringResponse(BaseModel):
    Output: str


# Structured list of strings response for LLM outputs:
class ListResponse(BaseModel):
    Output: List[str]


# Structured boolean response for LLM outputs:
class BoolResponse(BaseModel):
    Output: bool


# Structured custom table response for LLM outputs:
class TableGroup(BaseModel):
    group_description: str
    entity_column: str = Field(description="The column containing the names of the entities (e.g., 'Company Name')")
    filter_column: str = Field(description="The column used to rank or filter (e.g., 'Year founded')")
    filter_type: Literal["list", "numerical_range", "temporal_range", "top_n"]
    filter_value: Union[List[str], List[float], int] = Field(description="For top_n, this is the count (e.g., 7)")
    sort_order: Optional[Literal["asc", "desc"]] = Field(default="desc", description="desc for 'latest' or 'highest'")

class TableGroupResponse(BaseModel):
    Output: TableGroup


# Dictionary mapping output format strings to OpenAI API format (define as needed)
OUTPUT_FORMATS = {
    'string': StringResponse,
    'list': ListResponse,
    'bool': BoolResponse,
    'dict': TableGroupResponse
}


class gptAPI:
    """
    Asynchronous GPT API client with concurrency control and progress tracking.

    This class allows making both synchronous and asynchronous requests to the
    GPT model while managing parallel API calls using an asyncio semaphore.

    Args:
        system_prompt (str): The system message to guide GPT responses.
        model (str, optional): The GPT model to use (default: "gpt-4o-mini").
        output_format (str, optional): The response format (default: "string").
        temperature (float, optional): The model's randomness (default: 0.2).
        max_concurrent_requests (int, optional): Max parallel API calls (default: 10).
        max_rpm (int, optional): Max requests per minute to the GPT API based on usage tier (default: 500).

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
                 max_tokens: int = 4096):

        # Define system prompt:
        self.system_prompt = system_prompt

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
            client = AsyncOpenAI()  # Use async OpenAI client

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

                # Calculate cost
                usage_cost = INPUT_TOKENS_COST[self.model] * input_tokens + OUTPUT_TOKENS_COST[self.model] * output_tokens

                # Update total token usage
                self.total_usage += usage_cost

                # Update progress bar with current and estimated total usage
                if progress_bar:
                    current_step = progress_bar.n + 1
                    if current_step % self.max_rpm == self.max_rpm - 1:
                        current_runtime = time.time() - self.start_time
                        time.sleep(60 - (current_runtime % 60))
                    estimated_total = (self.total_usage / current_step) * progress_bar.total
                    progress_bar.set_description(f"Usage cost: {self.total_usage:.4f}$ (Estimated: {estimated_total:.2f}$)")
                    progress_bar.update(1)

                return text

            except LengthFinishReasonError as e:
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

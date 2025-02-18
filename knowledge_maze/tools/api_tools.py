"""
Asynchronous GPT API utilities

This module provides tools for interacting with GPT models, including the `gpt_api`
class for managing requests with concurrency control, predefined `OUTPUT_FORMATS`,
and custom output classes for structured responses.
"""

import os
import asyncio
from typing import List

import openai
from tqdm.asyncio import tqdm
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load OpenAI API key from .env file:
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


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


class gpt_api:
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
                 max_concurrent_requests: int = 10):

        # Define system prompt:
        self.system_prompt = system_prompt

        # Define OpenAI client parameters:
        self.model = model
        self.output_format = OUTPUT_FORMATS[output_format]
        self.temperature = temperature

        # Define a semaphore to limit the number of parallel API calls
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

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

            response = await client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": self.system_prompt},
                    {"role": "user",
                     "content": user_prompt}
                ],
                response_format=self.output_format,
                temperature=self.temperature
            )

            if progress_bar:
                progress_bar.update(1)  # Update tqdm progress bar

            return response.choices[0].message.parsed.Output

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
        with tqdm(total=len(queries)) as progress_bar:
            tasks = [self.get_response(q, progress_bar) for q in queries]
            results = await asyncio.gather(*tasks)

        return list(results)

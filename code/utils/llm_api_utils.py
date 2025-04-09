import os
import time
from typing import Dict, List
import random
# Groq imports
from groq import Groq

# Claude imports
import anthropic

# OpenAI imports
from openai import OpenAI

# API keys
# Replace with your actual API key
GROQ_API_KEY = ''
ANTHROPIC_API_KEY = ''
OPENAI_API_KEY = ''  # Replace with your actual OpenAI API key
DEEP_SEEK_API_KEY = ''

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["DEEP_SEEK_API_KEY"] = DEEP_SEEK_API_KEY

# Initialize clients
groq_client = Groq()
claude_client = anthropic.Anthropic()
openai_client = OpenAI(api_key=OPENAI_API_KEY)  # Ensure the OpenAI client is initialized with the correct API key
deepseek_client = OpenAI(api_key=DEEP_SEEK_API_KEY, base_url='https://api.deepseek.com/v1')  # Ensure the OpenAI client is initialized with the correct API key

# Model mappings
GROQ_MODELS = ['deepseek-r1-distill-llama-70b','deepseek-r1-distill-qwen-32b', "llama3-70b-8192","llama3-8b-8192", "mixtral-8x7b-32768",'gemma2-9b-it', 'llama-3.2-1b-preview', 'llama-3.2-3b-preview', 'llama-3.2-11b-text-preview','llama-3.2-90b-text-preview','llama-3.3-70b-versatile','llama-3.2-3b-preview']
CLAUDE_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229",'claude-3-5-sonnet-20241022','claude-3-5-sonnet-20240620']
GPT_MODELS = ["gpt-4o"]
DEEP_SEEK_MODELS = ["deepseek-chat"]

def is_groq_model(model: str) -> bool:
    return model in GROQ_MODELS

def is_claude_model(model: str) -> bool:
    return model in CLAUDE_MODELS

def is_gpt_model(model: str) -> bool:
    return model in GPT_MODELS

def is_deepseek_model(model: str) -> bool:
    return model in DEEP_SEEK_MODELS

def chat_completion(
        messages: List[Dict],
        model: str,
        temperature: float = 0,
        max_completion_tokens=10000,
        top_p: float = 1,
        max_retries: int = 3,
        initial_timeout: float = 5
) -> str:
    timeout = initial_timeout

    for attempt in range(max_retries):
        try:
            if is_groq_model(model):
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    top_p=top_p,
                    timeout=timeout,
                )
                return response.choices[0].message.content
            elif is_claude_model(model):
                response = claude_client.messages.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=1000,
                )
                return response.content[0].text

            elif is_deepseek_model(model):
                response = deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=1000,
                )
                return response.choices[0].message.content

            elif is_gpt_model(model):
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            error_message = str(e).lower()
            if "rate limit" in error_message:
                print(f"Rate limit reached for {model}. Turning off the client.")
                time.sleep(60)
            elif "timeout" in error_message:
                print(f"Timeout occurred for {model}. Attempt {attempt + 1}/{max_retries}. Increasing timeout...")
                timeout *= 1.5
            else:
                print(f"Error for {model}: {e}. Attempt {attempt + 1}/{max_retries}. Retrying...")
                time.sleep(random.uniform(2, 5))

    return f"Max retries reached. Unable to get a response from {model}."

def completion(
        prompt: str,
        model: str,
        temperature: float = 0,
        top_p: float = 1,
        max_completion_tokens=10000,
        max_retries: int = 5,
        initial_timeout: float = 30
) -> str:
    return chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        initial_timeout=initial_timeout
    )
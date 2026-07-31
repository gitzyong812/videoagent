from __future__ import annotations

import base64
import copy
import functools
import json
import random
import time
from mimetypes import guess_type

import cv2
import requests
from azure.identity import AzureCliCredential

import sgd.config as config

_MODEL_NAME_ALIASES = {
    "gpt-4.1mini": "gpt-4.1-mini",
    "gpt-4.1nano": "gpt-4.1-nano",
    "o3mini": "o3-mini",
    "o4mini": "o4-mini",
}
_WARNED_MODEL_ALIASES = set()


def _normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str):
        return model_name
    resolved = _MODEL_NAME_ALIASES.get(model_name.strip(), model_name.strip())
    if resolved != model_name:
        alias_key = (model_name, resolved)
        if alias_key not in _WARNED_MODEL_ALIASES:
            print(f"[MODEL] Normalized model name from {model_name!r} to {resolved!r}.")
            _WARNED_MODEL_ALIASES.add(alias_key)
    return resolved


def _select_azure_endpoint(endpoints, service_name: str) -> str:
    if isinstance(endpoints, str):
        candidates = [endpoints.strip()]
    elif isinstance(endpoints, (list, tuple)):
        candidates = [str(item).strip() for item in endpoints]
    else:
        raise ValueError(f"{service_name} endpoints must be a string or a list of strings.")
    candidates = [item.rstrip("/") for item in candidates if item]
    if not candidates:
        raise ValueError(
            f"No Azure endpoint configured for {service_name}. "
            "Set OPENAI_API_KEY or the corresponding SGD_*_ENDPOINTS variable."
        )
    return random.choice(candidates)


def _uses_reasoning_token_parameter(model_name: str) -> bool:
    normalized = model_name.lower()
    return len(normalized) >= 2 and normalized[0] == "o" and normalized[1].isdigit()


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def _is_retryable_error(err_msg: str) -> bool:
        low = err_msg.lower()
        retry_keywords = [
            "rate limit",
            "too many requests",
            "timed out",
            "timeout",
            "internal",
            "service unavailable",
            "connection aborted",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "bad gateway",
            "gateway timeout",
            "read timed out",
            "max retries exceeded",
            "forbidden for url",
            "openai api returned status 429",
            "openai api returned status 500",
            "openai api returned status 502",
            "openai api returned status 503",
            "openai api returned status 504",
        ]
        return any(k in low for k in retry_keywords)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if _is_retryable_error(str(e)):
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    raise

    return wrapper

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

@retry_with_exponential_backoff
def call_openai_model_with_tools(  
    messages,
    endpoints,
    model_name,
    api_key: str = None,
    tools: list | None = None,  # List of tool definitions
    image_paths: list | None = None,
    max_tokens: int = 4096,  
    temperature: float = 0.0,  
    tool_choice: str = "auto",  # Can be "auto", "none", or a specific tool
    return_json: bool = False,
    trace_logger=None,
    trace_context: dict = None,
) -> dict:  
    if tools is None:
        tools = []
    if image_paths is None:
        image_paths = []
    model = _normalize_model_name(model_name)
    if api_key:
        headers = {  
            "Content-Type": "application/json",  
            'Authorization': 'Bearer ' + api_key
        }
        endpoint = config.OPENAI_BASE_URL.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
        url = f"{endpoint}/chat/completions"
    else:
        endpoint = _select_azure_endpoint(endpoints, "chat completions")
        credential = AzureCliCredential()  
        token = credential.get_token('https://cognitiveservices.azure.com/')  
        headers = {  
            "Content-Type": "application/json",  
            'Authorization': 'Bearer ' + token.token  
        }  
        url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2025-03-01-preview"
      
    payload = {  
        "model": model,
        "messages": copy.deepcopy(messages),
        "max_completion_tokens": max_tokens,
    }
    if not _uses_reasoning_token_parameter(model):
        payload["temperature"] = temperature
    if return_json:
        payload["response_format"] = {"type": "json_object"}
  
    # Add tools to the payload if provided
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
  
    if image_paths:
        # with mp.Pool(processes=min(len(image_paths), mp.cpu_count())) as pool:
        #     image_data_list = pool.map(local_image_to_data_url, image_paths)
        image_data_list = [local_image_to_data_url(image_path) for image_path in image_paths]
        payload['messages'].append({"role": "user", "content": []})
        for image_data in image_data_list:
            payload['messages'][-1]['content'].append({"type": "image_url", "image_url": {"url": image_data}})
      
    start_time = time.perf_counter()
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.API_TIMEOUT_SECS,
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
  
    if response.status_code != 200:
        error_text = response.text
        raise Exception(f"OpenAI API returned status {response.status_code}: {error_text}")  
      
    response_data = response.json()  
    usage = response_data.get("usage", {})
    
    # Get the message from the response
    message = response_data['choices'][0]['message']

    if trace_logger is not None:
        step_idx = None
        if trace_context is not None:
            step_idx = trace_context.get("step_idx")
        trace_logger.log_llm_call(
            step_idx=step_idx,
            llm_input=payload,
            llm_output=message,
            model=response_data.get("model", model_name),
            latency_ms=latency_ms,
            tokens=usage,
            endpoint=url,
        )
    
    # Normalize tool-calling and direct-answer responses into a single shape.
    if "tool_calls" in message and message.get("tool_calls") is not None:
        return message

    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()

    return {
        "role": message.get("role", "assistant"),
        "content": content,
        "tool_calls": [],
    }

class AzureOpenAIEmbeddingService:  
    @staticmethod  
    @retry_with_exponential_backoff
    def get_embeddings(endpoints, model_name, input_text, api_key: str = None):  
        """  
        Call Azure OpenAI Embedding service and get embeddings for the input text.  
  
        :param api_key: Your Azure OpenAI API key.  
        :param endpoint: The endpoint URL for the OpenAI service.  
        :param model: The model name for generating embeddings.  
        :param input_text: The text for which you want to generate embeddings.  
        :return: The embeddings as a JSON response.  
        """  
        model = _normalize_model_name(model_name)
        if api_key:
            headers = {  
                "Content-Type": "application/json",  
                'Authorization': 'Bearer ' + api_key
            }
            endpoint = config.OPENAI_BASE_URL.rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint = f"{endpoint}/v1"
            url = f"{endpoint}/embeddings"
        else:
            endpoint = _select_azure_endpoint(endpoints, "embeddings")
            # Define the URL for the embeddings endpoint  
            url = f"{endpoint}/openai/deployments/{model}/embeddings?api-version=2023-05-15"
    
            credential = AzureCliCredential()  
            token = credential.get_token('https://cognitiveservices.azure.com/')  
            headers = {  
                "Content-Type": "application/json",  
                'Authorization': 'Bearer ' + token.token  
            }

        # Set up the payload for the request  
        payload = {  
            "input": input_text,
            "model": model
        }  
  
        # Make the request to the Azure OpenAI service  
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=config.API_TIMEOUT_SECS,
        )
  
        # Check if the request was successful  
        if response.status_code == 200:  
            return response.json()['data']
        else:  
            response.raise_for_status()

def extract_answer(message: dict) -> str | None:
    """
    Extract the plain-text answer from an assistant message that may include
    tool calls.

    The function first checks the normal `content` field (for responses that
    are not using tools). If the assistant responded via a tool call, it
    attempts to parse the JSON string stored in
    `message["tool_calls"][i]["function"]["arguments"]` and returns the value
    associated with the key `"answer"`.

    Parameters
    ----------
    message : dict
        The assistant message returned by `call_openai_model_with_tools`.

    Returns
    -------
    str | None
        The extracted answer, or ``None`` if no answer could be found.
    """
    # Tool-based response
    for call in message.get("tool_calls", []):
        args_json = call["function"]["arguments"]
        args = json.loads(args_json)
        if (answer := args.get("answer")):
            return answer

    # Direct text response
    if (content := message.get("content")):
        return content.strip()
    
    return None

import asyncio
import json
import re
from enum import Enum, auto
from typing import List, Union, AsyncGenerator
from openai import OpenAI
from configs import configs
from pydantic import BaseModel


class StreamMode(Enum):
    TOKEN = auto()
    BUFFER = auto()
    SENTENCE = auto()
    COMPLETE = auto()


SENTENCE_SPLIT_REGEX = r"(?<=[.!?])\s+"


class LLMMessage(BaseModel):
    role: str
    content: str


client = OpenAI()


async def call(prompt: str, response_type="json_object") -> dict:
    """
    Execute a system call to the LLM
    """
    messages = [{"role": "system", "content": prompt}]

    response = await asyncio.to_thread(
        client.chat.completions.create,
        messages=messages,
        model=configs.openai_model,
        temperature=configs.openai_temperature,
        timeout=configs.openai_timeout,
        response_format={"type": response_type},
    )

    result = response.choices[0].message.content
    if response_type == "text":
        return result
    return json.loads(result)


def sync_call(
    prompt: str,
    response_type="json_object",
    model: str = None,
    llm_settings: dict = None,
) -> dict:
    """
    Execute a system call to the LLM
    """
    messages = [{"role": "system", "content": prompt}]

    print(prompt)
    response = client.chat.completions.create(
        messages=messages,
        model=model if model else configs.openai_model,
        temperature=configs.openai_temperature,
        timeout=configs.openai_timeout,
        tools=llm_settings.get("tools", None) if llm_settings else None,
        response_format={"type": response_type},
    )

    print(f"HELLO {response}")

    if (
        response.choices[0].finish_reason == "tool_calls"
        and response.choices[0].message.tool_calls
        and response.choices[0].message.tool_calls[0].function
    ):
        if (
            function_arguments := response.choices[0]
            .message.tool_calls[0]
            .function.arguments
        ):
            function_arguments = json.loads(function_arguments)

        return {
            "function_call": {
                "name": response.choices[0].message.tool_calls[0].function.name,
                "arguments": function_arguments,
            }
        }

    result = response.choices[0].message.content
    if response_type == "text":
        return result
    return json.loads(result)


async def stream(
    chat_history: List[LLMMessage], llm_settings: dict = None
) -> AsyncGenerator[Union[str, dict], None]:
    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in chat_history
    ]

    response = await asyncio.to_thread(
        client.chat.completions.create,
        messages=messages,
        model=configs.openai_model,
        stream=True,
        temperature=configs.openai_temperature,
        timeout=configs.openai_timeout,
        tools=llm_settings.get("tools", None) if llm_settings else None,
    )

    tool_calls = []
    for chunk in response:
        delta = chunk.choices[0].delta

        # handling text generation
        if delta and delta.content:
            yield delta.content

        # handle tool calls
        elif delta and delta.tool_calls:
            tool_call_chunk_list = delta.tool_calls
            for tcchunk in tool_call_chunk_list:
                if len(tool_calls) <= tcchunk.index:
                    tool_calls.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                current_tool_call = tool_calls[tcchunk.index]

                if tcchunk.id:
                    current_tool_call["id"] += tcchunk.id
                if tcchunk.function.name:
                    current_tool_call["function"]["name"] += tcchunk.function.name
                if tcchunk.function.arguments:
                    current_tool_call["function"][
                        "arguments"
                    ] += tcchunk.function.arguments

        if (
            len(tool_calls)
            and "function" in tool_calls[0]
            and chunk.choices[0].finish_reason == "tool_calls"
        ):
            function_call = tool_calls[0]
            function_name = function_call["function"].get("name")
            function_arguments = function_call["function"].get("arguments")

            if function_arguments:
                function_arguments = json.loads(function_arguments)

            yield {
                "function_call": {
                    "name": function_name,
                    "arguments": function_arguments,
                }
            }


async def stream_by_mode(
    chat_history: List[LLMMessage],
    llm_settings: dict = None,
    stream_mode: StreamMode = StreamMode.SENTENCE,
) -> AsyncGenerator[Union[str, dict], None]:
    buffer = ""
    content = ""

    async for chunk in stream(chat_history, llm_settings):
        if "function_call" in chunk:
            yield chunk
        else:
            buffer += chunk
            content += chunk

            if stream_mode == StreamMode.SENTENCE:
                sentences = re.split(SENTENCE_SPLIT_REGEX, buffer)
                for sentence in sentences[:-1]:
                    yield {"sentence": sentence.strip()}
                buffer = sentences[-1]
            if stream_mode == StreamMode.BUFFER:
                yield {"buffer": content}

    if stream_mode == StreamMode.SENTENCE and buffer:
        yield {"sentence": buffer.strip()}

    yield {"complete": content.strip()}


def is_function_call(line: dict) -> bool:
    """
    Check if a response line is a function call.

    Args:
        line (dict): The response line to check.

    Returns:
        bool: True if it's a function call, False otherwise.
    """
    return "function_call" in line

import os
import re
import traceback

from openai import OpenAI

from memory import load_memory


LAST_ERROR = ""


def _model_name():
    return os.getenv("MODEL_NAME")


def _api_key():
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )


def _build_client():
    api_key = _api_key()
    if not api_key:
        return None

    base_url = os.getenv("API_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs["timeout"] = 20.0
    return OpenAI(**kwargs)


def extract_action(text):
    match = re.search(r'report\(".*?"\)|noop\(\)', text)
    return match.group(0) if match else "noop()"


def _message_text(message_content):
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(part for part in text_parts if part)
    return ""


def clear_last_error():
    global LAST_ERROR
    LAST_ERROR = ""


def get_last_error():
    return LAST_ERROR


def build_prompt(observation, memory):
    return f"""
You are a careful code review agent being trained in a benchmark environment.

Task title: {observation.task_title}
Task domain: {observation.task_domain}
Task objective: {observation.task_objective}
Current step: {observation.step_count}
Current score: {observation.score}

Issues already found:
{observation.found}

Issues still remaining:
{observation.remaining}

Previous attempts:
{observation.history}

High-reward memory:
{memory['good']}

Low-reward memory:
{memory['bad']}

Instructions:
- Find the single highest-impact remaining issue.
- Use concise issue phrases like "sql injection" or "path traversal".
- Do not explain your reasoning.
- Return exactly one action in one of these formats:
report("issue")
noop()

Code under review:
{observation.code}
"""


def act(observation):
    global LAST_ERROR

    model_name = _model_name()
    if not model_name:
        LAST_ERROR = "MODEL_NAME is not set."
        return "noop()"

    client = _build_client()
    if client is None:
        LAST_ERROR = "No API key found in HF_TOKEN, OPENAI_API_KEY, or API_KEY."
        return "noop()"

    memory = load_memory()
    LAST_ERROR = ""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software security and reliability reviewer.",
                },
                {"role": "user", "content": build_prompt(observation, memory)},
            ],
            temperature=0.2,
        )
        content = _message_text(response.choices[0].message.content)
        return extract_action(content)
    except Exception as exc:
        LAST_ERROR = str(exc)
        print(f"inference_error: {exc}", flush=True)
        traceback.print_exc()
        return "noop()"

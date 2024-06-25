import os
import re
from typing import Any, ClassVar

import httpx
import yaml
from rich import print

MD_START = re.compile(r"^```[^\n]*\n")
MD_END = re.compile(r"\n```$")


class ClaudeClient:
    _instance: ClassVar["ClaudeClient | None"] = None

    def __init__(
        self, api_key: str, base_url: str | httpx.URL = "https://api.anthropic.com/v1/"
    ):
        self.api_key = api_key
        self.base_url = httpx.URL(base_url)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "X-API-Key": f"{self.api_key}",
                "anthropic-version": "2023-06-01",
            },
            timeout=500,
        )

    @classmethod
    def get_instance(cls) -> "ClaudeClient":
        if cls._instance is None:
            cls._instance = cls(os.getenv("ANTHROPIC_API_KEY"))

        return cls._instance


async def ask_claude(system: str, data: str) -> str:
    cl = ClaudeClient.get_instance()
    resp = await cl.client.post(
        "messages",
        json=dict(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            system=system,
            messages=[
                {"role": "user", "content": data},
            ],
            max_tokens=4096,
        ),
    )

    if resp.is_error:
        print(resp.json())

    resp.raise_for_status()

    return resp.json()["content"][-1]["text"]


def parse_chat_output(parsed_raw: str) -> Any:
    parsed_raw = MD_START.sub("", parsed_raw)
    parsed_raw = MD_END.sub("", parsed_raw)

    try:
        parsed = yaml.safe_load(parsed_raw)
    except yaml.YAMLError:
        print(parsed_raw)
    else:
        return parsed


__all__ = ["parse_chat_output"]

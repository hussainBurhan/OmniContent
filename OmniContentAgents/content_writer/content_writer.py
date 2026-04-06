import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from OmniContentAgents.seo_specialist.schemas import UserQuery

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_ID = "gemini-2.5-flash"

gemini_async_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

gemini_chat_model = OpenAIChatCompletionsModel(
    model=GEMINI_MODEL_ID,
    openai_client=gemini_async_client,
)

INSTRUCTIONS = """
You are a creative content writer. You receive a user query plus SEO/search research (raw text or summaries).
Write a content piece that addresses the query and reflects the research. Return the full draft as plain text.
"""

content_writer = Agent(
    name="content_writer",
    handoff_description="A creative content writer who writes a content piece that addresses the user query and reflects the SEO/search research.",
    instructions=INSTRUCTIONS,
    tools=[],
    model=gemini_chat_model,
)


async def run_content_writer(user_query: UserQuery, seo_research: str) -> str:
    body = (
        f"User query:\n{user_query.query}\n\n"
        f"SEO / search research:\n{seo_research}"
    )
    runner = await Runner.run(content_writer, body)
    out = runner.final_output
    return out if isinstance(out, str) else str(out)


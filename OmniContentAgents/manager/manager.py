from agents import Agent, Runner
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os
from OmniContentAgents.seo_specialist.seo_specialist import seo_specialist
from OmniContentAgents.content_writer.content_writer import content_writer
from OmniContentAgents.seo_specialist.schemas import UserQuery
import asyncio

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
You are a manager. You are given a user query.
You need to decide which agent to use to handle the user query.
You need to return the agent to use.
"""

manager = Agent(
    name="manager",
    handoff_description="A manager who decides which agent to use to handle the user query.",
    instructions=INSTRUCTIONS,
    tools=[],
    model=gemini_chat_model,
    handoffs=[seo_specialist, content_writer]
)

async def run_manager(user_query: UserQuery):
    runner = await Runner.run(manager, user_query.query)
    return runner.final_output

async def main():
    user_query = UserQuery(
        query="Help me plan a content piece for my new SaaS tool that automates invoice processing. Target audience is small business owners."
    )
    print(await run_manager(user_query))

if __name__ == "__main__":
    asyncio.run(main())


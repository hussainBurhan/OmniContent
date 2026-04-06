from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from OmniContentAgents.seo_specialist.tools import validate_query, search_the_web
from OmniContentAgents.seo_specialist.schemas import UserQuery
import asyncio
from dotenv import load_dotenv
import os

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
You are a SEO specialist. You are given a user query. 
You need to validate the query and return the keywords that are most relevant to the query using the validate_query tool.
Use the keywords to find the most relevant pages on the internet, using the search_the_web tool.
You need to return the results of the search using the search_the_web tool.
"""

seo_specialist = Agent(
    name="seo_specialist",
    handoff_description="A SEO specialist who validates the user query and returns the keywords that are most relevant to the query.",
    instructions=INSTRUCTIONS,
    tools=[validate_query, search_the_web],
    model=gemini_chat_model,
)


async def run_seo_specialist(user_query: UserQuery):
    runner = await Runner.run(seo_specialist, user_query.query)
    return runner.final_output


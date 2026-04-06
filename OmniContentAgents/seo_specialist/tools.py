import asyncio

from OmniContentAgents.seo_specialist.schemas import (
    UserQuery,
    QueryResult,
    SearchRequest,
    SearchResults,
    SearchResult,
)
from openai import OpenAI
from agents import function_tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

gemini_client = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
tavily_client = TavilyClient(api_key=tavily_api_key)

@function_tool
async def validate_query(user_query: UserQuery) -> QueryResult:
    """
    Validate the query and return the keywords that are most relevant to the query.
    """

    # Check if the query is valid
    if not user_query.query:
        return QueryResult(is_query_valid=False, query=user_query.query, keywords=[])
    else:
        prompt = f"""
        You are a SEO specialist.
        You are given a user query.
        You need to validate the query and return the keywords that are most relevant to the query.
        Return at most 3 keywords, ranked by relevance (best first).

        User Query: {user_query.query}
        """
        response = gemini_client.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format=QueryResult,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return QueryResult(
                is_query_valid=False,
                query=user_query.query,
                keywords=[],
            )
        return parsed

@function_tool
async def search_the_web(search_request: SearchRequest) -> SearchResults:
    """
    Search the web for the most relevant pages for the given keywords.
    """
    rows: list[SearchResult] = []
    for keyword in search_request.keywords:
        raw: dict = tavily_client.search(keyword, max_results=search_request.max_results)
        for item in raw.get("results") or []:
            rows.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                )
            )
    return SearchResults(results=rows)

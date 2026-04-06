from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    query: str
    "The query that the user entered"

class QueryResult(BaseModel):
    is_query_valid: bool
    "Whether the query is valid or not"
    query: str
    "The query that the user entered"
    keywords: list[str] = Field(max_length=3)
    "At most 3 keywords most relevant to the query"
    
class SearchRequest(BaseModel):
    keywords: list[str]
    "The keywords to search for"
    max_results: int = Field(default=2)
    "The maximum number of results to return"

class SearchResults(BaseModel):
    results: list[SearchResult]
    "The results of the search"

class SearchResult(BaseModel):
    url: str
    "The URL of the result"
    title: str
    "The title of the result"
    snippet: str
    "The snippet of the result"

from typing_extensions import TypedDict, Annotated

class State(TypedDict):
    """Represents the state of our query generation and execution."""
    question: str
    query: str
    result: str
    answer: str
    chart_code: str
    table_info: str
    tables_prompt: str
    table_descriptions: str

class QueryOutput(TypedDict):
    """Generated SQL query from the LLM."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]
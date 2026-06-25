"""Query rewriter node.

Takes the user's natural-language topic and asks the LLM to produce N
targeted search queries for the academic APIs. We use structured output so
the LLM returns a Pydantic object directly, no JSON parsing required.
"""

import logging

from pydantic import BaseModel, Field

from research_gap_agent.config import load_settings
from research_gap_agent.llm import get_llm
from research_gap_agent.prompts.query_rewriter import (
    QUERY_REWRITER_SYSTEM,
    QUERY_REWRITER_USER,
)
from research_gap_agent.schemas import SearchQuery
from research_gap_agent.state import GraphState


logger = logging.getLogger(__name__)


class QueryList(BaseModel):
    primary_query: str = Field(..., description="Faithful reformulation of the user's topic for reranking.")
    queries: list[SearchQuery] = Field(..., min_length=1)


def query_rewriter_node(state: GraphState) -> dict:
    """Generate a primary query and N exploration queries from state.initial_topic."""
    settings = load_settings()
    n = settings.yaml.pipeline.num_queries

    llm = get_llm("query_rewriter").with_structured_output(QueryList)

    messages = [
        ("system", QUERY_REWRITER_SYSTEM.format(n=n)),
        ("human", QUERY_REWRITER_USER.format(topic=state.initial_topic, n=n)),
    ]

    result = llm.invoke(messages)
    queries = result.queries[:n]

    logger.info(
        "query_rewriter produced primary_query=%r and %d exploration queries for topic=%r",
        result.primary_query,
        len(queries),
        state.initial_topic,
    )
    for q in queries:
        logger.info("  - %s  (%s)", q.text, q.rationale)

    return {"queries": queries, "primary_query": result.primary_query}

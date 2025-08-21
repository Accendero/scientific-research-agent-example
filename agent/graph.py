import os
import time
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
import boto3
from langchain_aws import ChatBedrockConverse
from metapub import PubMedFetcher

from .tools_and_schemas import SearchQueryList, Reflection
from .state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    WebResearchState,
    SearchResult,
)
from .configuration import Configuration
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_summarizer_instructions,
    reflection_instructions,
    answer_instructions,
)
from .utils import (
    get_research_topic
)

load_dotenv()

# Used for Bedrock calls
bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Bedrock to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Bedrock
    llm = ChatBedrockConverse(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_tokens=None,
        client=bedrock_client,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research_search", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research_search(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that searches PubMed for results on a topic.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update - the recommended Tavily search query.
    """
    
    configurable = Configuration.from_runnable_config(config)
    search_results = []
    fetcher = PubMedFetcher()
    pm_ids = fetcher.pmids_for_query(
        query=state["search_query"],
        retmax=configurable.search_depth
    )
    #Only neccesary if you do not have an NCBI key.
    time.sleep(1.5)
    for id in pm_ids:
        abstract = fetcher.article_by_pmid(id)
        #Only neccesary if you do not have an NCBI key.
        time.sleep(1.5)
        if abstract.abstract is None:
            continue
        #Each abstract might be missing key pieces.
        try:
            search_results.append(SearchResult(
                query=state["search_query"],
                id=id, 
                title=abstract.title,
                year=int(abstract.year),
                citation=abstract.citation,
                abstract=abstract.abstract
            ))
        except:
            continue
    return {
        "search_results": search_results
    }

def web_research_report(state: WebResearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that summarizes results from a web search.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    sources_gathered = []
    text_results = ""
    searches = []
    for result in state["search_results"]:
        searches.append(result["query"])
        sources_gathered.append(result["id"])
        text_results += "Query:{}\nTitle:{}\nPMID:{}\nAbstract:{}\nCitation:{}".format(
            result["query"],
            result["title"], 
            result["id"],
            result["abstract"],
            result["citation"]
        )
        text_results += "\n---\n\n"

    # Configure
    formatted_prompt = web_summarizer_instructions.format(
        current_date=get_current_date(),
        results=text_results
    )

    llm = ChatBedrockConverse(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_tokens=None,
        client=bedrock_client,
    )
    result = llm.invoke(formatted_prompt)

    return {
        "sources_gathered": sources_gathered,
        "search_query": list(set(searches)),
        "web_research_result": [result.content],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries=state["web_research_result"],
    )
    # init Reasoning Model
    llm = ChatBedrockConverse(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_tokens=None,
        client=bedrock_client,
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research_search",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries=state["web_research_result"],
    )

    # init Reasoning Model
    llm = ChatBedrockConverse(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_tokens=None,
        client=bedrock_client,
    )
    result = llm.invoke(formatted_prompt)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": list(set(state["sources_gathered"])),
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research_search", web_research_search)
builder.add_node("web_research_report", web_research_report)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research_search"]
)
# Perform web research summary
builder.add_edge("web_research_search", "web_research_report")
# Reflect on the web research
builder.add_edge("web_research_report", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research_report", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")

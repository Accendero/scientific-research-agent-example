from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Only use keywords in the query

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What were the highest performing therapeutics approved in 2020 that target T Cells? 
```json
{{
    "rationale": "To answer this question accurately, we need to know which therapeutics that target T Cells were approved in 2020 by key regulators, like FDA and EMA. We also need to know performance of therapeutics approved in 2020, this could be fiscal or patient success.",
    "query": ["T Cell therapeutics revenue 2020", "T Cell Therapeutics FDA EMA approval", "T Cell therapeutics highest patient efficacy 2020"],
}}
```

Context: {research_topic}"""


web_summarizer_instructions = """Summarize targeted PubMed to provide the most recent, credible information and synthesize it into a verifiable text artifact.

Instructions:
- While older abstracts can still be accurate, abstracts closer to the current date can reflect more current information. The current date is {current_date}.
- Summarize key findings while meticulously tracking the source(s) for each specific piece of information.
- The research topic leading to each search result is part of each provided result as the "Query".
- Abstracts are provided, so only remove concepts or statements from summaries that are not relevant to the search.
- The output should be a well-written and concise based on your search findings, but, as above, should be comprehensive of the provided abstracts.
- Only include the information found in the search results, don't make up any information.

Output Format:
- Format the result as a text document of the summaries with --- between summary sections.

Example:

 Title:Article 1
 PMID:40818454
 Abstract summary:This is a summary of abstract 1.
 Citation:Citation 1.

---

 Title:Article 2
 PMID:40818454
 Abstract summary:This is a summary of abstract 2.
 Citation:Citation 2.

 ---

Reflect carefully on the Results to summarize each individually. Then, produce your output following the Example format:

Results:
{results}

"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format. THIS IS A MUST.
- Every fact stated in the write-up MUST have a corresponding citation, no exceptions.

Example of a statement of a fact:
This is a fact about the topic. [EFF](https://eff.org)

User Context:
- {research_topic}

Summaries:
{summaries}"""

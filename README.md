**Scientific Research Agent Example**

A demonstration of an AI-powered scientific research agent that automates literature review and report generation using Large Language Models (LLMs).

**Overview & Intended Purpose**

This repo is intended for exploration and prototyping. It is companion code for the "Let's Build" newsletter here:
https://www.linkedin.com/pulse/lets-build-scientific-research-agent-alan-barber-62x9c

As noted in the newsletter, significant parts of the code are based on the code found here:
https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart

The Gemini Fullstack LangGraph Quickstart is under the Apache 2.0 license found here:
https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart/blob/main/LICENSE

**Prerequisites**

Python 3.11 or later
Access to AWS Bedrock
Optional: Tavily API key

**Installation**

Clone the repository:

```bash
git clone https://github.com/Accendero/scientific-research-agent-example.git
cd scientific-research-agent-example
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Usage**

Basic Usage - See run_agent.py
```python
from agent.graph import graph

research_query = ""

message = {"messages": [("human", research_query)]}

for chunk in graph.stream(message, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

**License**

This project has no license as it is exploratory code meant for prototyping and educational purposes.

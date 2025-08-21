from agent.graph import graph

research_query = ""

message = {"messages": [("human", research_query)]}

for chunk in graph.stream(message, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
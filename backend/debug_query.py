import json
from rag import answer_query

query = "इस दस्तावेज में Second Party कौन है?"

print(f"Query: {query}")
for event in answer_query(query, filename_filter="CS.pdf"):
    print(f"EVENT: {event['type']}")
    if event['type'] == 'meta':
        print(f"  Table: {event['table']}")
        print(f"  Sources: {event['sources']}")
    elif event['type'] == 'content':
        print(f"  Content: {event['content']}")
    elif event['type'] == 'detail':
        print(f"  Detail length: {len(event['content'])}")
    elif event['type'] == 'sources_detail':
        print(f"  Sources detail count: {len(event['content'])}")
    else:
        print(f"  Data: {event}")

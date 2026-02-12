from semanticsearch import SemanticSearch
from chunker import chunk_text

engine = SemanticSearch()

with open("test_doc.txt", "r") as f:
    text = f.read()

chunks = chunk_text(text)

print("Chunks created:", len(chunks))

engine.add(chunks)

query = input("Ask something: ")

results = engine.search(query)

for text, score in results:
    print("\nMATCH:", text)
    print("Score:", score)

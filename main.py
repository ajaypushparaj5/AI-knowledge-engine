from semantic_search import SemanticSearch
from chunker import chunk_text
from pdf_loader import load_pdf


engine = SemanticSearch()

try:
    engine.load()
    print("Loaded memory.")
except:
    print("Building new memory...")
    pdf_text = load_pdf("sample.pdf")
    chunks = chunk_text(pdf_text)
    engine.add(chunks)
    engine.save()

query = input("Ask: ")
results = engine.search(query)

for text, score in results:
    print("\nMATCH:", text)
    print("Score:", score)

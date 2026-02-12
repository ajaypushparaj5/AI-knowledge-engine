from semanticsearch import SemanticSearch
from chunker import chunk_text
from pdf_loader import load_pdf

engine = SemanticSearch()

pdf_text = load_pdf("sample.pdf")

chunks = chunk_text(pdf_text)

print("Chunks:", len(chunks))

engine.add(chunks)

query = input("Ask about the PDF: ")

results = engine.search(query)

for text, score in results:
    print("\nMATCH:", text)
    print("Score:", score)

from semanticsearch import SemanticSearch

engine = SemanticSearch()

engine.add([
    "I love machine learning",
    "Deep learning is powerful",
    "Football is a fun sport",
    "Messi is a great player",
    "AI is changing the world"
])

results = engine.search("soccer player")

for text, score in results:
    print(text, score)

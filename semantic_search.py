import numpy as np
from embeddings import embed
from vector_store import save_store, load_store

class SemanticSearch:
    def __init__(self):
        self.texts = []
        self.vectors = []

    def add(self, texts):
        embeddings = embed(texts)
        for t, e in zip(texts, embeddings):
            self.texts.append(t)
            self.vectors.append(e)

    def search(self, query, top_k=3):
        print("Searching for:", query)
        print("Number of indexed texts:", len(self.texts))
        query_vec = embed([query])[0]
        print("Query vector shape:", query_vec.shape)   
        scores = []
        for vec in self.vectors:
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))

            scores.append(score)
            print("Computed score:", score)

        ranked = sorted(zip(self.texts, scores), key=lambda x: x[1], reverse=True)
        print("Ranked results:", ranked)
        return ranked[:top_k]

    def save(self):
        save_store(self.texts, self.vectors)

    def load(self):
        self.texts, self.vectors = load_store()

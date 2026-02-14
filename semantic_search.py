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

    def search(self, query, top_k=5, threshold=0.3, debug=False):
        query_vec = embed([query])[0]

        scores = []

        for vec in self.vectors:
            score = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec)
            )
            scores.append(score)

        ranked = sorted(zip(self.texts, scores),
                        key=lambda x: x[1],
                        reverse=True)

        if debug:
            print("\nAll scores:\n")
            for text, score in ranked[:10]:
                print(f"{score:.3f} → {text[:80]}")

        filtered = [r for r in ranked if r[1] >= threshold]

        return filtered[:top_k]


    def save(self):
        save_store(self.texts, self.vectors)

    def load(self):
        self.texts, self.vectors = load_store()

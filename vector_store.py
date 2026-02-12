import numpy as np
import json

def save_store(texts, vectors, prefix="store"):
    np.save(prefix + "_vectors.npy", np.array(vectors))
    
    with open(prefix + "_texts.json", "w") as f:
        json.dump(texts, f)


def load_store(prefix="store"):
    vectors = np.load(prefix + "_vectors.npy")
    
    with open(prefix + "_texts.json", "r") as f:
        texts = json.load(f)

    return texts, vectors

"""Demo: show top TF-IDF terms per chapter and query similarity."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import tokenize, clean_tokens

MODEL_PATH = Path(__file__).parent / "tfidf_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def query_to_tfidf(query: str, model) -> np.ndarray:
    """Convert a query string to a TF-IDF vector."""
    tokens = clean_tokens(tokenize(query), remove_punctuation=True)
    vocab = model["vocab"]
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    # Normalize TF
    if len(tokens) > 0:
        vec /= len(tokens)
    # Apply IDF
    vec *= model["idf"]
    return vec


def main():
    print("Loading TF-IDF model...")
    model = load_model()
    tfidf = model["tfidf"]
    names = model["chapter_names"]
    idx_to_word = model["idx_to_word"]

    # Top distinctive words per chapter
    print("=" * 60)
    print("TOP 5 DISTINCTIVE WORDS PER CHAPTER")
    print("=" * 60)
    for i, name in enumerate(names):
        top_idx = np.argsort(-tfidf[i])[:5]
        words = [idx_to_word[idx] for idx in top_idx]
        print(f"  {name}")
        print(f"    {', '.join(words)}")
        print()

    # Query similarity
    print("=" * 60)
    print("QUERY SIMILARITY")
    print("=" * 60)
    queries = [
        "Martians attacking with heat ray",
        "fleeing London in panic",
        "red weed growing everywhere",
        "artillery and military defense",
        "looking through telescope at Mars",
    ]

    for query in queries:
        q_vec = query_to_tfidf(query, model)
        sims = [(cosine_similarity(q_vec, tfidf[i]), i) for i in range(len(names))]
        sims.sort(reverse=True)
        print(f"\nQuery: \"{query}\"")
        print(f"  Most relevant chapters:")
        for sim, i in sims[:3]:
            if sim > 0:
                print(f"    {sim:.4f}  {names[i]}")


if __name__ == "__main__":
    main()

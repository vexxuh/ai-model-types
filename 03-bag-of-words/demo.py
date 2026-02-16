"""Demo: show chapter similarity and word frequencies using bag-of-words."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).parent / "bow_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def main():
    print("Loading bag-of-words model...")
    model = load_model()
    bow = model["bow_matrix"]
    names = model["chapter_names"]
    idx_to_word = model["idx_to_word"]

    print(f"Chapters: {len(names)}")
    print(f"Vocabulary: {len(model['vocab']):,} words")

    # Word frequency across entire book
    print("\n" + "=" * 60)
    print("TOP 20 WORDS IN THE ENTIRE BOOK")
    print("=" * 60)
    total_counts = bow.sum(axis=0)
    top_idx = np.argsort(-total_counts)[:20]
    for idx in top_idx:
        print(f"  {idx_to_word[idx]:15s} {int(total_counts[idx]):5d}")

    # Chapter similarity matrix
    print("\n" + "=" * 60)
    print("MOST SIMILAR CHAPTER PAIRS")
    print("=" * 60)
    n = len(names)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(bow[i], bow[j])
            similarities.append((sim, i, j))
    similarities.sort(reverse=True)

    for sim, i, j in similarities[:10]:
        print(f"  {sim:.4f}  {names[i]}")
        print(f"           {names[j]}")
        print()

    # Least similar
    print("LEAST SIMILAR CHAPTER PAIRS")
    print("=" * 60)
    for sim, i, j in similarities[-5:]:
        print(f"  {sim:.4f}  {names[i]}")
        print(f"           {names[j]}")
        print()


if __name__ == "__main__":
    main()

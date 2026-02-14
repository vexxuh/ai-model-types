"""Train a word-level bigram language model on The War of the Worlds."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_all_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "bigram_model.pkl"


def train():
    print("Loading and tokenizing text...")
    tokens = get_all_tokens()
    print(f"Total tokens: {len(tokens):,}")

    vocab = build_vocab(tokens, min_count=2)
    idx_to_word = {i: w for w, i in vocab.items()}
    print(f"Vocabulary size: {len(vocab):,}")

    # Build bigram counts
    print("Building bigram counts...")
    bigram_counts: dict[str, dict[str, int]] = {}
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 not in vocab or w2 not in vocab:
            continue
        if w1 not in bigram_counts:
            bigram_counts[w1] = {}
        bigram_counts[w1][w2] = bigram_counts[w1].get(w2, 0) + 1

    # Convert counts to probabilities
    print("Computing probabilities...")
    bigram_probs: dict[str, dict[str, float]] = {}
    for w1, followers in bigram_counts.items():
        total = sum(followers.values())
        bigram_probs[w1] = {w2: count / total for w2, count in followers.items()}

    model = {
        "bigram_probs": bigram_probs,
        "vocab": vocab,
        "idx_to_word": idx_to_word,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")

    # Show some example bigrams
    print("\nTop bigrams after 'the':")
    if "the" in bigram_probs:
        top = sorted(bigram_probs["the"].items(), key=lambda x: -x[1])[:10]
        for word, prob in top:
            print(f"  the {word}: {prob:.4f}")


if __name__ == "__main__":
    train()

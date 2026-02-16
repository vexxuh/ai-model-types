"""Train a word-level trigram language model on The War of the Worlds."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_all_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "trigram_model.pkl"


def train():
    print("Loading and tokenizing text...")
    tokens = get_all_tokens()
    print(f"Total tokens: {len(tokens):,}")

    vocab = build_vocab(tokens, min_count=2)
    print(f"Vocabulary size: {len(vocab):,}")

    # Build trigram counts: (w1, w2) -> {w3: count}
    print("Building trigram counts...")
    trigram_counts: dict[tuple[str, str], dict[str, int]] = {}
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        if w1 not in vocab or w2 not in vocab or w3 not in vocab:
            continue
        key = (w1, w2)
        if key not in trigram_counts:
            trigram_counts[key] = {}
        trigram_counts[key][w3] = trigram_counts[key].get(w3, 0) + 1

    # Convert to probabilities
    print("Computing probabilities...")
    trigram_probs: dict[tuple[str, str], dict[str, float]] = {}
    for key, followers in trigram_counts.items():
        total = sum(followers.values())
        trigram_probs[key] = {w: c / total for w, c in followers.items()}

    # Also keep bigram fallback for when trigram has no data
    bigram_counts: dict[str, dict[str, int]] = {}
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 not in vocab or w2 not in vocab:
            continue
        if w1 not in bigram_counts:
            bigram_counts[w1] = {}
        bigram_counts[w1][w2] = bigram_counts[w1].get(w2, 0) + 1

    bigram_probs: dict[str, dict[str, float]] = {}
    for w1, followers in bigram_counts.items():
        total = sum(followers.values())
        bigram_probs[w1] = {w2: c / total for w2, c in followers.items()}

    model = {
        "trigram_probs": trigram_probs,
        "bigram_probs": bigram_probs,
        "vocab": vocab,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Unique trigram contexts: {len(trigram_probs):,}")
    print(f"Unique bigram contexts: {len(bigram_probs):,}")

    # Show example
    key = ("the", "end")
    if key in trigram_probs:
        print(f"\nTop completions for 'the end':")
        top = sorted(trigram_probs[key].items(), key=lambda x: -x[1])[:5]
        for word, prob in top:
            print(f"  the end {word}: {prob:.4f}")


if __name__ == "__main__":
    train()

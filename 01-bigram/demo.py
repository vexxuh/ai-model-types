"""Generate text using the trained bigram model."""

import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).parent / "bigram_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def generate(model, seed_word=None, length=50):
    """Generate text by sampling from bigram probabilities."""
    probs = model["bigram_probs"]

    if seed_word is None or seed_word not in probs:
        seed_word = random.choice(list(probs.keys()))

    words = [seed_word]
    current = seed_word

    for _ in range(length - 1):
        if current not in probs:
            current = random.choice(list(probs.keys()))
        followers = probs[current]
        next_words = list(followers.keys())
        weights = list(followers.values())
        current = random.choices(next_words, weights=weights, k=1)[0]
        words.append(current)

    return " ".join(words)


def main():
    print("Loading bigram model...")
    model = load_model()
    print(f"Vocabulary size: {len(model['vocab']):,}")
    print(f"Words with bigram data: {len(model['bigram_probs']):,}")

    print("\n" + "=" * 60)
    print("BIGRAM TEXT GENERATION")
    print("=" * 60)

    for seed in ["the", "martians", "i", "war", "night"]:
        if seed in model["bigram_probs"]:
            print(f"\nSeed: '{seed}'")
            print(generate(model, seed_word=seed, length=30))

    print("\n" + "=" * 60)
    print("RANDOM GENERATION")
    print("=" * 60)
    print(generate(model, length=50))


if __name__ == "__main__":
    main()

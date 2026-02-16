"""Generate text using the trained trigram model."""

import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).parent / "trigram_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def sample_from(dist: dict[str, float]) -> str:
    words = list(dist.keys())
    weights = list(dist.values())
    return random.choices(words, weights=weights, k=1)[0]


def generate(model, seed=None, length=50):
    """Generate text using trigram probabilities with bigram fallback."""
    tri_probs = model["trigram_probs"]
    bi_probs = model["bigram_probs"]

    # Pick a starting bigram
    if seed and len(seed) == 2 and tuple(seed) in tri_probs:
        w1, w2 = seed
    else:
        key = random.choice(list(tri_probs.keys()))
        w1, w2 = key

    words = [w1, w2]

    for _ in range(length - 2):
        key = (w1, w2)
        if key in tri_probs:
            w3 = sample_from(tri_probs[key])
        elif w2 in bi_probs:
            w3 = sample_from(bi_probs[w2])
        else:
            w3 = sample_from(bi_probs[random.choice(list(bi_probs.keys()))])
        words.append(w3)
        w1, w2 = w2, w3

    return " ".join(words)


def main():
    print("Loading trigram model...")
    model = load_model()
    print(f"Vocabulary size: {len(model['vocab']):,}")
    print(f"Trigram contexts: {len(model['trigram_probs']):,}")

    print("\n" + "=" * 60)
    print("TRIGRAM TEXT GENERATION")
    print("=" * 60)

    seeds = [("the", "martians"), ("i", "saw"), ("in", "the"), ("it", "was")]
    for seed in seeds:
        if tuple(seed) in model["trigram_probs"]:
            print(f"\nSeed: '{seed[0]} {seed[1]}'")
            print(generate(model, seed=seed, length=40))

    print("\n" + "=" * 60)
    print("RANDOM GENERATION")
    print("=" * 60)
    print(generate(model, length=60))


if __name__ == "__main__":
    main()

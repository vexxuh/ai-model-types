"""Build bag-of-words representations for each chapter."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_chapters, tokenize, clean_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "bow_model.pkl"


def train():
    print("Loading chapters...")
    chapters = get_chapters()
    print(f"Found {len(chapters)} chapters")

    # Tokenize each chapter
    chapter_tokens = []
    for ch in chapters:
        tokens = clean_tokens(tokenize(ch["text"]), remove_punctuation=True)
        chapter_tokens.append(tokens)

    # Build vocabulary from all tokens
    all_tokens = [t for ct in chapter_tokens for t in ct]
    vocab = build_vocab(all_tokens, min_count=3)
    print(f"Vocabulary size (min_count=3): {len(vocab):,}")

    # Build BoW matrix: chapters x vocab_size
    bow_matrix = np.zeros((len(chapters), len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(chapter_tokens):
        for t in tokens:
            if t in vocab:
                bow_matrix[i, vocab[t]] += 1

    # Chapter metadata
    chapter_names = [
        f"Book {ch['book']}, Ch {ch['chapter']}: {ch['title']}"
        for ch in chapters
    ]

    model = {
        "bow_matrix": bow_matrix,
        "vocab": vocab,
        "idx_to_word": {i: w for w, i in vocab.items()},
        "chapter_names": chapter_names,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"BoW matrix shape: {bow_matrix.shape}")

    # Show top words per first chapter
    print(f"\nTop 10 words in '{chapter_names[0]}':")
    top_idx = np.argsort(-bow_matrix[0])[:10]
    for idx in top_idx:
        word = model["idx_to_word"][idx]
        count = int(bow_matrix[0, idx])
        print(f"  {word}: {count}")


if __name__ == "__main__":
    train()

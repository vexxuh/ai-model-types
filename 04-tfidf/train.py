"""Compute TF-IDF scores for each chapter of The War of the Worlds."""

import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_chapters, tokenize, clean_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "tfidf_model.pkl"


def train():
    print("Loading chapters...")
    chapters = get_chapters()
    print(f"Found {len(chapters)} chapters")

    # Tokenize each chapter
    chapter_tokens = []
    for ch in chapters:
        tokens = clean_tokens(tokenize(ch["text"]), remove_punctuation=True)
        chapter_tokens.append(tokens)

    # Build vocabulary
    all_tokens = [t for ct in chapter_tokens for t in ct]
    vocab = build_vocab(all_tokens, min_count=3)
    vocab_size = len(vocab)
    n_docs = len(chapters)
    print(f"Vocabulary size: {vocab_size:,}")

    # Term frequency: count of word in each document
    tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for i, tokens in enumerate(chapter_tokens):
        for t in tokens:
            if t in vocab:
                tf[i, vocab[t]] += 1
        # Normalize by document length
        doc_len = len(tokens)
        if doc_len > 0:
            tf[i] /= doc_len

    # Document frequency: how many documents contain each word
    df = np.zeros(vocab_size, dtype=np.float32)
    for i in range(n_docs):
        for j in range(vocab_size):
            if tf[i, j] > 0:
                df[j] += 1

    # IDF: log(N / df), with smoothing
    idf = np.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF

    # TF-IDF
    tfidf = tf * idf[np.newaxis, :]

    chapter_names = [
        f"Book {ch['book']}, Ch {ch['chapter']}: {ch['title']}"
        for ch in chapters
    ]

    model = {
        "tfidf": tfidf,
        "tf": tf,
        "idf": idf,
        "vocab": vocab,
        "idx_to_word": {i: w for w, i in vocab.items()},
        "chapter_names": chapter_names,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"TF-IDF matrix shape: {tfidf.shape}")

    # Show top TF-IDF terms for first chapter
    print(f"\nTop distinctive words in '{chapter_names[0]}':")
    top_idx = np.argsort(-tfidf[0])[:10]
    for idx in top_idx:
        word = model["idx_to_word"][idx]
        score = tfidf[0, idx]
        print(f"  {word:20s} {score:.4f}")


if __name__ == "__main__":
    train()

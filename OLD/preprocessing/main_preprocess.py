from preprocessing.abc_parser import load_abc_tunes, clean_abc_tunes, extract_melody_with_context
from preprocessing.normalize import normalize_abc
from preprocessing.tokenize import tokenize_abc
from preprocessing.vocab import build_vocab


def preprocess_abc_dataset(path="data/"):
    raw_tunes = load_abc_tunes(path)
    clean_tunes = clean_abc_tunes(raw_tunes)
    melodies = [extract_melody_with_context(tune) for tune in clean_tunes]
    normalized_melodies = [normalize_abc(melody) for melody in melodies]
    tokenized_melodies = [tokenize_abc(melody) for melody in normalized_melodies]

    vocab, indexed_melodies, token_freq = build_vocab(tokenized_melodies)
    inv_vocab = {v: k for k, v in vocab.items()}

    print(f"Total raw tunes extracted: {len(raw_tunes)}")
    print(f"Total tunes after cleaning: {len(clean_tunes)}")
    print("Example melody:", normalized_melodies[0][:45])
    print("Example tokens:", tokenized_melodies[0][:18])
    print(f"Number of unique tokens: {len(vocab)}")

    return vocab, inv_vocab, indexed_melodies, token_freq, normalized_melodies

import re
from collections import Counter
from collections import defaultdict

def build_vocab(tokenized_melodies):
    """
    Builds a vocabulary (token-to-index mapping) from a list of tokenized ABC melodies.

    Parameters:
        tokenized_melodies (List[List[str]]): A list where each melody is a list of ABC tokens.

    Returns:
        vocab (Dict[str, int]): Mapping from tokens to unique integer indices.
        indexed_melodies (List[List[int]]): Each melody as a list of integer token indices.
        token_freq (Counter): Frequency count of all tokens.
    """
    # Flatten all tokenized melodies into a single list of tokens
    all_tokens = [token for melody in tokenized_melodies for token in melody]

    # Count how often each token appears
    token_freq = Counter(all_tokens)

    # Create vocabulary: map each unique token to a unique index
    # Start from 1 so that 0 can be reserved for padding
    vocab = {token: idx for idx, (token, _) in enumerate(token_freq.items(), start=3)}
    vocab["<PAD>"] = 0   # Add padding token (index 0)
    vocab["<START>"] = 1 # Add start token (index 1)
    vocab["<END>"] = 2   # Add end token (index 2)


    # Convert each tokenized melody to a list of token indices
    indexed_melodies = [[vocab[token] for token in melody] for melody in tokenized_melodies]

    return vocab, indexed_melodies, token_freq


def print_grouped_tokens(vocab):
    """
    Prints grouped tokens from a vocabulary in musically meaningful categories.

    Parameters:
        vocab (Dict[str, int]): Mapping from tokens to indices.
    """
    grouped_tokens = defaultdict(list)

    for token in vocab:
        if token == "<PAD>":
            grouped_tokens["Special"].append(token)
        elif re.match(r'^K:[A-G][#b]?m?$', token):
            grouped_tokens["Key"].append(token)
        elif re.match(r'^M:\d+/\d+$', token):
            grouped_tokens["TimeSig"].append(token)
        elif re.match(r'^L:\d+/\d+$', token):
            grouped_tokens["NoteLength"].append(token)
        elif re.match(r'^"[^"]+"$', token):
            grouped_tokens["Chords"].append(token)
        elif re.match(r'^\[1|\[2$', token):
            grouped_tokens["RepeatBrackets"].append(token)
        elif re.match(r'^::|:\|\||\|:|:\||\|\||\|$', token):
            grouped_tokens["Barlines"].append(token)
        elif re.match(r'^[=^_]*[A-Ga-grz][\',]?\d*/?\d*$', token):
            grouped_tokens["Notes"].append(token)
        else:
            grouped_tokens["Other/Weird"].append(token)

    # Print summary
    for group, tokens in grouped_tokens.items():
        print(f"\n--- {group} ({len(tokens)} tokens) ---")
        for token in sorted(tokens):
            print(token)

import re

def normalize_slash_to_half(abc_str):
    """
    Replaces note durations written with a dangling '/' (e.g., 'b/') 
    with '/2' to standardize fractional durations.
    """
    return re.sub(r'([=^_]*[A-Ga-grz][\',]?)\/(?!\d)', r'\1/2', abc_str)

def normalize_chord_formatting(abc_str):
    """
    Cleans up chord notation:
    - Removes parentheses around chords (e.g., "(A7)" → "A7")
    - Fixes space typos (e.g., "D m" → "Dm")
    - Strips leading/trailing whitespace inside quotes (e.g., '" Em"' → '"Em"')
    """

    def clean_chord(match):
        chord = match.group(0)
        chord = chord.strip('"() ')       # remove outer symbols and spaces
        chord = chord.replace(" ", "")    # squash space typos like "D m"
        return f'"{chord}"'

    return re.sub(r'"[^"]+"', clean_chord, abc_str)

def normalize_abc(abc_str):
    abc_str = normalize_slash_to_half(abc_str)
    abc_str = normalize_chord_formatting(abc_str)
    return abc_str

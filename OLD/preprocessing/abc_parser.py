import os


def load_abc_tunes(data_dir):
    """
    Each .abc file in the Nottingham dataset contains multiple tunes.
    Loads all .abc files from data_dir and returns a list of all tunes.
    """
    abc_files = [f for f in os.listdir(data_dir) if f.endswith(".abc")]
    all_tunes = []

    for filename in abc_files:
        with open(os.path.join(data_dir, filename), 'r') as file:
            content = file.read()
            tunes = content.split('\n\n') # each song is separated by a blank line
            all_tunes.extend(tunes)

    return all_tunes


def clean_abc_tunes(raw_tunes):
    """
    Some tunes may be incomplete or junk. 
    A simple heuristic: keep only tunes with a key signature K: and a time signature M:
    """
    clean_tunes = [t for t in raw_tunes if 'K:' in t and 'M:' in t]
    return clean_tunes
 

def extract_melody_with_context(tune):
    """
    Extracts the essential musical information and melody from an ABC tune.
    Automatically fills in a default L: value if missing, based on M: (meter).

    Parameters:
        tune (str): A raw ABC tune.

    Returns:
        A string containing only the essential musical context and the melody.

    Notes:
        In ABC notation, tunes begin with a header section that may contain many fields.
        Only some of these fields are musically relevant to interpreting the notes.

        We keep:
            - M: Meter (e.g., "M:4/4")
            - L: Default note length (e.g., "L:1/8")
            - K: Key signature (e.g., "K:Cmaj")

        These fields are needed to interpret rhythm and pitch correctly.

        We discard:
            - X: Tune number (just an ID)
            - T: Title
            - N:, C:, Z:, Q:, etc. — any fields that are comments, composer names,
              tempo hints, etc., which are often inconsistent or irrelevant for modeling.

        Once we reach the 'K:' line (the key signature), we start including melody lines,
        which contain the actual note sequences.

        Rules for default L:
        - If no L: line is provided, compute decimal value of the meter (M:)
        - If meter >= 0.75 → default to L:1/8
        - If meter <  0.75 → default to L:1/16
    """
    lines = tune.strip().splitlines()

    meter_line = None
    length_line = None
    key_line = None
    melody_lines = []

    header_done = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not header_done:
            if line.startswith("M:") and meter_line is None:
                meter_line = line
            elif line.startswith("L:") and length_line is None:
                length_line = line
            elif line.startswith("K:") and key_line is None:
                key_line = line
                header_done = True  # everything after this is melody
        else:
            melody_lines.append(line)

    # Synthesize L: if missing
    if length_line is None:
        default_length = "1/8"  # fallback
        if meter_line:
            try:
                meter_value = meter_line[2:].strip()
                num, denom = map(int, meter_value.split('/'))
                meter_decimal = num / denom
                default_length = "1/8" if meter_decimal >= 0.75 else "1/16"
            except Exception:
                pass
        length_line = f"L:{default_length}"

    header = []
    if meter_line:
        header.append(meter_line)
    if length_line:
        header.append(length_line)
    if key_line:
        header.append(key_line)

    return ' '.join(header + melody_lines)
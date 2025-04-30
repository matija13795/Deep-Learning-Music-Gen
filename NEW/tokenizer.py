import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


_BAR_TOKENS = ["||", "|]", "|:", ":|", "::", "|"]
_ACCIDENTALS = ["^^", "__", "^", "_", "="]
_RESTS = {"z", "Z", "x"}
_NOTE_LETTERS = set("ABCDEFGabcdefg")
_HDR_RE = re.compile(r'^(?:K|M|L):[^\n]*\n', re.MULTILINE)
_INLINE_HDR_RE = re.compile(r'\[(?:K|M|L):[^\]]+\]') # [K:C], [M:3/4] ...
_TUPLET_RE = re.compile(r"\(\d+")                    # (3   (5   etc.
_CHORD_RE = re.compile(
    r'^"'
    r'(?P<root>[A-G](?:b|#)?)'          # root note
    r'(?P<body>[0-9A-Za-z+\-#]*)'       # quality / extensions
    r'(?:/(?P<bass>[A-G](?:b|#)?))?'    # optional slash bass
    r'"$'
)
_SKIP_Q_RE = re.compile(r'\[Q:[^\]]+\]')
_DENOM_POW2 = r'(?:2|4|8|16)' # power-of-two denominators up to 16
_DUR_RE = re.compile(
    rf'''
    (?:[1-9]\d?/{_DENOM_POW2})  |  # 3/2  7/8  12/16
    (?:/)  |      # / 
    (?:[2-8]|16)  # 2-8 or 16
    ''',
    re.VERBOSE
)
_SQUARE_BRACKET = re.compile(r"\[[^\]]+\]")


class ABCTokenizer:
    """Tokenizer for ABC tunes.

    Parameters
    ----------
    pad_token : str, default "<PAD>"
        Special symbol used for sequence padding.
    """
    def __init__(self, pad_token="<PAD>",
                 chord_map: Optional[Dict[str, str]] = None,
                 header_map: Optional[Dict[str, str]] = None,
                 inline_hdr_map: Optional[Dict[str, str]] = None):
        self.pad_token = pad_token
        self.chord_map = chord_map or {}
        self.header_map = header_map or {}
        self.inline_hdr_map = inline_hdr_map or {}
        self.stoi: dict[str, int] = {pad_token: 0}  # string‑to‑index
        self.itos: dict[int, str] = {0: pad_token}  # index‑to‑string

    def tokenize_abc(self, abc_str: str) -> List[str]:
        """Tokenize an ABC string into musically meaningful symbols."""
        tokens: List[str] = []
        i, s = 0, abc_str
        bar_long = _BAR_TOKENS

        while i < len(s):
            ch = s[i]

            # ---- skip everything in between explamation marks !__! -----
            if ch =='!': # stuff inside !__! is not musically relevant
                j = s.find('!', i+1)
                if j != -1:
                    i = j+1
                    continue
                else:
                    i+=1
                    continue

            if ch == '"':
                j = s.find('"', i + 1)
                if j != -1:
                    if s[j-1] == '\\':
                        # backslash before quote -> find next \""
                        j2 = s.find('\\""', j + 1)
                        j3 = s.find('\\")"', j + 1)
                        if j2 != -1:
                            i = j2 + 1  # skip after \""
                        elif j3 != -1:
                            i = j3 + 1
                else:
                    i += 1  # no closing quote at all
                    continue

            # ---- chord symbols *only if they match strict pattern* ------
            if ch == '"':
                j = s.find('"', i + 1)
                if j != -1:
                    chunk = s[i:j + 1]
                    if _CHORD_RE.fullmatch(chunk):
                        # normalise if mapping supplied
                        chunk = self.chord_map.get(chunk, chunk)
                        tokens.append(chunk)
                    # else: NOT a chord -> ignore the whole thing
                    i = j + 1 # jump past the closing quote
                    continue
                else:
                    # unmatched opening quote – ignore it and move on
                    i += 1
                    continue

            # skip inline tempo  [Q:...]
            m = _SKIP_Q_RE.match(s, i)
            if m:
                i = m.end()
                continue

            # ---- header lines (only K:, M:, L:) -------------------------
            if s[i] in "KML" and (i == 0 or s[i-1] == "\n"):
                m = _HDR_RE.match(s, i)
                if m:
                    raw = m.group()
                    mapped = self.header_map.get(raw, self.header_map.get(raw.rstrip("\n"), raw))
                    tokens.append(mapped)
                    i = m.end()
                    continue

            # ---- skip *other* header lines ------------------------------
            if (i == 0 or s[i - 1] == "\n") and ch.isupper() and i + 1 < len(s) and s[i + 1] == ":":
                j = i + 2
                while j < len(s) and s[j] != "\n":
                    j += 1
                i = j + 1
                continue

            # ---- inline key / meter changes -----------------------------
            m = _INLINE_HDR_RE.match(s, i)
            if m:
                raw = m.group()
                mapped = self.inline_hdr_map.get(raw, raw)
                tokens.append(mapped)
                i = m.end()
                continue

            # ---- tuplets  (3   (5 --------------------------------------
            m = _TUPLET_RE.match(s, i)
            if m:
                tokens.append(m.group())
                i = m.end()
                continue

            # ---- broken rhythm <   > -----------------------------------
            if ch in "<>":
                tokens.append(ch)
                i += 1
                continue

            # ---- bar-lines / repeats ------------------------------------
            matched = False
            for sym in bar_long:
                if s.startswith(sym, i):
                    tokens.append(sym)
                    i += len(sym)
                    matched = True
                    break
            if matched:
                continue

            # ---- accidentals -------------------------------------------
            for acc in _ACCIDENTALS:
                if s.startswith(acc, i):
                    tokens.append(acc)
                    i += len(acc)
                    matched = True
                    break
            if matched:
                continue

            # ---- articulations  -   . ----------------------------------
            if ch in "-.":
                tokens.append(ch)
                i += 1
                continue

            # ---- note letters ------------------------------------------
            if ch in _NOTE_LETTERS:
                tokens.append(ch)
                i += 1
                continue

            # ---- rests --------------------------------------------------
            if ch in _RESTS:
                tokens.append(ch)
                i += 1
                continue

            # ---- octave marks ,  ' -------------------------------------
            if ch in "',":
                tokens.append(ch)
                i += 1
                continue

            # ---- durations ---------------------------------------------
            m = _DUR_RE.match(s, i)
            if m:
                tokens.append(m.group())
                i = m.end()
                continue

            # ---- new-line ------------------------------------------------
            if ch == "\n":
                tokens.append("\n")
                i += 1
                continue

            # ---- collapse runs of spaces into one space token -----------
            if ch == " ":
                tokens.append(" ")
                while i < len(s) and s[i] == " ":
                    i += 1
                continue

            # ---- square brackets -----------------------------------------
            if ch == "[":
                tokens.append("[")
                i += 1
                continue
            if ch == "]":
                tokens.append("]")
                i += 1
                continue

            # ---- curly brackets (grace notes) ---------------------------
            if ch == "{":
                tokens.append("{")
                i += 1
                continue
            if ch == "}":
                tokens.append("}")
                i += 1
                continue

            if ch == "1":
                tokens.append("1")
                i += 1
                continue

            # ---- everything else – ignore ------------------------------
            i += 1

        return tokens

    def build_vocab(self, tunes: List[str]) -> None:
        """Create mapping tables from a list of ABC tunes."""
        unique_tokens = set()
        for tune in tunes:
            unique_tokens.update(self.tokenize_abc(tune))

        for token in sorted(unique_tokens):
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

    def encode(self, tune: str) -> List[int]:
        """Convert an ABC tune into a list of integer IDs."""
        return [self.stoi[token] for token in self.tokenize_abc(tune)]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of IDs back to a string (ignores PAD)."""
        return "".join(self.itos[i] for i in ids if i != 0)

    def vocab_size(self) -> int:
        return len(self.stoi)

    def print_grouped_tokens(self, vocab: Dict[str, int]) -> None:
        groups = defaultdict(list)
        for tok in vocab:
            if tok == self.pad_token:
                continue
            if tok in _ACCIDENTALS:
                groups["Accidental"].append(tok)
            elif tok in _BAR_TOKENS or tok in ("|", ":"):
                groups["Bar/Repeat"].append(tok)
            elif _HDR_RE.fullmatch(tok):
                groups["Header"].append(repr(tok))
            elif _INLINE_HDR_RE.fullmatch(tok):
                groups["Inline Header Change"].append(repr(tok))
            elif _CHORD_RE.fullmatch(tok):
                groups["Chord"].append(tok)
            elif _TUPLET_RE.fullmatch(tok):
                groups["Tuplet"].append(tok)
            elif tok in "<>":
                groups["Broken"].append(tok)
            elif tok in "-.":
                groups["Articulation"].append(tok)
            elif _DUR_RE.fullmatch(tok):
                groups["Duration"].append(tok)
            elif tok in _NOTE_LETTERS:
                groups["Note"].append(tok)
            elif tok in _RESTS:
                groups["Rest"].append(tok)
            elif tok in "',":
                groups["Octave"].append(tok)
            elif tok == " ":
                groups["Space"].append(repr(tok))
            elif tok == "\n":
                groups["Newline"].append(repr(tok))
            elif tok == "[" or tok == "]":
                groups["Chord-like melodic groupings"].append(tok)
            elif tok == "{" or tok == "}":
                groups["Grace Notes"].append(tok)
            else:
                groups["Misc"].append(tok)

        for name in sorted(groups):
            print(f"{name:<15}: {' '.join(groups[name])}")
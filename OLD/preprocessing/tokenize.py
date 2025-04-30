import re

def tokenize_abc(abc_str):
    """
    Tokenizes an ABC string into musically meaningful symbols.

    The tokenizer extracts:
        - Bar lines and repeat symbols: '|', '||', ':|', '|:', '::'
        - Repeat brackets: '[1', '[2'
        - Chords: enclosed in double quotes, e.g., "G", "Dmin"
        - Notes: including accidentals (=, ^, _), octave markers (',), rests (z), and durations (e.g., A3/2)
        - Headers: 
            - M: (time signature), e.g., M:4/4
            - L: (default note length), e.g., L:1/8
            - K: (key signature), e.g., K:D

    Parameters:
        abc_str (str): A string in ABC notation.

    Returns:
        list[str]: A list of tokens representing the music in ABC format.
    """

    tokens = re.findall(r'''
          \[1|\[2                          # repeat brackets [1, [2
        | :\|\|                            # :|| (repeat + end)
        | :\|                              # :|  (end of repeat)
        | \|:                              # |:  (start of repeat)
        | ::                               # ::  (double repeat)
        | \|\|                             # ||  (end of section)
        | \|                               # |   (single bar)
        | \|\||\|                          # barlines
        | "[^"]+"                          # chords, e.g., "G", "Dmin"
        | [=^_]*[A-Ga-grz][\',]?\d*\/?\d*  # notes (d2, A3/2, z, etc)
        | M:\d+\/\d+                       # time signature
        | L:\d+\/\d+                       # default note length
        | K:[A-G][#b]?m?                   # key signature
        ''',
        abc_str,
        re.VERBOSE
    )
    return tokens


def run_tokenizer_tests():
    test_cases = [
        {
            "input": 'M:4/4 L:1/8 K:C CDEF GABc|',
            "expected": ['M:4/4', 'L:1/8', 'K:C', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', '|']
        },
        {
            "input": 'K:G "D7" D2 G2 |: B4 :|',
            "expected": ['K:G', '"D7"', 'D2', 'G2', '|:', 'B4', ':|']
        },
        {
            "input": 'L:1/16 M:6/8 K:D z3 [1 A2 B2 :||',
            "expected": ['L:1/16', 'M:6/8', 'K:D', 'z3', '[1', 'A2', 'B2', ':||']
        },
        {
            "input": 'K:D | A3/2 B/2 c\'/ | "G" G,2 |',
            "expected": ['K:D', '|', 'A3/2', 'B/2', "c'/", '|', '"G"', 'G,2', '|']
        }
    ]

    for i, case in enumerate(test_cases, 1):
        result = tokenize_abc(case['input'])
        if result != case['expected']:
            print(f"[FAIL] Test {i}")
            print("Input:   ", case['input'])
            print("Expected:", case['expected'])
            print("Got:     ", result)
        else:
            print(f"[PASS] Test {i}")

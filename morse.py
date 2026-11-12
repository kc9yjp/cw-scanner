#!/usr/bin/env python3
"""morse.py — Morse code encoding and decoding, with no hardware dependency.

Two jobs live here:

1. Encoding: turn text into a keying envelope (a 0/1 array where 1 means
   "transmitter on"). This is how the self-test builds synthetic CW signals.

2. Decoding: turn a binary envelope (or a list of mark/space durations)
   back into text. The decoder is deliberately simple and explainable:

   * A "unit" is the length of one dit. A dah is 3 units, the gap inside a
     character is 1 unit, between characters 3 units, between words 7 units.
   * If the speed (WPM) is known we use it directly: unit = 1.2 / WPM.
   * If not, we estimate the unit from the received marks: dits are the
     shortest marks and usually the most common, so the median of the
     shorter half of the marks is a good estimate of one unit.
   * A mark shorter than 2 units is a dit, otherwise a dah. A space shorter
     than 2 units is inside a character, 2-5 units ends a character, and
     more than 5 units ends a word.

   This is a plain timing decoder: it assumes the envelope is already a
   clean keying waveform (the job of the detector in cwscan.py). Fading,
   interference, and sloppy hand-sent Morse will all hurt it, which is
   expected for a first version.
"""

import numpy as np

# Standard International Morse code.
MORSE_CODE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.",  "(": "-.--.",  ")": "-.--.-",
    "&": ".-...",  ":": "---...", ";": "-.-.-.", "=": "-...-",
    "+": ".-.-.",  "-": "-....-", "_": "..--.-", '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
}
CODE_TO_CHAR = {code: char for char, code in MORSE_CODE.items()}


def unit_duration(wpm):
    """Length of one dit, in seconds, at the given speed."""
    return 1.2 / float(wpm)


def text_to_morse(text):
    """'CQ' -> '-.-. --.-' (words separated by ' / ')."""
    words = []
    for word in text.upper().split(" "):
        chars = [MORSE_CODE[ch] for ch in word if ch in MORSE_CODE]
        words.append(" ".join(chars))
    return " / ".join(words)


def keying_envelope(text, wpm=20, sample_rate=8000, lead_s=0.2, tail_s=0.2):
    """Build a 0/1 keying envelope for `text` at `wpm`.

    Returns a numpy float array at `sample_rate`: 1.0 while the key is down,
    0.0 while it is up. A short silence is added at each end so decoders
    don't have to deal with a signal that starts mid-mark.
    """
    unit = unit_duration(wpm)
    # A list of (level, length_in_units) segments.
    segments = [(0.0, int(round(lead_s / unit)))]
    words = text.upper().split(" ")
    for wi, word in enumerate(words):
        for ci, char in enumerate(word):
            code = MORSE_CODE.get(char)
            if code is None:
                continue
            for symbol in code:
                # Dit = 1 unit down, dah = 3 units down, 1 unit up after each.
                segments.append((1.0, 1 if symbol == "." else 3))
                segments.append((0.0, 1))
            # Inter-character gap is 3 units total; 1 is already there.
            segments.append((0.0, 2))
        # Inter-word gap is 7 units total; 3 are already there.
        if wi < len(words) - 1:
            segments.append((0.0, 4))
    segments.append((0.0, int(round(tail_s / unit))))

    total = sum(int(round(n * unit * sample_rate)) for _, n in segments)
    envelope = np.zeros(total, dtype=float)
    pos = 0
    for level, n_units in segments:
        length = int(round(n_units * unit * sample_rate))
        envelope[pos:pos + length] = level
        pos += length
    return envelope


def _runs(binary):
    """Run-length encode a 0/1 array -> list of (level, length_in_samples)."""
    binary = np.asarray(binary) > 0.5
    if len(binary) == 0:
        return []
    change = np.flatnonzero(np.diff(binary.astype(np.int8))) + 1
    edges = np.concatenate(([0], change, [len(binary)]))
    return [(bool(binary[s]), int(e - s)) for s, e in zip(edges[:-1], edges[1:])]


def _estimate_unit(marks, wpm=None):
    """Estimate one dit-length (seconds) from observed mark durations."""
    if wpm is not None:
        return unit_duration(wpm)
    marks = sorted(marks)
    if not marks:
        return None
    # Dits are the shortest marks and usually the most common, so the
    # median of the shorter half is a robust estimate of one unit.
    lower_half = marks[:max(1, len(marks) // 2)]
    return float(np.median(lower_half))


def decode_transitions(transitions, wpm=None):
    """Decode a list of (is_mark, duration_seconds) into text.

    `transitions` is what you get from run-length encoding a binary keying
    envelope. This is the core decoder; decode_envelope() is a thin wrapper
    around it.
    """
    marks = [d for is_mark, d in transitions if is_mark]
    unit = _estimate_unit(marks, wpm=wpm)
    if unit is None or unit <= 0:
        return ""

    text = []
    current = ""  # dots/dashes of the character being built
    for is_mark, duration in transitions:
        if is_mark:
            current += "." if duration < 2.0 * unit else "-"
        else:
            if duration < 2.0 * unit:
                pass  # gap inside a character
            elif duration < 5.0 * unit:
                # End of a character.
                if current:
                    text.append(CODE_TO_CHAR.get(current, "?"))
                    current = ""
            else:
                # End of a character and a word gap.
                if current:
                    text.append(CODE_TO_CHAR.get(current, "?"))
                    current = ""
                text.append(" ")
    if current:
        text.append(CODE_TO_CHAR.get(current, "?"))
    # Collapse any repeated spaces from long pauses.
    return " ".join("".join(text).split())


def decode_envelope(envelope, sample_rate, wpm=None):
    """Decode a binary (or thresholded) keying envelope into text.

    `envelope` is a 0/1 array (values > 0.5 count as key-down) at
    `sample_rate` samples per second. Pass `wpm` when the speed is known;
    otherwise the unit length is estimated from the signal itself.
    """
    runs = _runs(envelope)
    if not runs:
        return ""
    # Ignore leading/trailing silence.
    while runs and not runs[0][0]:
        runs.pop(0)
    while runs and not runs[-1][0]:
        runs.pop()
    transitions = [(is_mark, length / float(sample_rate))
                   for is_mark, length in runs]
    return decode_transitions(transitions, wpm=wpm)

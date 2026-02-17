"""
Music theory utilities for MarkovMIDI.

Provides note/MIDI conversion, scale generation, and chord building
for key-agnostic Markov chain training and transposition on output.
"""

from typing import Final

# =============================================================================
# Note Names and MIDI Mapping
# =============================================================================

# Mapping from pitch class (0-11) to note names (with enharmonic equivalents)
# Index 0 = C, 1 = C#/Db, etc.
PITCH_CLASS_TO_NAMES: Final[dict[int, tuple[str, ...]]] = {
    0: ("C",),
    1: ("C#", "Db"),
    2: ("D",),
    3: ("D#", "Eb"),
    4: ("E",),
    5: ("F",),
    6: ("F#", "Gb"),
    7: ("G",),
    8: ("G#", "Ab"),
    9: ("A",),
    10: ("A#", "Bb"),
    11: ("B",),
}

# Reverse mapping: note name to pitch class
NAME_TO_PITCH_CLASS: Final[dict[str, int]] = {
    name: pitch_class
    for pitch_class, names in PITCH_CLASS_TO_NAMES.items()
    for name in names
}

# All valid note names for validation
ALL_NOTE_NAMES: Final[frozenset[str]] = frozenset(NAME_TO_PITCH_CLASS.keys())

# Preferred note names (using sharps by default)
SHARP_NAMES: Final[tuple[str, ...]] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

FLAT_NAMES: Final[tuple[str, ...]] = (
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

# =============================================================================
# Scale Definitions
# =============================================================================

# Scale intervals as semitones from root
SCALE_INTERVALS: Final[dict[str, tuple[int, ...]]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),  # Natural minor
}

# Keys that conventionally use flats (major keys with flats in key signature)
# F major (1 flat), Bb major (2 flats), Eb major (3 flats), etc.
FLAT_MAJOR_KEYS: Final[frozenset[str]] = frozenset(
    {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"}
)

# Minor keys that use flats (relative minors of flat major keys + others)
# D minor, G minor, C minor, F minor, Bb minor, Eb minor, Ab minor
FLAT_MINOR_KEYS: Final[frozenset[str]] = frozenset(
    {"D", "G", "C", "F", "Bb", "Eb", "Ab"}
)

# =============================================================================
# Chord Definitions
# =============================================================================

# Chord intervals as semitones from root
CHORD_INTERVALS: Final[dict[str, tuple[int, ...]]] = {
    # Triads
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    # Seventh chords
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "min7b5": (0, 3, 6, 10),  # Half-diminished
}

# =============================================================================
# Note/MIDI Conversion Functions
# =============================================================================


def note_to_midi(note: str, octave: int) -> int:
    """
    Convert a note name and octave to MIDI note number.

    Args:
        note: Note name (e.g., "C", "F#", "Bb")
        octave: Octave number (e.g., 4 for middle C)

    Returns:
        MIDI note number (0-127)

    Raises:
        ValueError: If note name is invalid

    Example:
        >>> note_to_midi("C", 4)
        60
        >>> note_to_midi("A", 4)
        69
    """
    note_upper = note.capitalize()
    if note_upper not in NAME_TO_PITCH_CLASS:
        raise ValueError(f"Invalid note name: {note}")

    pitch_class = NAME_TO_PITCH_CLASS[note_upper]
    midi_number = (octave + 1) * 12 + pitch_class

    if not 0 <= midi_number <= 127:
        raise ValueError(f"MIDI number {midi_number} out of range (0-127)")

    return midi_number


def midi_to_note(midi: int, prefer_sharps: bool = True) -> tuple[str, int]:
    """
    Convert a MIDI note number to note name and octave.

    Args:
        midi: MIDI note number (0-127)
        prefer_sharps: If True, use sharps; if False, use flats

    Returns:
        Tuple of (note_name, octave)

    Raises:
        ValueError: If MIDI number is out of range

    Example:
        >>> midi_to_note(60)
        ('C', 4)
        >>> midi_to_note(61, prefer_sharps=False)
        ('Db', 4)
    """
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI number {midi} out of range (0-127)")

    pitch_class = midi % 12
    octave = (midi // 12) - 1

    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES
    note_name = names[pitch_class]

    return (note_name, octave)


def get_pitch_class(note: str) -> int:
    """
    Get the pitch class (0-11) for a note name.

    Args:
        note: Note name (e.g., "C", "F#", "Bb")

    Returns:
        Pitch class (0 = C, 1 = C#/Db, etc.)

    Raises:
        ValueError: If note name is invalid
    """
    note_upper = note.capitalize()
    if note_upper not in NAME_TO_PITCH_CLASS:
        raise ValueError(f"Invalid note name: {note}")
    return NAME_TO_PITCH_CLASS[note_upper]


# =============================================================================
# Scale Functions
# =============================================================================


def get_scale_intervals(mode: str) -> tuple[int, ...]:
    """
    Get the intervals (in semitones) for a scale mode.

    Args:
        mode: Scale mode ("major" or "minor")

    Returns:
        Tuple of semitone intervals from the root

    Raises:
        ValueError: If mode is not supported

    Example:
        >>> get_scale_intervals("major")
        (0, 2, 4, 5, 7, 9, 11)
    """
    mode_lower = mode.lower()
    if mode_lower not in SCALE_INTERVALS:
        raise ValueError(
            f"Unsupported mode: {mode}. Use: {list(SCALE_INTERVALS.keys())}"
        )
    return SCALE_INTERVALS[mode_lower]


def get_scale_notes(root: str, mode: str) -> list[str]:
    """
    Get the notes in a scale.

    Args:
        root: Root note of the scale (e.g., "C", "F#")
        mode: Scale mode ("major" or "minor")

    Returns:
        List of note names in the scale

    Example:
        >>> get_scale_notes("C", "major")
        ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        >>> get_scale_notes("A", "minor")
        ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    """
    intervals = get_scale_intervals(mode)
    root_pitch_class = get_pitch_class(root)

    # Determine whether to use sharps or flats based on key and mode
    root_upper = root.capitalize()
    mode_lower = mode.lower()
    if mode_lower == "major":
        prefer_sharps = root_upper not in FLAT_MAJOR_KEYS
    else:  # minor
        prefer_sharps = root_upper not in FLAT_MINOR_KEYS
    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES

    scale_notes: list[str] = []
    for interval in intervals:
        pitch_class = (root_pitch_class + interval) % 12
        scale_notes.append(names[pitch_class])

    return scale_notes


def degree_to_semitones(degree: int, mode: str) -> int:
    """
    Convert a scale degree (1-7) to semitones from root.

    Args:
        degree: Scale degree (1 = root, 2 = second, etc.)
        mode: Scale mode ("major" or "minor")

    Returns:
        Semitones from root

    Raises:
        ValueError: If degree is not 1-7

    Example:
        >>> degree_to_semitones(5, "major")  # Perfect fifth
        7
    """
    if not 1 <= degree <= 7:
        raise ValueError(f"Degree must be 1-7, got {degree}")

    intervals = get_scale_intervals(mode)
    return intervals[degree - 1]


# =============================================================================
# Chord Functions
# =============================================================================


def get_chord_intervals(chord_type: str) -> tuple[int, ...]:
    """
    Get the intervals (in semitones) for a chord type.

    Args:
        chord_type: Chord type (e.g., "major", "min7", "dom7")

    Returns:
        Tuple of semitone intervals from the root

    Raises:
        ValueError: If chord type is not supported

    Example:
        >>> get_chord_intervals("major")
        (0, 4, 7)
        >>> get_chord_intervals("min7")
        (0, 3, 7, 10)
    """
    chord_lower = chord_type.lower()
    if chord_lower not in CHORD_INTERVALS:
        raise ValueError(
            f"Unsupported chord type: {chord_type}. Use: {list(CHORD_INTERVALS.keys())}"
        )
    return CHORD_INTERVALS[chord_lower]


def build_chord(root: str, chord_type: str) -> list[str]:
    """
    Build a chord from a root note and chord type.

    Args:
        root: Root note of the chord (e.g., "C", "F#")
        chord_type: Chord type (e.g., "major", "min7")

    Returns:
        List of note names in the chord

    Example:
        >>> build_chord("C", "major")
        ['C', 'E', 'G']
        >>> build_chord("A", "min7")
        ['A', 'C', 'E', 'G']
    """
    intervals = get_chord_intervals(chord_type)
    root_pitch_class = get_pitch_class(root)

    # Use flats if the root is a flat note, otherwise use sharps
    root_upper = root.capitalize()
    prefer_sharps = "b" not in root_upper  # Use sharps unless root is a flat
    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES

    chord_notes: list[str] = []
    for interval in intervals:
        pitch_class = (root_pitch_class + interval) % 12
        chord_notes.append(names[pitch_class])

    return chord_notes


# =============================================================================
# Transposition Functions
# =============================================================================


def transpose_pitch_class(pitch_class: int, semitones: int) -> int:
    """
    Transpose a pitch class by a number of semitones.

    Args:
        pitch_class: Original pitch class (0-11)
        semitones: Number of semitones to transpose (can be negative)

    Returns:
        New pitch class (0-11)
    """
    return (pitch_class + semitones) % 12


def transpose_note(note: str, semitones: int, prefer_sharps: bool = True) -> str:
    """
    Transpose a note by a number of semitones.

    Args:
        note: Note name (e.g., "C", "F#")
        semitones: Number of semitones to transpose (can be negative)
        prefer_sharps: If True, use sharps; if False, use flats

    Returns:
        Transposed note name

    Example:
        >>> transpose_note("C", 7)  # Up a fifth
        'G'
        >>> transpose_note("G", -7)  # Down a fifth
        'C'
    """
    pitch_class = get_pitch_class(note)
    new_pitch_class = transpose_pitch_class(pitch_class, semitones)

    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES
    return names[new_pitch_class]


def transpose_midi(midi: int, semitones: int) -> int:
    """
    Transpose a MIDI note by a number of semitones.

    Args:
        midi: Original MIDI note number
        semitones: Number of semitones to transpose (can be negative)

    Returns:
        Transposed MIDI note number

    Raises:
        ValueError: If result is out of MIDI range (0-127)
    """
    result = midi + semitones
    if not 0 <= result <= 127:
        raise ValueError(f"Transposed MIDI {result} out of range (0-127)")
    return result

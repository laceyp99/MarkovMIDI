"""
Utilities module for MarkovMIDI.

Provides music theory utilities (scales, chords, intervals) and
quantization functions for MIDI timing.
"""

from markov_midi.utils.music_theory import (
    # Constants
    PITCH_CLASS_TO_NAMES,
    NAME_TO_PITCH_CLASS,
    ALL_NOTE_NAMES,
    SHARP_NAMES,
    FLAT_NAMES,
    SCALE_INTERVALS,
    CHORD_INTERVALS,
    # Note/MIDI conversion
    note_to_midi,
    midi_to_note,
    get_pitch_class,
    # Scale functions
    get_scale_intervals,
    get_scale_notes,
    degree_to_semitones,
    # Chord functions
    get_chord_intervals,
    build_chord,
    # Transposition
    transpose_pitch_class,
    transpose_note,
    transpose_midi,
)

from markov_midi.utils.quantize import (
    get_ticks_per_beat,
    get_grid_size,
    get_16th_grid,
    quantize_to_grid,
    quantize_duration,
    ticks_to_beats,
    beats_to_ticks,
    ticks_to_bars,
    bars_to_ticks,
    get_bar_length,
)

__all__: list[str] = [
    # Music theory constants
    "PITCH_CLASS_TO_NAMES",
    "NAME_TO_PITCH_CLASS",
    "ALL_NOTE_NAMES",
    "SHARP_NAMES",
    "FLAT_NAMES",
    "SCALE_INTERVALS",
    "CHORD_INTERVALS",
    # Note/MIDI conversion
    "note_to_midi",
    "midi_to_note",
    "get_pitch_class",
    # Scale functions
    "get_scale_intervals",
    "get_scale_notes",
    "degree_to_semitones",
    # Chord functions
    "get_chord_intervals",
    "build_chord",
    # Transposition
    "transpose_pitch_class",
    "transpose_note",
    "transpose_midi",
    # Quantization
    "get_ticks_per_beat",
    "get_grid_size",
    "get_16th_grid",
    "quantize_to_grid",
    "quantize_duration",
    "ticks_to_beats",
    "beats_to_ticks",
    "ticks_to_bars",
    "bars_to_ticks",
    "get_bar_length",
]

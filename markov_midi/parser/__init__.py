"""
Parser module for MarkovMIDI.

Provides MIDI file loading and parsing capabilities using the mido library.
"""

from markov_midi.parser.midi_loader import (
    ParsedNote,
    ParsedTrack,
    ParsedMidi,
    parse_midi_file,
    quantize_parsed_midi,
    extract_intervals,
    extract_durations_16ths,
    notes_to_training_data,
)

__all__: list[str] = [
    "ParsedNote",
    "ParsedTrack",
    "ParsedMidi",
    "parse_midi_file",
    "quantize_parsed_midi",
    "extract_intervals",
    "extract_durations_16ths",
    "notes_to_training_data",
]

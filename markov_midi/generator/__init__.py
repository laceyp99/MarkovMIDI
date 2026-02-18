"""
Generator module for MarkovMIDI.

Orchestrates the full loop generation process, including chord voicing
and MIDI file writing.
"""

from markov_midi.generator.voicing import (
    VoicingStyle,
    VoicedNote,
    VoicedChord,
    get_chord_quality,
    degree_to_midi_notes,
    voice_chord_block,
    voice_chord_arpeggio,
    voice_chord_sequence,
    get_all_voiced_notes,
    MAJOR_SCALE_CHORD_QUALITIES,
    MINOR_SCALE_CHORD_QUALITIES,
)

from markov_midi.generator.midi_writer import (
    MidiTrackData,
    MidiFileData,
    sixteenths_to_ticks,
    melody_sequence_to_voiced_notes,
    create_midi_track,
    create_midi_file,
    write_midi_file,
    write_simple_loop,
    get_midi_duration_seconds,
    DEFAULT_TICKS_PER_BEAT,
    DEFAULT_TEMPO_BPM,
    DEFAULT_VELOCITY,
)

from markov_midi.generator.loop_generator import (
    GenerationParams,
    GeneratedLoop,
    LoopGenerator,
)

__all__: list[str] = [
    # Voicing
    "VoicingStyle",
    "VoicedNote",
    "VoicedChord",
    "get_chord_quality",
    "degree_to_midi_notes",
    "voice_chord_block",
    "voice_chord_arpeggio",
    "voice_chord_sequence",
    "get_all_voiced_notes",
    "MAJOR_SCALE_CHORD_QUALITIES",
    "MINOR_SCALE_CHORD_QUALITIES",
    # MIDI Writer
    "MidiTrackData",
    "MidiFileData",
    "sixteenths_to_ticks",
    "melody_sequence_to_voiced_notes",
    "create_midi_track",
    "create_midi_file",
    "write_midi_file",
    "write_simple_loop",
    "get_midi_duration_seconds",
    "DEFAULT_TICKS_PER_BEAT",
    "DEFAULT_TEMPO_BPM",
    "DEFAULT_VELOCITY",
    # Loop Generator
    "GenerationParams",
    "GeneratedLoop",
    "LoopGenerator",
]

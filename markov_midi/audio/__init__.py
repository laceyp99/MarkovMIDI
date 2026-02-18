"""
Audio module for MarkovMIDI.

Provides audio synthesis capabilities using FluidSynth for previewing
generated MIDI loops before export.
"""

from markov_midi.audio.synthesizer import (
    SynthesizerConfig,
    Synthesizer,
    find_fluidsynth,
    is_fluidsynth_available,
    find_soundfonts,
    render_midi_to_wav,
    render_midi_to_temp_wav,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_GAIN,
)

__all__: list[str] = [
    "SynthesizerConfig",
    "Synthesizer",
    "find_fluidsynth",
    "is_fluidsynth_available",
    "find_soundfonts",
    "render_midi_to_wav",
    "render_midi_to_temp_wav",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_GAIN",
]

"""
Model module for MarkovMIDI.

Contains the core Markov chain implementation, music models for chords
and melodies, theory priors, reward system, and persistence utilities.
"""

from markov_midi.model.markov_chain import MarkovChain
from markov_midi.model.chord_model import ChordModel, ChordEvent, ChordSequence
from markov_midi.model.melody_model import MelodyModel, MelodyNote, MelodySequence

from markov_midi.model.theory_priors import (
    # Constants
    CHORD_DEGREES,
    CHORD_TRANSITION_WEIGHTS,
    CHORD_SECOND_ORDER_BOOSTS,
    CHORD_RHYTHM_DURATIONS,
    CHORD_RHYTHM_WEIGHTS,
    MELODY_INTERVALS,
    MELODY_INTERVAL_WEIGHTS,
    MELODY_RHYTHM_DURATIONS,
    MELODY_RHYTHM_WEIGHTS,
    # Factory functions
    create_chord_chain,
    create_chord_rhythm_chain,
    create_melody_pitch_chain,
    create_melody_rhythm_chain,
    create_all_chains,
)

__all__: list[str] = [
    # Core classes
    "MarkovChain",
    # Chord model
    "ChordModel",
    "ChordEvent",
    "ChordSequence",
    # Melody model
    "MelodyModel",
    "MelodyNote",
    "MelodySequence",
    # Constants
    "CHORD_DEGREES",
    "CHORD_TRANSITION_WEIGHTS",
    "CHORD_SECOND_ORDER_BOOSTS",
    "CHORD_RHYTHM_DURATIONS",
    "CHORD_RHYTHM_WEIGHTS",
    "MELODY_INTERVALS",
    "MELODY_INTERVAL_WEIGHTS",
    "MELODY_RHYTHM_DURATIONS",
    "MELODY_RHYTHM_WEIGHTS",
    # Factory functions
    "create_chord_chain",
    "create_chord_rhythm_chain",
    "create_melody_pitch_chain",
    "create_melody_rhythm_chain",
    "create_all_chains",
]

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

from markov_midi.model.reward import (
    RewardSensitivity,
    SENSITIVITY_MULTIPLIERS,
    Rating,
    GenerationRecord,
    RewardManager,
)

from markov_midi.model.persistence import (
    SessionMetadata,
    Session,
    save_session,
    load_session,
    create_session_from_generator,
    restore_session,
    list_sessions,
    save_model_only,
    load_model_only,
    get_session_path,
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
    # Theory priors - Constants
    "CHORD_DEGREES",
    "CHORD_TRANSITION_WEIGHTS",
    "CHORD_SECOND_ORDER_BOOSTS",
    "CHORD_RHYTHM_DURATIONS",
    "CHORD_RHYTHM_WEIGHTS",
    "MELODY_INTERVALS",
    "MELODY_INTERVAL_WEIGHTS",
    "MELODY_RHYTHM_DURATIONS",
    "MELODY_RHYTHM_WEIGHTS",
    # Theory priors - Factory functions
    "create_chord_chain",
    "create_chord_rhythm_chain",
    "create_melody_pitch_chain",
    "create_melody_rhythm_chain",
    "create_all_chains",
    # Reward system
    "RewardSensitivity",
    "SENSITIVITY_MULTIPLIERS",
    "Rating",
    "GenerationRecord",
    "RewardManager",
    # Persistence
    "SessionMetadata",
    "Session",
    "save_session",
    "load_session",
    "create_session_from_generator",
    "restore_session",
    "list_sessions",
    "save_model_only",
    "load_model_only",
    "get_session_path",
]

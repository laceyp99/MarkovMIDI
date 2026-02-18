"""
UI module for MarkovMIDI.

Provides the Gradio web interface for generating loops, training
models with rewards, and managing sessions.
"""

from markov_midi.ui.app import (
    AppState,
    create_fresh_state,
    create_ui,
    launch,
)
from markov_midi.ui.visualizer import (
    create_empty_figure,
    create_piano_roll_figure,
)

__all__: list[str] = [
    "AppState",
    "create_fresh_state",
    "create_ui",
    "launch",
    "create_empty_figure",
    "create_piano_roll_figure",
]

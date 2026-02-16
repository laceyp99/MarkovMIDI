#!/usr/bin/env python3
"""
MarkovMIDI - CLI Entry Point

Generate MIDI music loops using second-order Markov chains.

Usage:
    python main.py

This launches the Gradio web UI for generating and training MIDI loops.
"""

from markov_midi.__main__ import main


if __name__ == "__main__":
    main()

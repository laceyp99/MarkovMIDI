"""
MarkovMIDI package entry point.

Allows running the package directly with: python -m markov_midi
"""

from typing import NoReturn


def main() -> NoReturn | None:
    """
    Main entry point for the MarkovMIDI application.

    Launches the Gradio web UI for generating and training
    MIDI loops using Markov chains.
    """
    # TODO: Import and launch the Gradio UI once implemented
    print("MarkovMIDI v0.1.0")
    print("Web UI not yet implemented. Coming in Phase 7!")
    print("\nProject structure is ready. Next steps:")
    print("  - Phase 2: Implement utils/music_theory.py and utils/quantize.py")
    print("  - Phase 3: Implement model/markov_chain.py")
    return None


if __name__ == "__main__":
    main()

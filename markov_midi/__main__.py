"""
MarkovMIDI package entry point.

Allows running the package directly with: python -m markov_midi
"""

from typing import NoReturn
import argparse


def main() -> NoReturn | None:
    """
    Main entry point for the MarkovMIDI application.

    Launches the Gradio web UI for generating and training
    MIDI loops using Markov chains.
    """

    parser = argparse.ArgumentParser(
        prog="markov_midi",
        description="Generate MIDI music loops using Markov chains"
    )
    parser.add_argument(
        "--version", action="version", version="MarkovMIDI 0.0.1"
    )
    args = parser.parse_args()

    # TODO: Import and launch the Gradio UI once implemented
    print(f"MarkovMIDI version {args.version} launched")
    print("Web UI not yet implemented")
    return None


if __name__ == "__main__":
    main()

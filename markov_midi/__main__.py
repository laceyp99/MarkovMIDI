"""
MarkovMIDI package entry point.

Allows running the package directly with: python -m markov_midi
"""

import argparse


def main() -> None:
    """
    Main entry point for the MarkovMIDI application.

    Launches the Gradio web UI for generating and training
    MIDI loops using Markov chains.
    """
    parser = argparse.ArgumentParser(
        prog="markov_midi",
        description="Generate MIDI music loops using Markov chains",
    )
    parser.add_argument("--version", action="version", version="MarkovMIDI v0.1.0")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web UI on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args()

    print("MarkovMIDI v0.1.0")
    print("Starting web UI...")

    from markov_midi.ui import launch

    launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

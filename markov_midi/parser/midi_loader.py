"""
MIDI file loading and parsing for MarkovMIDI.

Loads MIDI files using mido and extracts note sequences that can be
used for training Markov chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mido

from markov_midi.utils.quantize import (
    get_grid_size,
    quantize_to_grid,
    ticks_to_beats,
)


@dataclass
class ParsedNote:
    """
    A parsed MIDI note with timing information.

    Attributes:
        midi: MIDI note number (0-127)
        start_tick: Start time in MIDI ticks
        duration_ticks: Duration in MIDI ticks
        velocity: MIDI velocity (0-127)
        channel: MIDI channel (0-15)
    """

    midi: int
    start_tick: int
    duration_ticks: int
    velocity: int = 100
    channel: int = 0


@dataclass
class ParsedTrack:
    """
    A parsed MIDI track.

    Attributes:
        name: Track name
        notes: List of parsed notes
        channel: Primary MIDI channel
        program: MIDI program number (instrument)
    """

    name: str
    notes: list[ParsedNote] = field(default_factory=list)
    channel: int = 0
    program: int = 0


@dataclass
class ParsedMidi:
    """
    A fully parsed MIDI file.

    Attributes:
        tracks: List of parsed tracks
        ticks_per_beat: MIDI resolution
        tempo_bpm: Tempo (from first tempo event, or default 120)
        time_signature: Time signature as (numerator, denominator)
        source_path: Original file path
    """

    tracks: list[ParsedTrack] = field(default_factory=list)
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    source_path: str = ""

    def get_all_notes(self) -> list[ParsedNote]:
        """Get all notes from all tracks, sorted by start time."""
        all_notes: list[ParsedNote] = []
        for track in self.tracks:
            all_notes.extend(track.notes)
        all_notes.sort(key=lambda n: (n.start_tick, n.midi))
        return all_notes

    def get_duration_ticks(self) -> int:
        """Get total duration in ticks."""
        max_end = 0
        for track in self.tracks:
            for note in track.notes:
                end = note.start_tick + note.duration_ticks
                if end > max_end:
                    max_end = end
        return max_end

    def get_duration_beats(self) -> float:
        """Get total duration in beats."""
        return ticks_to_beats(self.get_duration_ticks(), self.ticks_per_beat)


def parse_midi_file(file_path: str | Path) -> ParsedMidi:
    """
    Parse a MIDI file and extract note information.

    Args:
        file_path: Path to the MIDI file

    Returns:
        ParsedMidi with all extracted data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid MIDI file
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    try:
        midi_file = mido.MidiFile(str(path))
    except Exception as e:
        raise ValueError(f"Invalid MIDI file: {path}") from e

    parsed = ParsedMidi(
        ticks_per_beat=midi_file.ticks_per_beat,
        source_path=str(path),
    )

    # Extract tempo and time signature from first track (usually meta track)
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                parsed.tempo_bpm = mido.tempo2bpm(msg.tempo)
                break
            if msg.type == "time_signature":
                parsed.time_signature = (msg.numerator, msg.denominator)

    # Parse each track for notes
    for track in midi_file.tracks:
        parsed_track = _parse_track(track)
        if parsed_track.notes:  # Only add tracks with notes
            parsed.tracks.append(parsed_track)

    return parsed


def _parse_track(track: mido.MidiTrack) -> ParsedTrack:
    """
    Parse a single MIDI track.

    Args:
        track: mido MidiTrack to parse

    Returns:
        ParsedTrack with extracted notes
    """
    parsed_track = ParsedTrack(name="")

    # Track active notes (note_on without corresponding note_off)
    active_notes: dict[
        tuple[int, int], tuple[int, int]
    ] = {}  # (note, channel) -> (start_tick, velocity)

    current_tick = 0
    current_program = 0

    for msg in track:
        current_tick += msg.time

        if msg.type == "track_name":
            parsed_track.name = msg.name

        elif msg.type == "program_change":
            current_program = msg.program
            parsed_track.program = current_program
            parsed_track.channel = msg.channel

        elif msg.type == "note_on" and msg.velocity > 0:
            # Note on
            key = (msg.note, msg.channel)
            active_notes[key] = (current_tick, msg.velocity)

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            # Note off
            key = (msg.note, msg.channel)
            if key in active_notes:
                start_tick, velocity = active_notes.pop(key)
                duration = current_tick - start_tick
                if duration > 0:
                    parsed_track.notes.append(
                        ParsedNote(
                            midi=msg.note,
                            start_tick=start_tick,
                            duration_ticks=duration,
                            velocity=velocity,
                            channel=msg.channel,
                        )
                    )

    # Sort notes by start time
    parsed_track.notes.sort(key=lambda n: (n.start_tick, n.midi))

    return parsed_track


def quantize_parsed_midi(
    parsed: ParsedMidi,
    grid_subdivision: int = 16,
) -> ParsedMidi:
    """
    Quantize all notes in a parsed MIDI to a grid.

    Args:
        parsed: ParsedMidi to quantize
        grid_subdivision: Grid subdivision (16 = 16th notes, 8 = 8th notes)

    Returns:
        New ParsedMidi with quantized notes
    """
    grid_size = get_grid_size(parsed.ticks_per_beat, grid_subdivision)

    quantized = ParsedMidi(
        ticks_per_beat=parsed.ticks_per_beat,
        tempo_bpm=parsed.tempo_bpm,
        time_signature=parsed.time_signature,
        source_path=parsed.source_path,
    )

    for track in parsed.tracks:
        quantized_track = ParsedTrack(
            name=track.name,
            channel=track.channel,
            program=track.program,
        )

        for note in track.notes:
            q_start = quantize_to_grid(note.start_tick, grid_size)
            q_duration = quantize_to_grid(note.duration_ticks, grid_size)
            # Ensure minimum duration of one grid unit
            if q_duration < grid_size:
                q_duration = grid_size

            quantized_track.notes.append(
                ParsedNote(
                    midi=note.midi,
                    start_tick=q_start,
                    duration_ticks=q_duration,
                    velocity=note.velocity,
                    channel=note.channel,
                )
            )

        if quantized_track.notes:
            quantized.tracks.append(quantized_track)

    return quantized


def extract_intervals(notes: list[ParsedNote]) -> list[int]:
    """
    Extract melodic intervals from a sequence of notes.

    Args:
        notes: List of notes sorted by start time

    Returns:
        List of intervals in semitones between consecutive notes
    """
    if len(notes) < 2:
        return []

    intervals: list[int] = []
    for i in range(1, len(notes)):
        interval = notes[i].midi - notes[i - 1].midi
        intervals.append(interval)

    return intervals


def extract_durations_16ths(
    notes: list[ParsedNote],
    ticks_per_beat: int,
) -> list[int]:
    """
    Extract note durations in 16th note units.

    Args:
        notes: List of parsed notes
        ticks_per_beat: MIDI resolution

    Returns:
        List of durations in 16th notes
    """
    grid_size = get_grid_size(ticks_per_beat, 16)
    durations: list[int] = []

    for note in notes:
        dur_16ths = max(1, round(note.duration_ticks / grid_size))
        durations.append(dur_16ths)

    return durations


def notes_to_training_data(
    notes: list[ParsedNote],
    ticks_per_beat: int,
) -> dict[str, list[int]]:
    """
    Convert parsed notes to training data for Markov chains.

    Args:
        notes: List of parsed notes
        ticks_per_beat: MIDI resolution

    Returns:
        Dictionary with 'intervals' and 'durations' lists
    """
    return {
        "intervals": extract_intervals(notes),
        "durations": extract_durations_16ths(notes, ticks_per_beat),
    }

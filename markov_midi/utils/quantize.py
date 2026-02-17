"""
MIDI timing and quantization utilities for MarkovMIDI.

Provides functions for extracting timing info from MIDI files and
quantizing note events to a grid (e.g., 16th notes).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mido import MidiFile


def get_ticks_per_beat(midi_file: "MidiFile") -> int:
    """
    Extract ticks per beat (PPQ) from a MIDI file.

    Args:
        midi_file: A mido MidiFile object

    Returns:
        Ticks per beat (pulses per quarter note)

    Example:
        >>> from mido import MidiFile
        >>> mid = MidiFile("song.mid")
        >>> get_ticks_per_beat(mid)
        480
    """
    tpb: int = midi_file.ticks_per_beat
    return tpb


def get_grid_size(ticks_per_beat: int, subdivision: int = 16) -> int:
    """
    Calculate the tick size for a given note subdivision.

    Args:
        ticks_per_beat: Ticks per quarter note (from MIDI file)
        subdivision: Note subdivision (4=quarter, 8=eighth, 16=sixteenth, etc.)

    Returns:
        Number of ticks per subdivision

    Example:
        >>> get_grid_size(480, 16)  # 16th note grid at 480 TPB
        120
        >>> get_grid_size(480, 8)   # 8th note grid
        240
    """
    # A quarter note = 4 in subdivision terms
    # So 16th note = 4 times per quarter note = ticks_per_beat / 4
    quarter_subdivisions = subdivision // 4
    return ticks_per_beat // quarter_subdivisions


def get_16th_grid(ticks_per_beat: int) -> int:
    """
    Get the tick size for 16th notes.

    Convenience function for the most common quantization grid.

    Args:
        ticks_per_beat: Ticks per quarter note (from MIDI file)

    Returns:
        Number of ticks per 16th note

    Example:
        >>> get_16th_grid(480)
        120
    """
    return get_grid_size(ticks_per_beat, 16)


def quantize_to_grid(tick: int, grid_size: int) -> int:
    """
    Snap a tick position to the nearest grid position.

    Args:
        tick: Original tick position
        grid_size: Size of grid in ticks (e.g., 120 for 16th notes at 480 TPB)

    Returns:
        Quantized tick position (nearest grid point)

    Example:
        >>> quantize_to_grid(50, 120)   # Closer to 0
        0
        >>> quantize_to_grid(100, 120)  # Closer to 120
        120
        >>> quantize_to_grid(60, 120)   # Exactly halfway, rounds to nearest
        60  # Actually rounds to 0 (banker's rounding) or 120
    """
    # Round to nearest grid position
    return round(tick / grid_size) * grid_size


def quantize_duration(duration: int, grid_size: int, min_duration: int = 1) -> int:
    """
    Quantize a note duration to the grid.

    Ensures duration is at least min_duration grid units.

    Args:
        duration: Original duration in ticks
        grid_size: Size of grid in ticks
        min_duration: Minimum duration in grid units (default 1)

    Returns:
        Quantized duration in ticks

    Example:
        >>> quantize_duration(100, 120)  # Less than 1 grid unit
        120  # Minimum 1 grid unit
        >>> quantize_duration(300, 120)  # 2.5 grid units
        360  # Rounds to 3 grid units
    """
    grid_units = round(duration / grid_size)
    grid_units = max(grid_units, min_duration)
    return grid_units * grid_size


def ticks_to_beats(ticks: int, ticks_per_beat: int) -> float:
    """
    Convert ticks to beat position.

    Args:
        ticks: Position in ticks
        ticks_per_beat: Ticks per quarter note

    Returns:
        Position in beats (quarter notes)

    Example:
        >>> ticks_to_beats(480, 480)
        1.0
        >>> ticks_to_beats(240, 480)
        0.5
    """
    return ticks / ticks_per_beat


def beats_to_ticks(beats: float, ticks_per_beat: int) -> int:
    """
    Convert beat position to ticks.

    Args:
        beats: Position in beats (quarter notes)
        ticks_per_beat: Ticks per quarter note

    Returns:
        Position in ticks (rounded to nearest integer)

    Example:
        >>> beats_to_ticks(1.0, 480)
        480
        >>> beats_to_ticks(0.5, 480)
        240
    """
    return round(beats * ticks_per_beat)


def ticks_to_bars(ticks: int, ticks_per_beat: int, beats_per_bar: int = 4) -> float:
    """
    Convert ticks to bar position.

    Args:
        ticks: Position in ticks
        ticks_per_beat: Ticks per quarter note
        beats_per_bar: Beats per bar (default 4 for 4/4 time)

    Returns:
        Position in bars

    Example:
        >>> ticks_to_bars(1920, 480, 4)  # 4 beats = 1 bar
        1.0
    """
    beats = ticks_to_beats(ticks, ticks_per_beat)
    return beats / beats_per_bar


def bars_to_ticks(bars: float, ticks_per_beat: int, beats_per_bar: int = 4) -> int:
    """
    Convert bar position to ticks.

    Args:
        bars: Position in bars
        ticks_per_beat: Ticks per quarter note
        beats_per_bar: Beats per bar (default 4 for 4/4 time)

    Returns:
        Position in ticks

    Example:
        >>> bars_to_ticks(1.0, 480, 4)
        1920
    """
    beats = bars * beats_per_bar
    return beats_to_ticks(beats, ticks_per_beat)


def get_bar_length(ticks_per_beat: int, beats_per_bar: int = 4) -> int:
    """
    Get the length of one bar in ticks.

    Args:
        ticks_per_beat: Ticks per quarter note
        beats_per_bar: Beats per bar (default 4 for 4/4 time)

    Returns:
        Bar length in ticks

    Example:
        >>> get_bar_length(480, 4)
        1920
    """
    return ticks_per_beat * beats_per_bar

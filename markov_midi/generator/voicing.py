"""
Chord voicing utilities for MarkovMIDI.

Converts chord degrees and sequences into MIDI note numbers with
proper voicings (block chords, arpeggios) in specified octaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from markov_midi.model.chord_model import ChordEvent, ChordSequence
from markov_midi.utils.music_theory import (
    degree_to_semitones,
    get_pitch_class,
    get_scale_notes,
    CHORD_INTERVALS,
)


class VoicingStyle(Enum):
    """Available chord voicing styles."""

    BLOCK = "block"  # All notes played simultaneously
    ARPEGGIO_UP = "arpeggio_up"  # Notes played ascending
    ARPEGGIO_DOWN = "arpeggio_down"  # Notes played descending
    ARPEGGIO_UP_DOWN = "arpeggio_up_down"  # Ascending then descending


# Chord quality for each scale degree (1-7) in major and minor
# Maps degree -> chord type
MAJOR_SCALE_CHORD_QUALITIES: Final[dict[int, str]] = {
    1: "major",  # I - major
    2: "minor",  # ii - minor
    3: "minor",  # iii - minor
    4: "major",  # IV - major
    5: "major",  # V - major (or dom7)
    6: "minor",  # vi - minor
    7: "diminished",  # vii° - diminished
}

MINOR_SCALE_CHORD_QUALITIES: Final[dict[int, str]] = {
    1: "minor",  # i - minor
    2: "diminished",  # ii° - diminished
    3: "major",  # III - major
    4: "minor",  # iv - minor
    5: "minor",  # v - minor (or major in harmonic minor)
    6: "major",  # VI - major
    7: "major",  # VII - major (subtonic, not leading tone)
}


@dataclass
class VoicedNote:
    """
    A single MIDI note with timing information.

    Attributes:
        midi: MIDI note number (0-127)
        start_time: Start time in 16th notes
        duration: Duration in 16th notes
        velocity: MIDI velocity (0-127)
    """

    midi: int
    start_time: int
    duration: int
    velocity: int = 100


@dataclass
class VoicedChord:
    """
    A chord voiced as MIDI notes.

    Attributes:
        notes: List of VoicedNote objects
        degree: Original scale degree (1-7)
        chord_type: Chord quality (major, minor, etc.)
    """

    notes: list[VoicedNote]
    degree: int
    chord_type: str


def get_chord_quality(degree: int, mode: str) -> str:
    """
    Get the chord quality for a scale degree in a given mode.

    Args:
        degree: Scale degree (1-7)
        mode: Scale mode ("major" or "minor")

    Returns:
        Chord type string (e.g., "major", "minor", "diminished")

    Raises:
        ValueError: If degree is not 1-7 or mode is unsupported
    """
    if not 1 <= degree <= 7:
        raise ValueError(f"Degree must be 1-7, got {degree}")

    mode_lower = mode.lower()
    if mode_lower == "major":
        return MAJOR_SCALE_CHORD_QUALITIES[degree]
    elif mode_lower == "minor":
        return MINOR_SCALE_CHORD_QUALITIES[degree]
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'major' or 'minor'.")


def degree_to_midi_notes(
    degree: int,
    root: str,
    mode: str,
    octave: int = 4,
    use_seventh: bool = False,
) -> list[int]:
    """
    Convert a scale degree to MIDI note numbers for a chord.

    Args:
        degree: Scale degree (1-7)
        root: Root note of the key (e.g., "C", "F#")
        mode: Scale mode ("major" or "minor")
        octave: Base octave for the chord (default 4)
        use_seventh: If True, use 7th chords instead of triads

    Returns:
        List of MIDI note numbers for the chord

    Example:
        >>> degree_to_midi_notes(1, "C", "major", octave=4)
        [60, 64, 67]  # C major triad: C4, E4, G4
    """
    if not 1 <= degree <= 7:
        raise ValueError(f"Degree must be 1-7, got {degree}")

    # Get the scale notes
    scale_notes = get_scale_notes(root, mode)

    # Get the root of this chord (the scale degree)
    chord_root = scale_notes[degree - 1]
    chord_root_pitch_class = get_pitch_class(chord_root)

    # Get the chord quality
    quality = get_chord_quality(degree, mode)

    # Determine chord type (triad or 7th)
    if use_seventh:
        # Map triad qualities to 7th chord types
        seventh_map = {
            "major": "maj7",
            "minor": "min7",
            "diminished": "dim7",
            "augmented": "maj7",  # Rare, use maj7
        }
        # Special case: V chord typically uses dom7
        if degree == 5:
            chord_type = "dom7"
        else:
            chord_type = seventh_map.get(quality, "maj7")
    else:
        chord_type = quality

    # Get chord intervals
    intervals = CHORD_INTERVALS[chord_type]

    # Calculate MIDI notes
    base_midi = (octave + 1) * 12 + chord_root_pitch_class
    midi_notes = [base_midi + interval for interval in intervals]

    # Clamp to valid MIDI range
    midi_notes = [max(0, min(127, n)) for n in midi_notes]

    return midi_notes


def voice_chord_block(
    event: ChordEvent,
    root: str,
    mode: str,
    octave: int = 3,
    use_seventh: bool = False,
    velocity: int = 80,
) -> VoicedChord:
    """
    Voice a chord event as a block chord (all notes at once).

    Args:
        event: ChordEvent with degree, duration, start_time
        root: Root note of the key
        mode: Scale mode
        octave: Base octave for voicing
        use_seventh: Use 7th chords
        velocity: MIDI velocity

    Returns:
        VoicedChord with all notes having same start time
    """
    midi_notes = degree_to_midi_notes(event.degree, root, mode, octave, use_seventh)
    quality = get_chord_quality(event.degree, mode)

    voiced_notes = [
        VoicedNote(
            midi=midi,
            start_time=event.start_time,
            duration=event.duration,
            velocity=velocity,
        )
        for midi in midi_notes
    ]

    return VoicedChord(
        notes=voiced_notes,
        degree=event.degree,
        chord_type=quality,
    )


def voice_chord_arpeggio(
    event: ChordEvent,
    root: str,
    mode: str,
    octave: int = 3,
    use_seventh: bool = False,
    velocity: int = 80,
    direction: str = "up",
) -> VoicedChord:
    """
    Voice a chord event as an arpeggio.

    Args:
        event: ChordEvent with degree, duration, start_time
        root: Root note of the key
        mode: Scale mode
        octave: Base octave for voicing
        use_seventh: Use 7th chords
        velocity: MIDI velocity
        direction: "up", "down", or "up_down"

    Returns:
        VoicedChord with notes spread across the duration
    """
    midi_notes = degree_to_midi_notes(event.degree, root, mode, octave, use_seventh)
    quality = get_chord_quality(event.degree, mode)

    num_notes = len(midi_notes)
    if num_notes == 0:
        return VoicedChord(notes=[], degree=event.degree, chord_type=quality)

    # Determine note order based on direction
    if direction == "down":
        midi_notes = list(reversed(midi_notes))
    elif direction == "up_down":
        # Up then down (without repeating top/bottom)
        if num_notes > 2:
            midi_notes = midi_notes + list(reversed(midi_notes[1:-1]))

    # Calculate timing for each arpeggio note
    num_arp_notes = len(midi_notes)

    # Each note gets a portion of the total duration
    # Minimum 1 sixteenth per note
    note_duration = max(1, event.duration // num_arp_notes)

    voiced_notes: list[VoicedNote] = []
    current_time = event.start_time

    for i, midi in enumerate(midi_notes):
        # Last note gets remaining duration
        if i == num_arp_notes - 1:
            dur = event.start_time + event.duration - current_time
        else:
            dur = note_duration

        voiced_notes.append(
            VoicedNote(
                midi=midi,
                start_time=current_time,
                duration=max(1, dur),
                velocity=velocity,
            )
        )
        current_time += note_duration

    return VoicedChord(
        notes=voiced_notes,
        degree=event.degree,
        chord_type=quality,
    )


def voice_chord_sequence(
    sequence: ChordSequence,
    root: str,
    mode: str,
    octave: int = 3,
    style: VoicingStyle = VoicingStyle.BLOCK,
    use_seventh: bool = False,
    velocity: int = 80,
) -> list[VoicedChord]:
    """
    Voice an entire chord sequence.

    Args:
        sequence: ChordSequence to voice
        root: Root note of the key
        mode: Scale mode
        octave: Base octave for voicing
        style: VoicingStyle (BLOCK, ARPEGGIO_UP, etc.)
        use_seventh: Use 7th chords
        velocity: MIDI velocity

    Returns:
        List of VoicedChord objects
    """
    voiced_chords: list[VoicedChord] = []

    for event in sequence.events:
        if style == VoicingStyle.BLOCK:
            voiced = voice_chord_block(event, root, mode, octave, use_seventh, velocity)
        elif style == VoicingStyle.ARPEGGIO_UP:
            voiced = voice_chord_arpeggio(
                event, root, mode, octave, use_seventh, velocity, "up"
            )
        elif style == VoicingStyle.ARPEGGIO_DOWN:
            voiced = voice_chord_arpeggio(
                event, root, mode, octave, use_seventh, velocity, "down"
            )
        elif style == VoicingStyle.ARPEGGIO_UP_DOWN:
            voiced = voice_chord_arpeggio(
                event, root, mode, octave, use_seventh, velocity, "up_down"
            )
        else:
            # Default to block
            voiced = voice_chord_block(event, root, mode, octave, use_seventh, velocity)

        voiced_chords.append(voiced)

    return voiced_chords


def get_all_voiced_notes(voiced_chords: list[VoicedChord]) -> list[VoicedNote]:
    """
    Flatten a list of voiced chords into a single list of notes.

    Args:
        voiced_chords: List of VoicedChord objects

    Returns:
        List of all VoicedNote objects, sorted by start time
    """
    all_notes: list[VoicedNote] = []
    for chord in voiced_chords:
        all_notes.extend(chord.notes)

    # Sort by start time, then by MIDI note number
    all_notes.sort(key=lambda n: (n.start_time, n.midi))

    return all_notes

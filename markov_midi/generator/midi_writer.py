"""
MIDI file writing utilities for MarkovMIDI.

Uses mido to create MIDI files from generated chord and melody sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mido

from markov_midi.generator.voicing import VoicedNote
from markov_midi.model.melody_model import MelodySequence

if TYPE_CHECKING:
    pass


# Default MIDI settings
DEFAULT_TICKS_PER_BEAT: int = 480
DEFAULT_TEMPO_BPM: int = 120
DEFAULT_VELOCITY: int = 100


@dataclass
class MidiTrackData:
    """
    Data for a single MIDI track.

    Attributes:
        name: Track name
        notes: List of VoicedNote objects
        channel: MIDI channel (0-15)
        program: MIDI program/instrument number (0-127)
    """

    name: str
    notes: list[VoicedNote] = field(default_factory=list)
    channel: int = 0
    program: int = 0  # 0 = Acoustic Grand Piano


@dataclass
class MidiFileData:
    """
    Complete MIDI file data.

    Attributes:
        tracks: List of track data
        ticks_per_beat: MIDI resolution
        tempo_bpm: Tempo in beats per minute
    """

    tracks: list[MidiTrackData] = field(default_factory=list)
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
    tempo_bpm: int = DEFAULT_TEMPO_BPM


def sixteenths_to_ticks(sixteenths: int, ticks_per_beat: int) -> int:
    """
    Convert 16th note units to MIDI ticks.

    Args:
        sixteenths: Duration in 16th notes
        ticks_per_beat: MIDI resolution (ticks per quarter note)

    Returns:
        Duration in MIDI ticks
    """
    # 4 sixteenths per beat (quarter note)
    return sixteenths * (ticks_per_beat // 4)


def melody_sequence_to_voiced_notes(
    sequence: MelodySequence,
    start_midi: int = 60,
    velocity: int = DEFAULT_VELOCITY,
) -> list[VoicedNote]:
    """
    Convert a MelodySequence to VoicedNote objects.

    Args:
        sequence: MelodySequence with relative intervals
        start_midi: Starting MIDI note number
        velocity: MIDI velocity for all notes

    Returns:
        List of VoicedNote objects with absolute MIDI numbers
    """
    pitches = sequence.to_absolute_pitches(start_midi)
    voiced_notes: list[VoicedNote] = []

    for note, pitch in zip(sequence.notes, pitches):
        voiced_notes.append(
            VoicedNote(
                midi=pitch,
                start_time=note.start_time,
                duration=note.duration,
                velocity=velocity,
            )
        )

    return voiced_notes


def create_midi_track(
    track_data: MidiTrackData,
    ticks_per_beat: int,
) -> mido.MidiTrack:
    """
    Create a mido MidiTrack from track data.

    Args:
        track_data: Track data with notes
        ticks_per_beat: MIDI resolution

    Returns:
        Configured mido.MidiTrack
    """
    track = mido.MidiTrack()

    # Set track name
    track.append(mido.MetaMessage("track_name", name=track_data.name))

    # Set instrument (program change)
    track.append(
        mido.Message(
            "program_change",
            channel=track_data.channel,
            program=track_data.program,
        )
    )

    # Convert notes to MIDI events
    # We need to create note_on and note_off events and sort by absolute time
    events: list[tuple[int, str, int, int]] = []  # (tick, type, note, velocity)

    for note in track_data.notes:
        start_tick = sixteenths_to_ticks(note.start_time, ticks_per_beat)
        end_tick = sixteenths_to_ticks(note.start_time + note.duration, ticks_per_beat)

        events.append((start_tick, "note_on", note.midi, note.velocity))
        events.append((end_tick, "note_off", note.midi, 0))

    # Sort events by tick time, with note_off before note_on at same time
    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    # Convert to delta times and add to track
    current_tick = 0
    for tick, event_type, midi_note, velocity in events:
        delta = tick - current_tick
        track.append(
            mido.Message(
                event_type,
                note=midi_note,
                velocity=velocity,
                time=delta,
                channel=track_data.channel,
            )
        )
        current_tick = tick

    # End of track
    track.append(mido.MetaMessage("end_of_track"))

    return track


def create_midi_file(
    file_data: MidiFileData,
) -> mido.MidiFile:
    """
    Create a mido MidiFile from file data.

    Args:
        file_data: Complete MIDI file data

    Returns:
        Configured mido.MidiFile ready for saving
    """
    midi_file = mido.MidiFile(ticks_per_beat=file_data.ticks_per_beat)

    # Create tempo track (track 0)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="Tempo"))
    tempo_track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(file_data.tempo_bpm))
    )
    tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    tempo_track.append(mido.MetaMessage("end_of_track"))
    midi_file.tracks.append(tempo_track)

    # Create instrument tracks
    for track_data in file_data.tracks:
        track = create_midi_track(track_data, file_data.ticks_per_beat)
        midi_file.tracks.append(track)

    return midi_file


def write_midi_file(
    file_data: MidiFileData,
    output_path: str | Path,
) -> Path:
    """
    Write MIDI file data to disk.

    Args:
        file_data: Complete MIDI file data
        output_path: Output file path

    Returns:
        Path to the written file
    """
    midi_file = create_midi_file(file_data)
    path = Path(output_path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    midi_file.save(str(path))
    return path


def write_simple_loop(
    chord_notes: list[VoicedNote],
    melody_notes: list[VoicedNote],
    output_path: str | Path,
    tempo_bpm: int = DEFAULT_TEMPO_BPM,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    chord_program: int = 0,  # Piano
    melody_program: int = 0,  # Piano
) -> Path:
    """
    Write a simple two-track loop (chords + melody).

    Args:
        chord_notes: Voiced chord notes
        melody_notes: Melody notes
        output_path: Output file path
        tempo_bpm: Tempo
        ticks_per_beat: MIDI resolution
        chord_program: MIDI program for chords
        melody_program: MIDI program for melody

    Returns:
        Path to the written file
    """
    file_data = MidiFileData(
        tracks=[
            MidiTrackData(
                name="Chords",
                notes=chord_notes,
                channel=0,
                program=chord_program,
            ),
            MidiTrackData(
                name="Melody",
                notes=melody_notes,
                channel=1,
                program=melody_program,
            ),
        ],
        ticks_per_beat=ticks_per_beat,
        tempo_bpm=tempo_bpm,
    )

    return write_midi_file(file_data, output_path)


def get_midi_duration_seconds(
    total_sixteenths: int,
    tempo_bpm: int,
) -> float:
    """
    Calculate the duration of a loop in seconds.

    Args:
        total_sixteenths: Total duration in 16th notes
        tempo_bpm: Tempo in beats per minute

    Returns:
        Duration in seconds
    """
    # 4 sixteenths per beat
    beats = total_sixteenths / 4.0
    seconds_per_beat = 60.0 / tempo_bpm
    return beats * seconds_per_beat

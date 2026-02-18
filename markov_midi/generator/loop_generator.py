"""
Loop generator for MarkovMIDI.

Orchestrates the full generation process, combining chord and melody models
with voicing and MIDI output to create complete musical loops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markov_midi.model.chord_model import ChordModel, ChordSequence
from markov_midi.model.melody_model import MelodyModel, MelodySequence
from markov_midi.generator.voicing import (
    VoicingStyle,
    VoicedNote,
    voice_chord_sequence,
    get_all_voiced_notes,
)
from markov_midi.generator.midi_writer import (
    MidiFileData,
    MidiTrackData,
    write_midi_file,
    melody_sequence_to_voiced_notes,
    get_midi_duration_seconds,
)


@dataclass
class GenerationParams:
    """
    Parameters for loop generation.

    Attributes:
        key: Root note of the key (e.g., "C", "F#")
        mode: Scale mode ("major" or "minor")
        num_bars: Number of bars (4 or 8)
        beats_per_bar: Beats per bar (default 4 for 4/4 time)
        tempo_bpm: Tempo in beats per minute
        chord_octave: Octave for chord voicing
        melody_octave: Octave for melody (as starting MIDI note)
        voicing_style: Chord voicing style
        use_seventh_chords: Use 7th chords instead of triads
        chord_velocity: MIDI velocity for chords
        melody_velocity: MIDI velocity for melody
        note_density: Melody note density (0.0-1.0)
        end_on_tonic: End chord progression on tonic
    """

    key: str = "C"
    mode: str = "major"
    num_bars: int = 4
    beats_per_bar: int = 4
    tempo_bpm: int = 120
    chord_octave: int = 3
    melody_octave: int = 5
    voicing_style: VoicingStyle = VoicingStyle.BLOCK
    use_seventh_chords: bool = False
    chord_velocity: int = 70
    melody_velocity: int = 90
    note_density: float = 0.75
    end_on_tonic: bool = True


@dataclass
class GeneratedLoop:
    """
    A complete generated loop with all data.

    Attributes:
        params: Generation parameters used
        chord_sequence: Raw chord sequence (degrees + rhythms)
        melody_sequence: Raw melody sequence (intervals + rhythms)
        chord_notes: Voiced chord notes
        melody_notes: Voiced melody notes
        chord_transitions: Transitions used in chord generation
        melody_transitions: Transitions used in melody generation
        duration_seconds: Total duration in seconds
    """

    params: GenerationParams
    chord_sequence: ChordSequence
    melody_sequence: MelodySequence
    chord_notes: list[VoicedNote] = field(default_factory=list)
    melody_notes: list[VoicedNote] = field(default_factory=list)
    chord_transitions: list[tuple[tuple[int, ...], int]] = field(default_factory=list)
    melody_transitions: list[tuple[tuple[int, ...], int]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def get_all_transitions(self) -> list[tuple[tuple[int, ...], int]]:
        """Get all transitions from both chord and melody generation."""
        return self.chord_transitions + self.melody_transitions


class LoopGenerator:
    """
    Generator for complete musical loops.

    Combines ChordModel and MelodyModel to generate loops with
    chord progressions and melodic toplines.

    Example:
        >>> generator = LoopGenerator()
        >>> params = GenerationParams(key="C", mode="major", num_bars=4)
        >>> loop = generator.generate(params)
        >>> generator.save_midi(loop, "output/my_loop.mid")
    """

    def __init__(
        self,
        chord_model: ChordModel | None = None,
        melody_model: MelodyModel | None = None,
    ) -> None:
        """
        Initialize the loop generator.

        Args:
            chord_model: Optional pre-configured chord model
            melody_model: Optional pre-configured melody model
        """
        self.chord_model = chord_model or ChordModel()
        self.melody_model = melody_model or MelodyModel()

    def generate(
        self,
        params: GenerationParams | None = None,
        seed: int | None = None,
    ) -> GeneratedLoop:
        """
        Generate a complete loop.

        Args:
            params: Generation parameters (uses defaults if None)
            seed: Random seed for reproducibility

        Returns:
            GeneratedLoop with all generated data
        """
        params = params or GenerationParams()
        rng = random.Random(seed) if seed is not None else None

        # Generate chord progression
        chord_sequence = self.chord_model.generate(
            num_bars=params.num_bars,
            beats_per_bar=params.beats_per_bar,
            start_degree=1,  # Start on tonic
            end_on_tonic=params.end_on_tonic,
            rng=rng,
        )

        # Generate melody
        melody_sequence = self.melody_model.generate(
            num_bars=params.num_bars,
            beats_per_bar=params.beats_per_bar,
            note_density=params.note_density,
            rng=rng,
        )

        # Voice the chords
        voiced_chords = voice_chord_sequence(
            sequence=chord_sequence,
            root=params.key,
            mode=params.mode,
            octave=params.chord_octave,
            style=params.voicing_style,
            use_seventh=params.use_seventh_chords,
            velocity=params.chord_velocity,
        )
        chord_notes = get_all_voiced_notes(voiced_chords)

        # Convert melody to voiced notes
        # Calculate starting MIDI note for melody (root in specified octave)
        from markov_midi.utils.music_theory import note_to_midi

        melody_start_midi = note_to_midi(params.key, params.melody_octave)
        melody_notes = melody_sequence_to_voiced_notes(
            sequence=melody_sequence,
            start_midi=melody_start_midi,
            velocity=params.melody_velocity,
        )

        # Calculate duration
        total_sixteenths = params.num_bars * params.beats_per_bar * 4
        duration_seconds = get_midi_duration_seconds(total_sixteenths, params.tempo_bpm)

        return GeneratedLoop(
            params=params,
            chord_sequence=chord_sequence,
            melody_sequence=melody_sequence,
            chord_notes=chord_notes,
            melody_notes=melody_notes,
            chord_transitions=chord_sequence.transitions_used,
            melody_transitions=melody_sequence.transitions_used,
            duration_seconds=duration_seconds,
        )

    def save_midi(
        self,
        loop: GeneratedLoop,
        output_path: str | Path,
        ticks_per_beat: int = 480,
        chord_program: int = 0,  # Acoustic Grand Piano
        melody_program: int = 0,  # Acoustic Grand Piano
    ) -> Path:
        """
        Save a generated loop to a MIDI file.

        Args:
            loop: Generated loop to save
            output_path: Output file path
            ticks_per_beat: MIDI resolution
            chord_program: MIDI program for chord track
            melody_program: MIDI program for melody track

        Returns:
            Path to the saved file
        """
        file_data = MidiFileData(
            tracks=[
                MidiTrackData(
                    name="Chords",
                    notes=loop.chord_notes,
                    channel=0,
                    program=chord_program,
                ),
                MidiTrackData(
                    name="Melody",
                    notes=loop.melody_notes,
                    channel=1,
                    program=melody_program,
                ),
            ],
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=loop.params.tempo_bpm,
        )

        return write_midi_file(file_data, output_path)

    def apply_reward(
        self,
        loop: GeneratedLoop,
        chord_reward: float = 0.0,
        melody_reward: float = 0.0,
        overall_reward: float = 0.0,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to the models based on a generated loop.

        Args:
            loop: Generated loop with transition tracking
            chord_reward: Reward for chord progression (can be negative)
            melody_reward: Reward for melody (can be negative)
            overall_reward: Reward applied to both models
            sensitivity: Multiplier for reward magnitude
        """
        # Apply chord-specific + overall reward
        total_chord_reward = chord_reward + overall_reward
        if total_chord_reward != 0.0:
            self.chord_model.apply_reward(
                loop.chord_transitions,
                total_chord_reward,
                sensitivity,
            )

        # Apply melody-specific + overall reward
        total_melody_reward = melody_reward + overall_reward
        if total_melody_reward != 0.0:
            self.melody_model.apply_reward(
                loop.melody_transitions,
                total_melody_reward,
                sensitivity,
            )

    def reset_models(self) -> None:
        """Reset both models to their theory priors."""
        self.chord_model.reset_to_priors()
        self.melody_model.reset_to_priors()

    def to_dict(self) -> dict[str, Any]:
        """Serialize generator state (models) to dictionary."""
        return {
            "chord_model": self.chord_model.to_dict(),
            "melody_model": self.melody_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopGenerator":
        """Deserialize generator from dictionary."""
        chord_model = ChordModel.from_dict(data["chord_model"])
        melody_model = MelodyModel.from_dict(data["melody_model"])
        return cls(chord_model=chord_model, melody_model=melody_model)

    def __repr__(self) -> str:
        return (
            f"LoopGenerator(chord_model={self.chord_model}, "
            f"melody_model={self.melody_model})"
        )

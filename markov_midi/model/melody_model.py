"""
Melody pitch and rhythm model for MarkovMIDI.

Generates melodic lines using relative pitch encoding (intervals) so the
model is key-agnostic. Pitches are transposed to the target key on output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from markov_midi.model.markov_chain import MarkovChain
from markov_midi.model.theory_priors import (
    create_melody_pitch_chain,
    create_melody_rhythm_chain,
    MELODY_INTERVALS,
    MELODY_RHYTHM_DURATIONS,
)

if TYPE_CHECKING:
    pass


@dataclass
class MelodyNote:
    """
    A melody note with relative pitch and duration.

    Attributes:
        interval: Interval from previous note in semitones (0 = repeat)
        duration: Duration in 16th notes
        start_time: Start time in 16th notes from loop start
    """

    interval: int
    duration: int
    start_time: int = 0


@dataclass
class MelodySequence:
    """
    A sequence of melody notes with metadata.

    Attributes:
        notes: List of MelodyNote objects (relative intervals)
        total_duration: Total duration in 16th notes
        transitions_used: Transitions used during generation (for reward learning)
    """

    notes: list[MelodyNote] = field(default_factory=list)
    total_duration: int = 0
    transitions_used: list[tuple[tuple[int, ...], int]] = field(default_factory=list)

    def to_absolute_pitches(self, start_midi: int = 60) -> list[int]:
        """
        Convert relative intervals to absolute MIDI pitches.

        Args:
            start_midi: Starting MIDI note number (default 60 = middle C)

        Returns:
            List of MIDI note numbers
        """
        pitches: list[int] = []
        current_pitch = start_midi

        for note in self.notes:
            current_pitch += note.interval
            # Clamp to valid MIDI range
            current_pitch = max(0, min(127, current_pitch))
            pitches.append(current_pitch)

        return pitches


class MelodyModel:
    """
    Model for generating melodies with rhythms.

    Uses two Markov chains:
    - Pitch chain: Generates intervals in semitones (relative encoding)
    - Rhythm chain: Generates durations in 16th notes

    The relative pitch encoding makes the model key-agnostic - it learns
    melodic patterns that can be transposed to any key.

    Example:
        >>> model = MelodyModel()
        >>> sequence = model.generate(num_bars=4)
        >>> pitches = sequence.to_absolute_pitches(start_midi=60)
        >>> for note, pitch in zip(sequence.notes, pitches):
        ...     print(f"MIDI {pitch} at {note.start_time} for {note.duration}")
    """

    def __init__(
        self,
        pitch_chain: MarkovChain[int] | None = None,
        rhythm_chain: MarkovChain[int] | None = None,
        smoothing: float = 0.5,
    ) -> None:
        """
        Initialize the melody model.

        Args:
            pitch_chain: Optional pre-configured pitch/interval chain
            rhythm_chain: Optional pre-configured rhythm chain
            smoothing: Smoothing factor if creating new chains
        """
        self.pitch_chain = pitch_chain or create_melody_pitch_chain(smoothing)
        self.rhythm_chain = rhythm_chain or create_melody_rhythm_chain(smoothing)

    def generate(
        self,
        num_bars: int = 4,
        beats_per_bar: int = 4,
        note_density: float = 0.75,
        rng: random.Random | None = None,
    ) -> MelodySequence:
        """
        Generate a melody sequence with rhythms.

        Args:
            num_bars: Number of bars to generate
            beats_per_bar: Beats per bar (4 for 4/4 time)
            note_density: Approximate ratio of note time to total time (0.0-1.0)
            rng: Optional random number generator for reproducibility

        Returns:
            MelodySequence with notes and transition tracking
        """
        total_16ths = num_bars * beats_per_bar * 4

        notes: list[MelodyNote] = []
        all_transitions: list[tuple[tuple[int, ...], int]] = []

        current_time = 0
        prev_interval = 0  # Start with no movement
        prev_prev_interval = 0
        prev_duration = 4  # Start with quarter note context

        # Use RNG for density check
        local_rng = rng or random.Random()

        while current_time < total_16ths:
            remaining = total_16ths - current_time

            # Generate interval
            pitch_context = (prev_prev_interval, prev_interval)
            interval, pitch_trans = self.pitch_chain.sample(pitch_context, rng=rng)
            all_transitions.extend(pitch_trans)

            # Generate duration
            rhythm_context = (prev_duration,)
            duration, rhythm_trans = self.rhythm_chain.sample(rhythm_context, rng=rng)
            all_transitions.extend(rhythm_trans)

            # Clamp duration
            duration = min(duration, remaining)

            # Add note
            notes.append(
                MelodyNote(
                    interval=interval,
                    duration=duration,
                    start_time=current_time,
                )
            )

            # Update context
            prev_prev_interval = prev_interval
            prev_interval = interval
            prev_duration = duration
            current_time += duration

            # Occasionally add small gaps based on density
            if local_rng.random() > note_density and current_time < total_16ths:
                gap = min(local_rng.choice([1, 2]), total_16ths - current_time)
                current_time += gap

        return MelodySequence(
            notes=notes,
            total_duration=total_16ths,
            transitions_used=all_transitions,
        )

    def apply_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to transitions used in generation.

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        delta = reward * sensitivity

        for context, next_state in transitions:
            # Determine which chain based on state values
            if next_state in MELODY_INTERVALS:
                # Check if context looks like intervals
                if all(c in MELODY_INTERVALS for c in context):
                    ctx = cast(tuple[int, int] | tuple[int], context)
                    self.pitch_chain.update_transition(ctx, next_state, delta)
            elif next_state in MELODY_RHYTHM_DURATIONS:
                ctx = cast(tuple[int, int] | tuple[int], context)
                self.rhythm_chain.update_transition(ctx, next_state, delta)

    def reset_to_priors(self) -> None:
        """Reset both chains to theory priors."""
        self.pitch_chain = create_melody_pitch_chain(self.pitch_chain.smoothing)
        self.rhythm_chain = create_melody_rhythm_chain(self.rhythm_chain.smoothing)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "pitch_chain": self.pitch_chain.to_dict(),
            "rhythm_chain": self.rhythm_chain.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MelodyModel":
        """Deserialize model from dictionary."""
        pitch_chain: MarkovChain[int] = MarkovChain.from_dict(data["pitch_chain"])
        rhythm_chain: MarkovChain[int] = MarkovChain.from_dict(data["rhythm_chain"])
        return cls(pitch_chain=pitch_chain, rhythm_chain=rhythm_chain)

    def __repr__(self) -> str:
        return f"MelodyModel(pitch_chain={self.pitch_chain}, rhythm_chain={self.rhythm_chain})"

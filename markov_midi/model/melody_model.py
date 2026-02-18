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
        interval: Interval from previous note in SCALE DEGREES (not semitones).
                  0 = repeat, +1 = one scale step up, -2 = two scale steps down, etc.
                  This ensures all generated notes stay in the target key.
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
        notes: List of MelodyNote objects (relative scale-degree intervals)
        total_duration: Total duration in 16th notes
        transitions_used: Transitions used during generation (for reward learning)
    """

    notes: list[MelodyNote] = field(default_factory=list)
    total_duration: int = 0
    transitions_used: list[tuple[tuple[int, ...], int]] = field(default_factory=list)

    def to_absolute_pitches(
        self,
        start_midi: int = 60,
        scale_pitches: list[int] | None = None,
    ) -> list[int]:
        """
        Convert relative scale-degree intervals to absolute MIDI pitches.

        Args:
            start_midi: Starting MIDI note number (default 60 = middle C).
                        This should be the root of the scale in the target octave.
            scale_pitches: List of 7 MIDI pitch classes (0-11) for the scale.
                          e.g., C major = [0, 2, 4, 5, 7, 9, 11]
                          If None, defaults to major scale (chromatic fallback).

        Returns:
            List of MIDI note numbers, all guaranteed to be in the scale.
        """
        # Default to major scale pitch classes if not provided
        if scale_pitches is None:
            scale_pitches = [0, 2, 4, 5, 7, 9, 11]  # Major scale

        pitches: list[int] = []

        # Calculate base octave and starting scale degree
        base_octave = start_midi // 12
        root_pitch_class = start_midi % 12

        # Find which scale degree corresponds to our starting note
        # (usually 0 = root, but we support starting on any scale tone)
        try:
            current_degree = scale_pitches.index(root_pitch_class)
        except ValueError:
            # If start_midi isn't in the scale, start on the root
            current_degree = 0

        current_octave_offset = 0  # How many octaves above/below base

        for note in self.notes:
            # Move by scale degrees
            current_degree += note.interval

            # Handle octave wrapping
            while current_degree >= 7:
                current_degree -= 7
                current_octave_offset += 1
            while current_degree < 0:
                current_degree += 7
                current_octave_offset -= 1

            # Calculate absolute MIDI pitch
            pitch_class = scale_pitches[current_degree]
            midi_pitch = (base_octave + current_octave_offset) * 12 + pitch_class

            # Clamp to valid MIDI range
            midi_pitch = max(0, min(127, midi_pitch))
            pitches.append(midi_pitch)

        return pitches


class MelodyModel:
    """
    Model for generating melodies with rhythms.

    Uses two Markov chains:
    - Pitch chain: Generates intervals in SCALE DEGREES (not semitones)
    - Rhythm chain: Generates durations in 16th notes

    The scale-degree encoding ensures all generated notes stay in the target key.
    Intervals are relative steps within the scale:
    - 0 = repeat the same note
    - +1 = move up one scale step
    - -2 = move down two scale steps (a third)
    - etc.

    Example:
        >>> model = MelodyModel()
        >>> sequence = model.generate(num_bars=4)
        >>> # Convert to C major pitches
        >>> c_major = [0, 2, 4, 5, 7, 9, 11]
        >>> pitches = sequence.to_absolute_pitches(start_midi=60, scale_pitches=c_major)
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

    def apply_pitch_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to melody pitch (interval) transitions only.

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        delta = reward * sensitivity

        for context, next_state in transitions:
            # Only apply to interval transitions
            if next_state in MELODY_INTERVALS:
                if all(c in MELODY_INTERVALS for c in context):
                    ctx = cast(tuple[int, int] | tuple[int], context)
                    self.pitch_chain.update_transition(ctx, next_state, delta)

    def apply_rhythm_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to melody rhythm (duration) transitions only.

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        delta = reward * sensitivity

        for context, next_state in transitions:
            # Only apply to rhythm duration transitions
            if next_state in MELODY_RHYTHM_DURATIONS:
                ctx = cast(tuple[int, int] | tuple[int], context)
                self.rhythm_chain.update_transition(ctx, next_state, delta)

    def apply_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to all transitions (both pitch and rhythm).

        This is a convenience method that applies the same reward to both chains.
        For separate control, use apply_pitch_reward() and apply_rhythm_reward().

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        self.apply_pitch_reward(transitions, reward, sensitivity)
        self.apply_rhythm_reward(transitions, reward, sensitivity)

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

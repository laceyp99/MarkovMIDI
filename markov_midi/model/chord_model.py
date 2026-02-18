"""
Chord progression and rhythm model for MarkovMIDI.

Combines chord progression and chord rhythm Markov chains to generate
complete chord sequences with timing information.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from markov_midi.model.markov_chain import MarkovChain
from markov_midi.model.theory_priors import (
    create_chord_chain,
    create_chord_rhythm_chain,
    CHORD_DEGREES,
    CHORD_RHYTHM_DURATIONS,
    CHORD_START_WEIGHTS,
    CHORD_END_WEIGHTS,
)

if TYPE_CHECKING:
    pass


@dataclass
class ChordEvent:
    """
    A chord event with degree and duration.

    Attributes:
        degree: Scale degree (1-7, where 1=I, 2=ii, etc.)
        duration: Duration in 16th notes
        start_time: Start time in 16th notes from loop start
    """

    degree: int
    duration: int
    start_time: int = 0


@dataclass
class ChordSequence:
    """
    A sequence of chord events with metadata.

    Attributes:
        events: List of ChordEvent objects
        total_duration: Total duration in 16th notes
        transitions_used: Transitions used during generation (for reward learning)
    """

    events: list[ChordEvent] = field(default_factory=list)
    total_duration: int = 0
    transitions_used: list[tuple[tuple[int, ...], int]] = field(default_factory=list)


class ChordModel:
    """
    Model for generating chord progressions with rhythms.

    Uses two Markov chains:
    - Chord progression chain: Generates scale degrees (I, ii, iii, IV, V, vi, vii)
    - Chord rhythm chain: Generates durations in 16th notes

    Example:
        >>> model = ChordModel()
        >>> sequence = model.generate(num_bars=4, beats_per_bar=4)
        >>> for event in sequence.events:
        ...     print(f"Chord {event.degree} at {event.start_time} for {event.duration}")
    """

    def __init__(
        self,
        chord_chain: MarkovChain[int] | None = None,
        rhythm_chain: MarkovChain[int] | None = None,
        smoothing: float = 0.5,
    ) -> None:
        """
        Initialize the chord model.

        Args:
            chord_chain: Optional pre-configured chord progression chain
            rhythm_chain: Optional pre-configured rhythm chain
            smoothing: Smoothing factor if creating new chains
        """
        self.chord_chain = chord_chain or create_chord_chain(smoothing)
        self.rhythm_chain = rhythm_chain or create_chord_rhythm_chain(smoothing)

    def generate(
        self,
        num_bars: int = 4,
        beats_per_bar: int = 4,
        start_degree: int | None = None,
        end_on_tonic: bool = False,
        position_aware: bool = True,
        rng: random.Random | None = None,
    ) -> ChordSequence:
        """
        Generate a chord progression with rhythms.

        Args:
            num_bars: Number of bars to generate
            beats_per_bar: Beats per bar (4 for 4/4 time)
            start_degree: Starting chord degree. If None and position_aware=True,
                         samples from CHORD_START_WEIGHTS (favoring I, vi, IV).
            end_on_tonic: If True, force the last chord to be I (tonic).
                         Note: with position_aware=True, the last chord already
                         favors V/IV which resolve nicely when loop repeats.
            position_aware: If True, use position-based weights:
                           - First chord: sample from CHORD_START_WEIGHTS
                           - Last chord: multiply by CHORD_END_WEIGHTS
            rng: Optional random number generator for reproducibility

        Returns:
            ChordSequence with events and transition tracking
        """
        total_16ths = num_bars * beats_per_bar * 4  # 4 16ths per beat
        local_rng = rng or random.Random()

        events: list[ChordEvent] = []
        all_transitions: list[tuple[tuple[int, ...], int]] = []

        current_time = 0
        prev_duration = 4  # Start with quarter note context
        is_first_chord = True

        # Determine first chord
        if start_degree is not None:
            # Explicit start degree
            first_degree = start_degree
        elif position_aware:
            # Sample from start weights
            first_degree = self._sample_from_weights(CHORD_START_WEIGHTS, local_rng)
        else:
            # Default to tonic
            first_degree = 1

        prev_degree = first_degree
        prev_prev_degree = first_degree

        while current_time < total_16ths:
            remaining = total_16ths - current_time

            if is_first_chord:
                # Use the pre-determined first chord
                next_degree = first_degree
                is_first_chord = False
            else:
                # Generate next chord degree
                chord_context = (prev_prev_degree, prev_degree)

                # Check if this will be the last chord (estimate based on typical duration)
                is_likely_last = remaining <= 16  # One bar or less remaining

                if position_aware and is_likely_last:
                    # Sample with end weights applied
                    next_degree, chord_trans = self._sample_with_position_weights(
                        chord_context, CHORD_END_WEIGHTS, rng
                    )
                else:
                    next_degree, chord_trans = self.chord_chain.sample(
                        chord_context, rng=rng
                    )

                all_transitions.extend(chord_trans)

            # Generate duration
            rhythm_context = (prev_duration,)
            duration, rhythm_trans = self.rhythm_chain.sample(rhythm_context, rng=rng)
            all_transitions.extend(rhythm_trans)

            # Clamp duration to remaining time
            duration = min(duration, remaining)

            # If ending on tonic and this is the last chord, force I
            if end_on_tonic and current_time + duration >= total_16ths:
                next_degree = 1

            events.append(
                ChordEvent(
                    degree=next_degree,
                    duration=duration,
                    start_time=current_time,
                )
            )

            # Update context
            prev_prev_degree = prev_degree
            prev_degree = next_degree
            prev_duration = duration
            current_time += duration

        return ChordSequence(
            events=events,
            total_duration=total_16ths,
            transitions_used=all_transitions,
        )

    def _sample_from_weights(
        self,
        weights: dict[int, float],
        rng: random.Random,
    ) -> int:
        """Sample a chord degree from a weight dictionary."""
        degrees = list(weights.keys())
        weight_values = [weights[d] for d in degrees]
        total = sum(weight_values)
        probs = [w / total for w in weight_values]
        return rng.choices(degrees, weights=probs, k=1)[0]

    def _sample_with_position_weights(
        self,
        context: tuple[int, int],
        position_weights: dict[int, float],
        rng: random.Random | None = None,
    ) -> tuple[int, list[tuple[tuple[int, ...], int]]]:
        """
        Sample next chord with position weights applied.

        Multiplies the Markov chain probabilities by position weights
        to bias toward certain chords based on position in the loop.
        """
        # Get base probabilities from the chain
        base_probs = self.chord_chain.get_probabilities(context)

        # Apply position weights
        adjusted: dict[int, float] = {}
        for degree in CHORD_DEGREES:
            base = base_probs.get(degree, self.chord_chain.smoothing)
            position_mult = position_weights.get(degree, 1.0)
            adjusted[degree] = base * position_mult

        # Normalize and sample
        total = sum(adjusted.values())
        if total == 0:
            # Fallback
            return 1, []

        local_rng = rng or random.Random()
        degrees = list(adjusted.keys())
        probs = [adjusted[d] / total for d in degrees]
        chosen = local_rng.choices(degrees, weights=probs, k=1)[0]

        # Record transition for reward learning
        transitions: list[tuple[tuple[int, ...], int]] = [(context, chosen)]

        return chosen, transitions

    def apply_chord_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to chord degree transitions only.

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        delta = reward * sensitivity

        for context, next_state in transitions:
            # Only apply to chord degree transitions
            if next_state in CHORD_DEGREES:
                if all(c in CHORD_DEGREES for c in context):
                    ctx = cast(tuple[int, int] | tuple[int], context)
                    self.chord_chain.update_transition(ctx, next_state, delta)

    def apply_rhythm_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to chord rhythm transitions only.

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        delta = reward * sensitivity

        for context, next_state in transitions:
            # Only apply to rhythm duration transitions
            if next_state in CHORD_RHYTHM_DURATIONS:
                ctx = cast(tuple[int, int] | tuple[int], context)
                self.rhythm_chain.update_transition(ctx, next_state, delta)

    def apply_reward(
        self,
        transitions: list[tuple[tuple[int, ...], int]],
        reward: float,
        sensitivity: float = 1.0,
    ) -> None:
        """
        Apply reward to all transitions (both chord and rhythm).

        This is a convenience method that applies the same reward to both chains.
        For separate control, use apply_chord_reward() and apply_rhythm_reward().

        Args:
            transitions: List of (context, next_state) tuples from generation
            reward: Reward value (positive = good, negative = bad)
            sensitivity: Multiplier for reward magnitude
        """
        self.apply_chord_reward(transitions, reward, sensitivity)
        self.apply_rhythm_reward(transitions, reward, sensitivity)

    def reset_to_priors(self) -> None:
        """Reset both chains to theory priors."""
        self.chord_chain = create_chord_chain(self.chord_chain.smoothing)
        self.rhythm_chain = create_chord_rhythm_chain(self.rhythm_chain.smoothing)

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "chord_chain": self.chord_chain.to_dict(),
            "rhythm_chain": self.rhythm_chain.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChordModel":
        """Deserialize model from dictionary."""
        chord_chain: MarkovChain[int] = MarkovChain.from_dict(data["chord_chain"])
        rhythm_chain: MarkovChain[int] = MarkovChain.from_dict(data["rhythm_chain"])
        return cls(chord_chain=chord_chain, rhythm_chain=rhythm_chain)

    def __repr__(self) -> str:
        return f"ChordModel(chord_chain={self.chord_chain}, rhythm_chain={self.rhythm_chain})"

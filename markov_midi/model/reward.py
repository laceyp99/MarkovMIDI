"""
Reward-based learning system for MarkovMIDI.

Manages the human-in-the-loop reward training workflow, including:
- Rating criteria (melodic, melodic_rhythm, harmonic, harmonic_rhythm, cohesion, overall)
- Reward sensitivity levels
- Generation and rating history tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class RewardSensitivity(Enum):
    """Reward sensitivity levels for model updates."""

    GENTLE = "gentle"  # Small adjustments, slow learning
    MODERATE = "moderate"  # Balanced (default)
    AGGRESSIVE = "aggressive"  # Large adjustments, fast learning


# Multipliers for each sensitivity level
SENSITIVITY_MULTIPLIERS: dict[RewardSensitivity, float] = {
    RewardSensitivity.GENTLE: 0.5,
    RewardSensitivity.MODERATE: 1.0,
    RewardSensitivity.AGGRESSIVE: 2.0,
}


@dataclass
class Rating:
    """
    A rating for a generated loop.

    All ratings are on a 1-5 star scale:
    - 1-2 stars: Negative (decreases probability)
    - 3 stars: Neutral (no change)
    - 4-5 stars: Positive (increases probability)

    Attributes:
        melodic: Melody pitch/interval choices
        melodic_rhythm: Melody note timing and phrasing
        harmonic: Chord progression choices
        harmonic_rhythm: Chord change timing
        cohesion: How well elements work together
        overall: General quality rating
    """

    melodic: int = 3
    melodic_rhythm: int = 3
    harmonic: int = 3
    harmonic_rhythm: int = 3
    cohesion: int = 3
    overall: int = 3

    def __post_init__(self) -> None:
        """Validate ratings are in range 1-5."""
        for name in [
            "melodic",
            "melodic_rhythm",
            "harmonic",
            "harmonic_rhythm",
            "cohesion",
            "overall",
        ]:
            value = getattr(self, name)
            if not 1 <= value <= 5:
                raise ValueError(f"{name} must be 1-5, got {value}")

    def to_chord_reward(self) -> float:
        """
        Calculate chord degree model reward from ratings.

        Weights harmonic rating most heavily for chord choices.
        """
        # Center ratings around 0 (3 stars = neutral)
        harmonic_delta = self.harmonic - 3
        overall_delta = self.overall - 3
        cohesion_delta = self.cohesion - 3

        return harmonic_delta * 1.5 + overall_delta * 0.5 + cohesion_delta * 0.5

    def to_chord_rhythm_reward(self) -> float:
        """
        Calculate chord rhythm model reward from ratings.

        Weights harmonic_rhythm rating most heavily for chord timing.
        """
        harmonic_rhythm_delta = self.harmonic_rhythm - 3
        overall_delta = self.overall - 3
        cohesion_delta = self.cohesion - 3

        return harmonic_rhythm_delta * 1.5 + overall_delta * 0.5 + cohesion_delta * 0.5

    def to_melody_reward(self) -> float:
        """
        Calculate melody pitch model reward from ratings.

        Weights melodic rating most heavily for pitch/interval choices.
        """
        melodic_delta = self.melodic - 3
        overall_delta = self.overall - 3
        cohesion_delta = self.cohesion - 3

        return melodic_delta * 1.5 + overall_delta * 0.5 + cohesion_delta * 0.5

    def to_melody_rhythm_reward(self) -> float:
        """
        Calculate melody rhythm model reward from ratings.

        Weights melodic_rhythm rating most heavily for melody timing.
        """
        melodic_rhythm_delta = self.melodic_rhythm - 3
        overall_delta = self.overall - 3
        cohesion_delta = self.cohesion - 3

        return melodic_rhythm_delta * 1.5 + overall_delta * 0.5 + cohesion_delta * 0.5

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "melodic": self.melodic,
            "melodic_rhythm": self.melodic_rhythm,
            "harmonic": self.harmonic,
            "harmonic_rhythm": self.harmonic_rhythm,
            "cohesion": self.cohesion,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Rating":
        """Create from dictionary."""
        return cls(
            melodic=data.get("melodic", 3),
            melodic_rhythm=data.get("melodic_rhythm", 3),
            harmonic=data.get("harmonic", 3),
            harmonic_rhythm=data.get("harmonic_rhythm", 3),
            cohesion=data.get("cohesion", 3),
            overall=data.get("overall", 3),
        )


@dataclass
class GenerationRecord:
    """
    Record of a generated loop for history tracking.

    Attributes:
        generation_id: Unique identifier
        timestamp: When the loop was generated
        params: Generation parameters used
        chord_transitions: Chord model transitions used
        melody_transitions: Melody model transitions used
        rating: Rating given (None if not rated)
        midi_path: Path to saved MIDI file (if saved)
    """

    generation_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    params: dict[str, Any] = field(default_factory=dict)
    chord_transitions: list[tuple[tuple[int, ...], int]] = field(default_factory=list)
    melody_transitions: list[tuple[tuple[int, ...], int]] = field(default_factory=list)
    rating: Rating | None = None
    midi_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation_id": self.generation_id,
            "timestamp": self.timestamp,
            "params": self.params,
            "chord_transitions": [
                (list(ctx), next_state) for ctx, next_state in self.chord_transitions
            ],
            "melody_transitions": [
                (list(ctx), next_state) for ctx, next_state in self.melody_transitions
            ],
            "rating": self.rating.to_dict() if self.rating else None,
            "midi_path": self.midi_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRecord":
        """Create from dictionary."""
        return cls(
            generation_id=data.get("generation_id", str(uuid4())[:8]),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            params=data.get("params", {}),
            chord_transitions=[
                (tuple(ctx), next_state)
                for ctx, next_state in data.get("chord_transitions", [])
            ],
            melody_transitions=[
                (tuple(ctx), next_state)
                for ctx, next_state in data.get("melody_transitions", [])
            ],
            rating=Rating.from_dict(data["rating"]) if data.get("rating") else None,
            midi_path=data.get("midi_path"),
        )


class RewardManager:
    """
    Manages reward-based learning workflow.

    Tracks generation history, calculates rewards from ratings,
    and applies rewards to models.

    Example:
        >>> from markov_midi.generator import LoopGenerator, GenerationParams
        >>> generator = LoopGenerator()
        >>> manager = RewardManager()
        >>>
        >>> # Generate and record
        >>> params = GenerationParams(key="C", mode="major")
        >>> loop = generator.generate(params)
        >>> record = manager.record_generation(loop, params)
        >>>
        >>> # Rate and apply reward
        >>> rating = Rating(overall=4, melodic=5, harmonic=4, rhythmic=3, cohesion=4)
        >>> manager.apply_rating(record.generation_id, rating, generator)
    """

    def __init__(
        self,
        sensitivity: RewardSensitivity = RewardSensitivity.MODERATE,
    ) -> None:
        """
        Initialize the reward manager.

        Args:
            sensitivity: Reward sensitivity level
        """
        self.sensitivity = sensitivity
        self.history: list[GenerationRecord] = []
        self._generation_map: dict[str, GenerationRecord] = {}

    @property
    def sensitivity_multiplier(self) -> float:
        """Get the current sensitivity multiplier."""
        return SENSITIVITY_MULTIPLIERS[self.sensitivity]

    def record_generation(
        self,
        loop: Any,  # GeneratedLoop - avoiding circular import
        params: Any,  # GenerationParams
        midi_path: str | None = None,
    ) -> GenerationRecord:
        """
        Record a new generation for potential rating.

        Args:
            loop: The generated loop
            params: Generation parameters used
            midi_path: Path to saved MIDI file

        Returns:
            GenerationRecord for this generation
        """
        # Convert params to dict for serialization
        params_dict: dict[str, Any] = {}
        if hasattr(params, "__dataclass_fields__"):
            for field_name in params.__dataclass_fields__:
                value = getattr(params, field_name)
                # Handle enums
                if hasattr(value, "value"):
                    value = value.value
                params_dict[field_name] = value

        record = GenerationRecord(
            params=params_dict,
            chord_transitions=list(loop.chord_transitions),
            melody_transitions=list(loop.melody_transitions),
            midi_path=midi_path,
        )

        self.history.append(record)
        self._generation_map[record.generation_id] = record

        return record

    def get_generation(self, generation_id: str) -> GenerationRecord | None:
        """Get a generation record by ID."""
        return self._generation_map.get(generation_id)

    def apply_rating(
        self,
        generation_id: str,
        rating: Rating,
        generator: Any,  # LoopGenerator
    ) -> bool:
        """
        Apply a rating to a generation and update the models.

        Applies separate rewards for:
        - Chord choices (harmonic rating)
        - Chord rhythm (harmonic_rhythm rating)
        - Melody pitches (melodic rating)
        - Melody rhythm (melodic_rhythm rating)

        Args:
            generation_id: ID of the generation to rate
            rating: Rating to apply
            generator: LoopGenerator to update

        Returns:
            True if successful, False if generation not found
        """
        record = self.get_generation(generation_id)
        if not record:
            return False

        # Store the rating
        record.rating = rating

        # Calculate separate rewards for each dimension
        chord_reward = rating.to_chord_reward()
        chord_rhythm_reward = rating.to_chord_rhythm_reward()
        melody_reward = rating.to_melody_reward()
        melody_rhythm_reward = rating.to_melody_rhythm_reward()

        # Apply to chord model (separate pitch and rhythm)
        if record.chord_transitions:
            if chord_reward != 0.0:
                generator.chord_model.apply_chord_reward(
                    record.chord_transitions,
                    chord_reward,
                    self.sensitivity_multiplier,
                )
            if chord_rhythm_reward != 0.0:
                generator.chord_model.apply_rhythm_reward(
                    record.chord_transitions,
                    chord_rhythm_reward,
                    self.sensitivity_multiplier,
                )

        # Apply to melody model (separate pitch and rhythm)
        if record.melody_transitions:
            if melody_reward != 0.0:
                generator.melody_model.apply_pitch_reward(
                    record.melody_transitions,
                    melody_reward,
                    self.sensitivity_multiplier,
                )
            if melody_rhythm_reward != 0.0:
                generator.melody_model.apply_rhythm_reward(
                    record.melody_transitions,
                    melody_rhythm_reward,
                    self.sensitivity_multiplier,
                )

        return True

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the reward training session.

        Returns:
            Dictionary with session statistics
        """
        total = len(self.history)
        rated = sum(1 for r in self.history if r.rating is not None)

        if rated == 0:
            return {
                "total_generations": total,
                "rated_generations": 0,
                "average_melodic": None,
                "average_melodic_rhythm": None,
                "average_harmonic": None,
                "average_harmonic_rhythm": None,
                "average_cohesion": None,
                "average_overall": None,
            }

        ratings = [r.rating for r in self.history if r.rating is not None]

        return {
            "total_generations": total,
            "rated_generations": rated,
            "average_melodic": sum(r.melodic for r in ratings) / rated,
            "average_melodic_rhythm": sum(r.melodic_rhythm for r in ratings) / rated,
            "average_harmonic": sum(r.harmonic for r in ratings) / rated,
            "average_harmonic_rhythm": sum(r.harmonic_rhythm for r in ratings) / rated,
            "average_cohesion": sum(r.cohesion for r in ratings) / rated,
            "average_overall": sum(r.overall for r in ratings) / rated,
        }

    def clear_history(self) -> None:
        """Clear all generation history."""
        self.history.clear()
        self._generation_map.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sensitivity": self.sensitivity.value,
            "history": [r.to_dict() for r in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardManager":
        """Deserialize from dictionary."""
        sensitivity = RewardSensitivity(data.get("sensitivity", "moderate"))
        manager = cls(sensitivity=sensitivity)

        for record_data in data.get("history", []):
            record = GenerationRecord.from_dict(record_data)
            manager.history.append(record)
            manager._generation_map[record.generation_id] = record

        return manager

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"RewardManager(sensitivity={self.sensitivity.value}, "
            f"generations={stats['total_generations']}, "
            f"rated={stats['rated_generations']})"
        )

"""
Second-order Markov chain implementation for MarkovMIDI.

Provides a generic Markov chain that can be used for:
- Chord progressions
- Chord rhythms
- Melody pitches (as intervals)
- Melody rhythms

Features:
- Second-order transitions (context of 2 previous states)
- Laplace smoothing to prevent zero probabilities
- First-order fallback when 2nd-order context is unseen
- Transition tracking for reward-based learning
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Generic, TypeVar, Hashable, Any

import numpy as np

# Generic type for states (can be str, int, tuple, etc.)
S = TypeVar("S", bound=Hashable)


class MarkovChain(Generic[S]):
    """
    A second-order Markov chain with smoothing and fallback.

    The chain tracks transitions of the form:
        (state_n-2, state_n-1) -> state_n

    When a second-order context hasn't been seen, it falls back to
    first-order transitions:
        (state_n-1) -> state_n

    Attributes:
        smoothing: Laplace smoothing factor (added to all counts)
        states: Set of all known states

    Example:
        >>> chain = MarkovChain[str](smoothing=1.0)
        >>> chain.train(["I", "IV", "V", "I", "IV", "V", "I"])
        >>> next_chord, transitions = chain.generate(("IV", "V"))
        >>> print(next_chord)  # Likely "I"
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        states: set[S] | None = None,
    ) -> None:
        """
        Initialize the Markov chain.

        Args:
            smoothing: Laplace smoothing factor. Higher values make the
                      distribution more uniform. Default 1.0 (add-one smoothing).
            states: Optional set of all possible states. If not provided,
                   states are learned from training data.
        """
        self.smoothing = smoothing
        self.states: set[S] = states.copy() if states else set()

        # Second-order transition counts: (prev2, prev1) -> {next: count}
        self._order2_counts: dict[tuple[S, S], dict[S, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # First-order transition counts: prev1 -> {next: count}
        self._order1_counts: dict[S, dict[S, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Track total counts for each context (for normalization)
        self._order2_totals: dict[tuple[S, S], float] = defaultdict(float)
        self._order1_totals: dict[S, float] = defaultdict(float)

        # Global state distribution (fallback when no context matches)
        self._global_counts: dict[S, float] = defaultdict(float)
        self._global_total: float = 0.0

    def add_states(self, states: set[S]) -> None:
        """
        Add states to the known state set.

        This is useful for pre-defining states before training,
        especially when using theory priors.

        Args:
            states: Set of states to add
        """
        self.states.update(states)

    def train(self, sequence: list[S]) -> None:
        """
        Train the chain on a sequence of states.

        Updates both first-order and second-order transition counts.

        Args:
            sequence: List of states in order
        """
        if len(sequence) < 2:
            return

        # Add all states to known states
        self.states.update(sequence)

        # Update global counts
        for state in sequence:
            self._global_counts[state] += 1.0
            self._global_total += 1.0

        # First-order transitions
        for i in range(len(sequence) - 1):
            prev1 = sequence[i]
            next_state = sequence[i + 1]
            self._order1_counts[prev1][next_state] += 1.0
            self._order1_totals[prev1] += 1.0

        # Second-order transitions
        for i in range(len(sequence) - 2):
            prev2 = sequence[i]
            prev1 = sequence[i + 1]
            next_state = sequence[i + 2]
            context = (prev2, prev1)
            self._order2_counts[context][next_state] += 1.0
            self._order2_totals[context] += 1.0

    def get_probabilities(
        self,
        context: tuple[S, S] | tuple[S] | None = None,
    ) -> dict[S, float]:
        """
        Get transition probabilities for a given context.

        Uses second-order context if available, falls back to first-order,
        then to global distribution.

        Args:
            context: Tuple of (prev2, prev1) for 2nd order, (prev1,) for 1st order,
                    or None for global distribution

        Returns:
            Dictionary mapping each state to its probability
        """
        if not self.states:
            return {}

        num_states = len(self.states)

        # Try second-order
        if context is not None and len(context) == 2:
            ctx2 = (context[0], context[1])
            if ctx2 in self._order2_counts:
                return self._compute_probabilities(
                    self._order2_counts[ctx2],
                    self._order2_totals[ctx2],
                    num_states,
                )
            # Fall back to first-order
            context = (context[1],)

        # Try first-order
        if context is not None and len(context) == 1:
            prev1 = context[0]
            if prev1 in self._order1_counts:
                return self._compute_probabilities(
                    self._order1_counts[prev1],
                    self._order1_totals[prev1],
                    num_states,
                )

        # Fall back to global distribution
        return self._compute_probabilities(
            self._global_counts,
            self._global_total,
            num_states,
        )

    def _compute_probabilities(
        self,
        counts: dict[S, float],
        total: float,
        num_states: int,
    ) -> dict[S, float]:
        """
        Compute smoothed probabilities from counts.

        Uses Laplace smoothing: P(s) = (count(s) + k) / (total + k*N)
        where k is the smoothing factor and N is the number of states.
        """
        smoothed_total = total + self.smoothing * num_states

        probs: dict[S, float] = {}
        for state in self.states:
            count = counts.get(state, 0.0)
            probs[state] = (count + self.smoothing) / smoothed_total

        return probs

    def sample(
        self,
        context: tuple[S, S] | tuple[S] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[S, list[tuple[tuple[S, ...], S]]]:
        """
        Sample the next state given a context.

        Args:
            context: Previous state(s) as context
            rng: Optional random number generator for reproducibility

        Returns:
            Tuple of (sampled_state, transitions_used) where transitions_used
            is a list of (context, next_state) tuples for tracking

        Raises:
            ValueError: If no states are defined
        """
        if not self.states:
            raise ValueError("No states defined. Train the chain or add states first.")

        probs = self.get_probabilities(context)

        # Convert to arrays for numpy sampling
        states_list = list(probs.keys())
        prob_array = np.array([probs[s] for s in states_list])

        # Normalize (should already be normalized, but ensure)
        prob_array = prob_array / prob_array.sum()

        # Sample
        if rng is not None:
            # Use provided RNG
            r = rng.random()
            cumsum = 0.0
            for i, p in enumerate(prob_array):
                cumsum += p
                if r < cumsum:
                    sampled = states_list[i]
                    break
            else:
                sampled = states_list[-1]
        else:
            # Use numpy
            idx = np.random.choice(len(states_list), p=prob_array)
            sampled = states_list[idx]

        # Track which transition was used
        transitions: list[tuple[tuple[S, ...], S]] = []
        if context is not None:
            transitions.append((context, sampled))

        return sampled, transitions

    def generate(
        self,
        context: tuple[S, S] | tuple[S],
        length: int = 1,
        rng: random.Random | None = None,
    ) -> tuple[list[S], list[tuple[tuple[S, ...], S]]]:
        """
        Generate a sequence of states.

        Args:
            context: Initial context (1 or 2 previous states)
            length: Number of states to generate
            rng: Optional random number generator

        Returns:
            Tuple of (generated_sequence, all_transitions_used)
        """
        sequence: list[S] = []
        all_transitions: list[tuple[tuple[S, ...], S]] = []

        # Build initial context as list for sliding window
        ctx_list = list(context)

        for _ in range(length):
            # Use last 2 states as context (or less if not enough)
            if len(ctx_list) >= 2:
                ctx: tuple[S, ...] = (ctx_list[-2], ctx_list[-1])
            elif len(ctx_list) == 1:
                ctx = (ctx_list[-1],)
            else:
                ctx = ()

            sampled, transitions = self.sample(
                ctx if ctx else None,  # type: ignore
                rng=rng,
            )
            sequence.append(sampled)
            all_transitions.extend(transitions)
            ctx_list.append(sampled)

        return sequence, all_transitions

    def update_transition(
        self,
        context: tuple[S, S] | tuple[S],
        next_state: S,
        delta: float,
    ) -> None:
        """
        Update a transition's count by a delta value.

        Used by the reward system to increase/decrease probabilities.
        Counts are clamped to a minimum of 0.

        Args:
            context: The context (prev states)
            next_state: The next state
            delta: Amount to add (positive) or subtract (negative)
        """
        if len(context) == 2:
            ctx2 = (context[0], context[1])
            old_count = self._order2_counts[ctx2][next_state]
            new_count = max(0.0, old_count + delta)
            delta_actual = new_count - old_count
            self._order2_counts[ctx2][next_state] = new_count
            self._order2_totals[ctx2] += delta_actual

        if len(context) >= 1:
            prev1 = context[-1]
            old_count = self._order1_counts[prev1][next_state]
            new_count = max(0.0, old_count + delta)
            delta_actual = new_count - old_count
            self._order1_counts[prev1][next_state] = new_count
            self._order1_totals[prev1] += delta_actual

    def set_transition(
        self,
        context: tuple[S, S] | tuple[S],
        next_state: S,
        count: float,
    ) -> None:
        """
        Set a transition's count to a specific value.

        Used for initializing with theory priors.

        Args:
            context: The context (prev states)
            next_state: The next state
            count: The count value to set
        """
        count = max(0.0, count)

        if len(context) == 2:
            ctx2 = (context[0], context[1])
            old_count = self._order2_counts[ctx2].get(next_state, 0.0)
            self._order2_counts[ctx2][next_state] = count
            self._order2_totals[ctx2] += count - old_count

        if len(context) >= 1:
            prev1 = context[-1]
            old_count = self._order1_counts[prev1].get(next_state, 0.0)
            self._order1_counts[prev1][next_state] = count
            self._order1_totals[prev1] += count - old_count

        # Ensure states are tracked
        self.states.add(next_state)
        for s in context:
            self.states.add(s)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the chain to a dictionary for JSON persistence.

        Returns:
            Dictionary with all chain state
        """
        return {
            "smoothing": self.smoothing,
            "states": list(self.states),
            "order2_counts": {
                f"{k[0]}|{k[1]}": dict(v) for k, v in self._order2_counts.items()
            },
            "order1_counts": {str(k): dict(v) for k, v in self._order1_counts.items()},
            "order2_totals": {
                f"{k[0]}|{k[1]}": v for k, v in self._order2_totals.items()
            },
            "order1_totals": {str(k): v for k, v in self._order1_totals.items()},
            "global_counts": dict(self._global_counts),
            "global_total": self._global_total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkovChain[S]":
        """
        Deserialize a chain from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed MarkovChain
        """
        chain: MarkovChain[S] = cls(smoothing=data["smoothing"])
        chain.states = set(data["states"])

        # Restore order2 counts
        for key_str, counts in data.get("order2_counts", {}).items():
            parts = key_str.split("|")
            # Try to preserve original types (int vs str)
            try:
                key2: tuple[S, S] = (int(parts[0]), int(parts[1]))  # type: ignore[assignment]
            except ValueError:
                key2 = (parts[0], parts[1])
            chain._order2_counts[key2] = defaultdict(float, counts)

        # Restore order1 counts
        for key_str, counts in data.get("order1_counts", {}).items():
            try:
                key1: S = int(key_str)  # type: ignore[assignment]
            except ValueError:
                key1 = key_str
            chain._order1_counts[key1] = defaultdict(float, counts)

        # Restore totals
        for key_str, total in data.get("order2_totals", {}).items():
            parts = key_str.split("|")
            try:
                key2_t: tuple[S, S] = (int(parts[0]), int(parts[1]))  # type: ignore[assignment]
            except ValueError:
                key2_t = (parts[0], parts[1])
            chain._order2_totals[key2_t] = total

        for key_str, total in data.get("order1_totals", {}).items():
            try:
                key1_t: S = int(key_str)  # type: ignore[assignment]
            except ValueError:
                key1_t = key_str
            chain._order1_totals[key1_t] = total

        chain._global_counts = defaultdict(float, data.get("global_counts", {}))
        chain._global_total = data.get("global_total", 0.0)

        return chain

    def reset(self) -> None:
        """
        Reset all learned transitions, keeping only the state set.
        """
        self._order2_counts.clear()
        self._order1_counts.clear()
        self._order2_totals.clear()
        self._order1_totals.clear()
        self._global_counts.clear()
        self._global_total = 0.0

    def __repr__(self) -> str:
        return (
            f"MarkovChain(states={len(self.states)}, "
            f"order2_contexts={len(self._order2_counts)}, "
            f"order1_contexts={len(self._order1_counts)}, "
            f"smoothing={self.smoothing})"
        )

"""
Unit tests for MarkovMIDI model module.

Tests cover:
- markov_chain.py: Core Markov chain functionality
- theory_priors.py: Music theory prior initialization
"""

import random

import pytest

from markov_midi.model.markov_chain import MarkovChain
from markov_midi.model.theory_priors import (
    CHORD_DEGREES,
    CHORD_RHYTHM_DURATIONS,
    MELODY_INTERVALS,
    MELODY_RHYTHM_DURATIONS,
    create_chord_chain,
    create_chord_rhythm_chain,
    create_melody_pitch_chain,
    create_melody_rhythm_chain,
    create_all_chains,
)


# =============================================================================
# MarkovChain Tests
# =============================================================================


class TestMarkovChainBasics:
    """Tests for basic MarkovChain functionality."""

    def test_init_empty(self) -> None:
        """Chain initializes with no states."""
        chain: MarkovChain[str] = MarkovChain()
        assert len(chain.states) == 0
        assert chain.smoothing == 1.0

    def test_init_with_states(self) -> None:
        """Chain can be initialized with predefined states."""
        states = {"A", "B", "C"}
        chain: MarkovChain[str] = MarkovChain(states=states)
        assert chain.states == states

    def test_init_custom_smoothing(self) -> None:
        """Custom smoothing factor is stored."""
        chain: MarkovChain[str] = MarkovChain(smoothing=0.5)
        assert chain.smoothing == 0.5

    def test_add_states(self) -> None:
        """States can be added after initialization."""
        chain: MarkovChain[str] = MarkovChain()
        chain.add_states({"X", "Y"})
        assert "X" in chain.states
        assert "Y" in chain.states


class TestMarkovChainTraining:
    """Tests for training the Markov chain."""

    def test_train_adds_states(self) -> None:
        """Training adds states to the chain."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C", "A"])
        assert chain.states == {"A", "B", "C"}

    def test_train_short_sequence(self) -> None:
        """Training with < 2 items does nothing."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A"])
        assert len(chain.states) == 0

    def test_train_first_order_counts(self) -> None:
        """Training updates first-order transition counts."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "A", "B", "A"])

        # A -> B appears twice
        # B -> A appears twice
        probs = chain.get_probabilities(("A",))
        assert probs["B"] > probs["A"]  # B is more likely after A

    def test_train_second_order_counts(self) -> None:
        """Training updates second-order transition counts."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C", "A", "B", "C"])

        # (A, B) -> C appears twice
        probs = chain.get_probabilities(("A", "B"))
        assert probs["C"] > probs["A"]


class TestMarkovChainProbabilities:
    """Tests for probability calculations."""

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities should sum to 1.0."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C", "A", "B"])

        probs = chain.get_probabilities(("A",))
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.0001

    def test_smoothing_prevents_zero(self) -> None:
        """Smoothing ensures no probability is zero."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "A", "B"])  # C never follows anything

        probs = chain.get_probabilities(("A",))
        assert probs["C"] > 0  # C still has non-zero probability

    def test_fallback_to_first_order(self) -> None:
        """Falls back to first-order when second-order context unseen."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C"])  # Only one occurrence of each

        # (X, A) is unseen, should fall back to just (A,)
        probs = chain.get_probabilities(("X", "A"))
        assert "B" in probs
        assert probs["B"] > probs["A"]  # B follows A in training

    def test_fallback_to_global(self) -> None:
        """Falls back to global distribution when no context matches."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C"])

        # Completely unknown context
        probs = chain.get_probabilities(("Z",))
        # Should return global distribution
        assert len(probs) == 3


class TestMarkovChainSampling:
    """Tests for sampling from the chain."""

    def test_sample_returns_valid_state(self) -> None:
        """Sampling returns a state from the state set."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"])

        sampled, transitions = chain.sample(("A",))
        assert sampled in chain.states

    def test_sample_returns_transitions(self) -> None:
        """Sampling returns the transitions used."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"])

        sampled, transitions = chain.sample(("A", "B"))
        assert len(transitions) == 1
        assert transitions[0][0] == ("A", "B")
        assert transitions[0][1] == sampled

    def test_sample_reproducible_with_rng(self) -> None:
        """Sampling is reproducible with same RNG seed."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C", "D", "E"})
        chain.train(["A", "B", "C", "D", "E"] * 10)

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        results1 = [chain.sample(("A",), rng=rng1)[0] for _ in range(10)]
        results2 = [chain.sample(("A",), rng=rng2)[0] for _ in range(10)]

        assert results1 == results2

    def test_sample_no_states_raises(self) -> None:
        """Sampling with no states raises ValueError."""
        chain: MarkovChain[str] = MarkovChain()
        with pytest.raises(ValueError, match="No states defined"):
            chain.sample(None)


class TestMarkovChainGenerate:
    """Tests for sequence generation."""

    def test_generate_correct_length(self) -> None:
        """Generate produces sequence of correct length."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"] * 5)

        sequence, _ = chain.generate(("A", "B"), length=10)
        assert len(sequence) == 10

    def test_generate_returns_all_transitions(self) -> None:
        """Generate returns all transitions used."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"] * 5)

        sequence, transitions = chain.generate(("A", "B"), length=5)
        assert len(transitions) == 5

    def test_generate_single_context(self) -> None:
        """Generate works with single-element context."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"] * 5)

        sequence, _ = chain.generate(("A",), length=5)
        assert len(sequence) == 5


class TestMarkovChainUpdates:
    """Tests for updating transition probabilities."""

    def test_update_transition_positive(self) -> None:
        """Positive delta increases transition probability."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B", "C"])

        probs_before = chain.get_probabilities(("A",))
        chain.update_transition(("A",), "B", 10.0)
        probs_after = chain.get_probabilities(("A",))

        assert probs_after["B"] > probs_before["B"]

    def test_update_transition_negative(self) -> None:
        """Negative delta decreases transition probability."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.train(["A", "B"] * 10)  # Strong A->B association

        probs_before = chain.get_probabilities(("A",))
        chain.update_transition(("A",), "B", -5.0)
        probs_after = chain.get_probabilities(("A",))

        assert probs_after["B"] < probs_before["B"]

    def test_update_transition_clamped_to_zero(self) -> None:
        """Counts can't go below zero."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B"})
        chain.train(["A", "B"])

        # Try to reduce by more than existing count
        chain.update_transition(("A",), "B", -1000.0)

        probs = chain.get_probabilities(("A",))
        # Should still have positive probability due to smoothing
        assert probs["B"] > 0

    def test_set_transition(self) -> None:
        """Set transition sets count to exact value."""
        chain: MarkovChain[str] = MarkovChain(states={"A", "B", "C"})
        chain.set_transition(("A",), "B", 10.0)
        chain.set_transition(("A",), "C", 1.0)

        probs = chain.get_probabilities(("A",))
        # B should be ~10x more likely than C (before smoothing)
        assert probs["B"] > probs["C"]


class TestMarkovChainSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_and_back(self) -> None:
        """Chain can be serialized and deserialized."""
        chain: MarkovChain[str] = MarkovChain(smoothing=0.5)
        chain.train(["A", "B", "C", "A", "B", "C"])

        data = chain.to_dict()
        restored: MarkovChain[str] = MarkovChain.from_dict(data)

        assert restored.smoothing == chain.smoothing
        assert restored.states == chain.states

    def test_serialization_preserves_probabilities(self) -> None:
        """Probabilities are preserved after serialization."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C"] * 10)

        probs_before = chain.get_probabilities(("A", "B"))

        data = chain.to_dict()
        restored: MarkovChain[str] = MarkovChain.from_dict(data)

        probs_after = restored.get_probabilities(("A", "B"))

        for state in probs_before:
            assert abs(probs_before[state] - probs_after[state]) < 0.0001

    def test_reset_clears_counts(self) -> None:
        """Reset clears all learned transitions."""
        chain: MarkovChain[str] = MarkovChain()
        chain.train(["A", "B", "C"] * 10)

        chain.reset()

        # States are preserved, but counts are cleared
        assert chain.states == {"A", "B", "C"}
        assert chain._global_total == 0.0


# =============================================================================
# Theory Priors Tests
# =============================================================================


class TestTheoryPriorsChords:
    """Tests for chord progression priors."""

    def test_create_chord_chain(self) -> None:
        """Creates chord chain with all degrees."""
        chain = create_chord_chain()
        assert chain.states == set(CHORD_DEGREES)

    def test_chord_chain_can_sample(self) -> None:
        """Chord chain can sample immediately."""
        chain = create_chord_chain()
        sampled, _ = chain.sample((1,))  # After I chord
        assert sampled in CHORD_DEGREES

    def test_v_to_i_likely(self) -> None:
        """V -> I should be a high probability transition."""
        chain = create_chord_chain()
        probs = chain.get_probabilities((5,))  # After V chord
        # I should be most likely after V
        assert probs[1] == max(probs.values())


class TestTheoryPriorsRhythm:
    """Tests for rhythm priors."""

    def test_create_chord_rhythm_chain(self) -> None:
        """Creates chord rhythm chain with correct durations."""
        chain = create_chord_rhythm_chain()
        assert chain.states == set(CHORD_RHYTHM_DURATIONS)

    def test_create_melody_rhythm_chain(self) -> None:
        """Creates melody rhythm chain with correct durations."""
        chain = create_melody_rhythm_chain()
        assert chain.states == set(MELODY_RHYTHM_DURATIONS)


class TestTheoryPriorsMelody:
    """Tests for melody pitch priors."""

    def test_create_melody_pitch_chain(self) -> None:
        """Creates melody pitch chain with correct intervals."""
        chain = create_melody_pitch_chain()
        assert chain.states == set(MELODY_INTERVALS)

    def test_stepwise_motion_preferred(self) -> None:
        """Small intervals should be more likely than large ones."""
        chain = create_melody_pitch_chain()
        probs = chain.get_probabilities((0,))  # After unison

        # Small intervals (steps) should be more likely than large leaps
        step_prob = probs[1] + probs[-1] + probs[2] + probs[-2]
        large_leap_prob = probs[10] + probs[-10] + probs[11] + probs[-11]

        assert step_prob > large_leap_prob


class TestCreateAllChains:
    """Tests for the create_all_chains factory."""

    def test_creates_four_chains(self) -> None:
        """Creates all four chains."""
        chains = create_all_chains()
        assert "chord" in chains
        assert "chord_rhythm" in chains
        assert "melody_pitch" in chains
        assert "melody_rhythm" in chains

    def test_all_chains_can_generate(self) -> None:
        """All chains can generate sequences."""
        chains = create_all_chains()

        for name, chain in chains.items():
            # Get a valid starting state
            start = next(iter(chain.states))
            sequence, _ = chain.generate((start,), length=5)
            assert len(sequence) == 5, f"{name} chain failed to generate"

    def test_custom_smoothing_applied(self) -> None:
        """Custom smoothing is applied to all chains."""
        chains = create_all_chains(smoothing=0.25)

        for chain in chains.values():
            assert chain.smoothing == 0.25

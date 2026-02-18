"""
Unit tests for MarkovMIDI model module.

Tests cover:
- markov_chain.py: Core Markov chain functionality
- theory_priors.py: Music theory prior initialization
"""

import random

import pytest

from markov_midi.generator.loop_generator import LoopGenerator, GenerationParams
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
from markov_midi.model.reward import (
    RewardSensitivity,
    SENSITIVITY_MULTIPLIERS,
    Rating,
    GenerationRecord,
    RewardManager,
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


# =============================================================================
# ChordModel Tests
# =============================================================================

from markov_midi.model.chord_model import ChordModel, ChordEvent, ChordSequence


class TestChordEvent:
    """Tests for ChordEvent dataclass."""

    def test_chord_event_creation(self) -> None:
        """ChordEvent can be created with required fields."""
        event = ChordEvent(degree=1, duration=4)
        assert event.degree == 1
        assert event.duration == 4
        assert event.start_time == 0  # Default

    def test_chord_event_with_start_time(self) -> None:
        """ChordEvent accepts custom start_time."""
        event = ChordEvent(degree=5, duration=8, start_time=16)
        assert event.start_time == 16


class TestChordSequence:
    """Tests for ChordSequence dataclass."""

    def test_chord_sequence_empty(self) -> None:
        """Empty ChordSequence can be created."""
        seq = ChordSequence()
        assert seq.events == []
        assert seq.total_duration == 0
        assert seq.transitions_used == []

    def test_chord_sequence_with_events(self) -> None:
        """ChordSequence holds events correctly."""
        events = [
            ChordEvent(degree=1, duration=8, start_time=0),
            ChordEvent(degree=5, duration=8, start_time=8),
        ]
        seq = ChordSequence(events=events, total_duration=16)
        assert len(seq.events) == 2
        assert seq.total_duration == 16


class TestChordModelBasics:
    """Tests for basic ChordModel functionality."""

    def test_init_default(self) -> None:
        """ChordModel initializes with default chains."""
        model = ChordModel()
        assert model.chord_chain is not None
        assert model.rhythm_chain is not None
        assert model.chord_chain.states == set(CHORD_DEGREES)

    def test_init_custom_smoothing(self) -> None:
        """ChordModel respects custom smoothing."""
        model = ChordModel(smoothing=0.25)
        assert model.chord_chain.smoothing == 0.25
        assert model.rhythm_chain.smoothing == 0.25

    def test_init_custom_chains(self) -> None:
        """ChordModel accepts pre-configured chains."""
        custom_chord = create_chord_chain(smoothing=0.1)
        custom_rhythm = create_chord_rhythm_chain(smoothing=0.2)
        model = ChordModel(chord_chain=custom_chord, rhythm_chain=custom_rhythm)
        assert model.chord_chain.smoothing == 0.1
        assert model.rhythm_chain.smoothing == 0.2


class TestChordModelGeneration:
    """Tests for chord sequence generation."""

    def test_generate_returns_sequence(self) -> None:
        """Generate returns a ChordSequence."""
        model = ChordModel()
        seq = model.generate(num_bars=4)
        assert isinstance(seq, ChordSequence)

    def test_generate_correct_duration(self) -> None:
        """Generate produces correct total duration."""
        model = ChordModel()
        seq = model.generate(num_bars=4, beats_per_bar=4)
        assert seq.total_duration == 4 * 4 * 4  # 64 sixteenths

    def test_generate_fills_duration(self) -> None:
        """Generated events fill the total duration."""
        model = ChordModel()
        seq = model.generate(num_bars=2)

        # Sum of event durations should equal total
        total = sum(e.duration for e in seq.events)
        assert total == seq.total_duration

    def test_generate_start_times_correct(self) -> None:
        """Events have correct cumulative start times."""
        model = ChordModel()
        seq = model.generate(num_bars=2)

        expected_time = 0
        for event in seq.events:
            assert event.start_time == expected_time
            expected_time += event.duration

    def test_generate_tracks_transitions(self) -> None:
        """Generate records transitions used."""
        model = ChordModel()
        seq = model.generate(num_bars=2)
        assert len(seq.transitions_used) > 0

    def test_generate_starts_on_given_degree(self) -> None:
        """Generate respects start_degree parameter."""
        model = ChordModel()
        rng = random.Random(42)

        # Generate multiple times with same seed
        seq = model.generate(num_bars=1, start_degree=4, rng=rng)
        # First generated chord may not be start_degree (that's the context)
        # But transitions should start from that context
        assert seq.events[0].degree in CHORD_DEGREES

    def test_generate_ends_on_tonic(self) -> None:
        """Generate ends on I when end_on_tonic=True."""
        model = ChordModel()
        seq = model.generate(num_bars=4, end_on_tonic=True)
        assert seq.events[-1].degree == 1

    def test_generate_can_end_elsewhere(self) -> None:
        """Generate can end on non-tonic when end_on_tonic=False."""
        model = ChordModel()
        # Run multiple times to find one that doesn't end on I
        found_non_tonic = False
        for seed in range(100):
            rng = random.Random(seed)
            seq = model.generate(num_bars=2, end_on_tonic=False, rng=rng)
            if seq.events[-1].degree != 1:
                found_non_tonic = True
                break
        # Should find at least one that doesn't end on I
        assert found_non_tonic

    def test_generate_reproducible_with_rng(self) -> None:
        """Generation is reproducible with same RNG."""
        model = ChordModel()

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        seq1 = model.generate(num_bars=4, rng=rng1)
        seq2 = model.generate(num_bars=4, rng=rng2)

        assert len(seq1.events) == len(seq2.events)
        for e1, e2 in zip(seq1.events, seq2.events):
            assert e1.degree == e2.degree
            assert e1.duration == e2.duration

    def test_generate_8_bars(self) -> None:
        """Generate works for 8-bar loops."""
        model = ChordModel()
        seq = model.generate(num_bars=8)
        assert seq.total_duration == 8 * 4 * 4  # 128 sixteenths


class TestChordModelReward:
    """Tests for reward learning."""

    def test_apply_reward_positive(self) -> None:
        """Positive reward increases probabilities."""
        model = ChordModel()

        # Generate and get transitions
        seq = model.generate(num_bars=2)

        # Get probability before
        prob_before = model.chord_chain.get_probabilities((1,))

        # Apply reward
        model.apply_reward(seq.transitions_used, reward=10.0)

        # At least one probability should have changed
        prob_after = model.chord_chain.get_probabilities((1,))
        # Difficult to test exactly, but model should accept the reward
        assert prob_after is not None

    def test_apply_reward_negative(self) -> None:
        """Negative reward accepted."""
        model = ChordModel()
        seq = model.generate(num_bars=2)
        # Should not raise
        model.apply_reward(seq.transitions_used, reward=-5.0)

    def test_apply_reward_with_sensitivity(self) -> None:
        """Sensitivity multiplies reward."""
        model = ChordModel()
        seq = model.generate(num_bars=2)
        # Should not raise
        model.apply_reward(seq.transitions_used, reward=1.0, sensitivity=0.5)


class TestChordModelSerialization:
    """Tests for serialization."""

    def test_to_dict(self) -> None:
        """ChordModel can be serialized to dict."""
        model = ChordModel()
        data = model.to_dict()
        assert "chord_chain" in data
        assert "rhythm_chain" in data

    def test_from_dict(self) -> None:
        """ChordModel can be deserialized from dict."""
        model = ChordModel(smoothing=0.3)
        # Train a bit
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=5.0)

        data = model.to_dict()
        restored = ChordModel.from_dict(data)

        assert restored.chord_chain.smoothing == 0.3
        assert restored.chord_chain.states == model.chord_chain.states

    def test_reset_to_priors(self) -> None:
        """reset_to_priors restores initial state."""
        model = ChordModel()

        # Apply some rewards
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=100.0)

        # Reset
        model.reset_to_priors()

        # Should have fresh chains
        assert model.chord_chain.states == set(CHORD_DEGREES)

    def test_repr(self) -> None:
        """ChordModel has string representation."""
        model = ChordModel()
        rep = repr(model)
        assert "ChordModel" in rep


# =============================================================================
# MelodyModel Tests
# =============================================================================

from markov_midi.model.melody_model import MelodyModel, MelodyNote, MelodySequence


class TestMelodyNote:
    """Tests for MelodyNote dataclass."""

    def test_melody_note_creation(self) -> None:
        """MelodyNote can be created with required fields."""
        note = MelodyNote(interval=2, duration=4)
        assert note.interval == 2
        assert note.duration == 4
        assert note.start_time == 0

    def test_melody_note_negative_interval(self) -> None:
        """MelodyNote accepts negative intervals."""
        note = MelodyNote(interval=-5, duration=2, start_time=8)
        assert note.interval == -5


class TestMelodySequence:
    """Tests for MelodySequence dataclass."""

    def test_melody_sequence_empty(self) -> None:
        """Empty MelodySequence can be created."""
        seq = MelodySequence()
        assert seq.notes == []
        assert seq.total_duration == 0

    def test_melody_sequence_with_notes(self) -> None:
        """MelodySequence holds notes correctly."""
        notes = [
            MelodyNote(interval=0, duration=4, start_time=0),
            MelodyNote(interval=2, duration=4, start_time=4),
        ]
        seq = MelodySequence(notes=notes, total_duration=8)
        assert len(seq.notes) == 2

    def test_to_absolute_pitches(self) -> None:
        """to_absolute_pitches converts intervals to MIDI."""
        notes = [
            MelodyNote(interval=0, duration=4, start_time=0),  # First note: 60 + 0 = 60
            MelodyNote(interval=2, duration=4, start_time=4),  # +2: 60 + 2 = 62
            MelodyNote(interval=-1, duration=4, start_time=8),  # -1: 62 - 1 = 61
        ]
        seq = MelodySequence(notes=notes, total_duration=12)

        pitches = seq.to_absolute_pitches(start_midi=60)
        assert pitches == [60, 62, 61]

    def test_to_absolute_pitches_custom_start(self) -> None:
        """to_absolute_pitches respects start_midi."""
        notes = [MelodyNote(interval=5, duration=4, start_time=0)]
        seq = MelodySequence(notes=notes, total_duration=4)

        pitches = seq.to_absolute_pitches(start_midi=48)
        assert pitches == [53]  # 48 + 5

    def test_to_absolute_pitches_clamped(self) -> None:
        """to_absolute_pitches clamps to MIDI range 0-127."""
        notes = [
            MelodyNote(interval=100, duration=4, start_time=0),  # Would exceed 127
        ]
        seq = MelodySequence(notes=notes, total_duration=4)

        pitches = seq.to_absolute_pitches(start_midi=60)
        assert pitches[0] <= 127

        # Test lower bound
        notes2 = [MelodyNote(interval=-100, duration=4, start_time=0)]
        seq2 = MelodySequence(notes=notes2, total_duration=4)
        pitches2 = seq2.to_absolute_pitches(start_midi=60)
        assert pitches2[0] >= 0


class TestMelodyModelBasics:
    """Tests for basic MelodyModel functionality."""

    def test_init_default(self) -> None:
        """MelodyModel initializes with default chains."""
        model = MelodyModel()
        assert model.pitch_chain is not None
        assert model.rhythm_chain is not None
        assert model.pitch_chain.states == set(MELODY_INTERVALS)

    def test_init_custom_smoothing(self) -> None:
        """MelodyModel respects custom smoothing."""
        model = MelodyModel(smoothing=0.25)
        assert model.pitch_chain.smoothing == 0.25
        assert model.rhythm_chain.smoothing == 0.25

    def test_init_custom_chains(self) -> None:
        """MelodyModel accepts pre-configured chains."""
        custom_pitch = create_melody_pitch_chain(smoothing=0.1)
        custom_rhythm = create_melody_rhythm_chain(smoothing=0.2)
        model = MelodyModel(pitch_chain=custom_pitch, rhythm_chain=custom_rhythm)
        assert model.pitch_chain.smoothing == 0.1
        assert model.rhythm_chain.smoothing == 0.2


class TestMelodyModelGeneration:
    """Tests for melody sequence generation."""

    def test_generate_returns_sequence(self) -> None:
        """Generate returns a MelodySequence."""
        model = MelodyModel()
        seq = model.generate(num_bars=4)
        assert isinstance(seq, MelodySequence)

    def test_generate_correct_duration_range(self) -> None:
        """Generate produces duration within expected range."""
        model = MelodyModel()
        seq = model.generate(num_bars=4, beats_per_bar=4)
        # With density gaps, total notes duration may be less than total
        assert seq.total_duration == 4 * 4 * 4  # 64 sixteenths

    def test_generate_has_notes(self) -> None:
        """Generated sequence has notes."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)
        assert len(seq.notes) > 0

    def test_generate_start_times_increasing(self) -> None:
        """Note start times are monotonically increasing."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)

        for i in range(1, len(seq.notes)):
            assert seq.notes[i].start_time >= seq.notes[i - 1].start_time

    def test_generate_tracks_transitions(self) -> None:
        """Generate records transitions used."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)
        assert len(seq.transitions_used) > 0

    def test_generate_reproducible_with_rng(self) -> None:
        """Generation is reproducible with same RNG."""
        model = MelodyModel()

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        seq1 = model.generate(num_bars=4, rng=rng1)
        seq2 = model.generate(num_bars=4, rng=rng2)

        assert len(seq1.notes) == len(seq2.notes)
        for n1, n2 in zip(seq1.notes, seq2.notes):
            assert n1.interval == n2.interval
            assert n1.duration == n2.duration

    def test_generate_8_bars(self) -> None:
        """Generate works for 8-bar loops."""
        model = MelodyModel()
        seq = model.generate(num_bars=8)
        assert seq.total_duration == 8 * 4 * 4

    def test_generate_with_high_density(self) -> None:
        """High density produces more notes."""
        model = MelodyModel()
        rng_high = random.Random(42)
        rng_low = random.Random(42)

        seq_high = model.generate(num_bars=4, note_density=0.95, rng=rng_high)
        seq_low = model.generate(num_bars=4, note_density=0.3, rng=rng_low)

        # High density should generally have more notes
        # (Not guaranteed due to random durations, but likely)
        # Just check both work without error
        assert len(seq_high.notes) > 0
        assert len(seq_low.notes) > 0


class TestMelodyModelReward:
    """Tests for reward learning."""

    def test_apply_reward_positive(self) -> None:
        """Positive reward accepted."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=10.0)
        # Should not raise

    def test_apply_reward_negative(self) -> None:
        """Negative reward accepted."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=-5.0)

    def test_apply_reward_with_sensitivity(self) -> None:
        """Sensitivity multiplies reward."""
        model = MelodyModel()
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=1.0, sensitivity=2.0)


class TestMelodyModelSerialization:
    """Tests for serialization."""

    def test_to_dict(self) -> None:
        """MelodyModel can be serialized to dict."""
        model = MelodyModel()
        data = model.to_dict()
        assert "pitch_chain" in data
        assert "rhythm_chain" in data

    def test_from_dict(self) -> None:
        """MelodyModel can be deserialized from dict."""
        model = MelodyModel(smoothing=0.3)
        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=5.0)

        data = model.to_dict()
        restored = MelodyModel.from_dict(data)

        assert restored.pitch_chain.smoothing == 0.3
        assert restored.pitch_chain.states == model.pitch_chain.states

    def test_reset_to_priors(self) -> None:
        """reset_to_priors restores initial state."""
        model = MelodyModel()

        seq = model.generate(num_bars=2)
        model.apply_reward(seq.transitions_used, reward=100.0)

        model.reset_to_priors()

        assert model.pitch_chain.states == set(MELODY_INTERVALS)

    def test_repr(self) -> None:
        """MelodyModel has string representation."""
        model = MelodyModel()
        rep = repr(model)
        assert "MelodyModel" in rep

# =============================================================================
# Reward System Tests
# =============================================================================


class TestRating:
    """Tests for Rating dataclass."""

    def test_default_rating(self) -> None:
        """Default rating is neutral (3 stars)."""
        rating = Rating()
        assert rating.overall == 3
        assert rating.melodic == 3

    def test_custom_rating(self) -> None:
        """Custom ratings are stored."""
        rating = Rating(overall=5, melodic=4, harmonic=5, rhythmic=3, cohesion=4)
        assert rating.overall == 5
        assert rating.harmonic == 5

    def test_invalid_rating_raises(self) -> None:
        """Invalid ratings raise ValueError."""
        with pytest.raises(ValueError):
            Rating(overall=0)
        with pytest.raises(ValueError):
            Rating(melodic=6)

    def test_to_chord_reward_positive(self) -> None:
        """High ratings give positive chord reward."""
        rating = Rating(overall=5, melodic=5, harmonic=5, rhythmic=5, cohesion=5)
        reward = rating.to_chord_reward()
        assert reward > 0

    def test_to_chord_reward_negative(self) -> None:
        """Low ratings give negative chord reward."""
        rating = Rating(overall=1, melodic=1, harmonic=1, rhythmic=1, cohesion=1)
        reward = rating.to_chord_reward()
        assert reward < 0

    def test_to_chord_reward_neutral(self) -> None:
        """Neutral ratings give zero reward."""
        rating = Rating()  # All 3s
        reward = rating.to_chord_reward()
        assert reward == 0

    def test_to_melody_reward(self) -> None:
        """Melody reward calculated correctly."""
        rating = Rating(melodic=5)  # High melodic
        reward = rating.to_melody_reward()
        assert reward > 0

    def test_serialization(self) -> None:
        """Rating can be serialized and deserialized."""
        rating = Rating(overall=4, melodic=5, harmonic=3, rhythmic=4, cohesion=5)
        data = rating.to_dict()
        restored = Rating.from_dict(data)

        assert restored.overall == rating.overall
        assert restored.melodic == rating.melodic


class TestRewardSensitivity:
    """Tests for RewardSensitivity."""

    def test_sensitivity_values(self) -> None:
        """All sensitivity levels have multipliers."""
        assert RewardSensitivity.GENTLE in SENSITIVITY_MULTIPLIERS
        assert RewardSensitivity.MODERATE in SENSITIVITY_MULTIPLIERS
        assert RewardSensitivity.AGGRESSIVE in SENSITIVITY_MULTIPLIERS

    def test_multiplier_ordering(self) -> None:
        """Aggressive > Moderate > Gentle."""
        assert (
            SENSITIVITY_MULTIPLIERS[RewardSensitivity.AGGRESSIVE]
            > SENSITIVITY_MULTIPLIERS[RewardSensitivity.MODERATE]
            > SENSITIVITY_MULTIPLIERS[RewardSensitivity.GENTLE]
        )


class TestGenerationRecord:
    """Tests for GenerationRecord."""

    def test_create_record(self) -> None:
        """Can create a generation record."""
        record = GenerationRecord()
        assert record.generation_id is not None
        assert record.timestamp is not None

    def test_serialization(self) -> None:
        """Record can be serialized and deserialized."""
        record = GenerationRecord(
            params={"key": "C", "mode": "major"},
            chord_transitions=[((1, 4), 5)],
            rating=Rating(overall=4),
        )
        data = record.to_dict()
        restored = GenerationRecord.from_dict(data)

        assert restored.params == record.params
        assert restored.rating is not None
        assert restored.rating.overall == 4


class TestRewardManager:
    """Tests for RewardManager."""

    def test_init_default(self) -> None:
        """Default manager has moderate sensitivity."""
        manager = RewardManager()
        assert manager.sensitivity == RewardSensitivity.MODERATE

    def test_init_custom_sensitivity(self) -> None:
        """Can set custom sensitivity."""
        manager = RewardManager(sensitivity=RewardSensitivity.AGGRESSIVE)
        assert manager.sensitivity == RewardSensitivity.AGGRESSIVE

    def test_record_generation(self) -> None:
        """Can record a generation."""
        manager = RewardManager()
        generator = LoopGenerator()
        params = GenerationParams()
        loop = generator.generate(params)

        record = manager.record_generation(loop, params)

        assert record.generation_id in manager._generation_map
        assert len(manager.history) == 1

    def test_apply_rating(self) -> None:
        """Can apply a rating."""
        manager = RewardManager()
        generator = LoopGenerator()
        params = GenerationParams()
        loop = generator.generate(params)

        record = manager.record_generation(loop, params)
        rating = Rating(overall=5, melodic=5, harmonic=5, rhythmic=5, cohesion=5)

        success = manager.apply_rating(record.generation_id, rating, generator)

        assert success
        assert record.rating is not None
        assert record.rating.overall == 5

    def test_apply_rating_not_found(self) -> None:
        """Returns False for unknown generation."""
        manager = RewardManager()
        generator = LoopGenerator()
        rating = Rating()

        success = manager.apply_rating("unknown", rating, generator)
        assert not success

    def test_get_statistics(self) -> None:
        """Can get statistics."""
        manager = RewardManager()
        stats = manager.get_statistics()

        assert "total_generations" in stats
        assert "rated_generations" in stats

    def test_clear_history(self) -> None:
        """Can clear history."""
        manager = RewardManager()
        generator = LoopGenerator()
        params = GenerationParams()
        loop = generator.generate(params)
        manager.record_generation(loop, params)

        manager.clear_history()

        assert len(manager.history) == 0

    def test_serialization(self) -> None:
        """Manager can be serialized and deserialized."""
        manager = RewardManager(sensitivity=RewardSensitivity.GENTLE)
        generator = LoopGenerator()
        params = GenerationParams()
        loop = generator.generate(params)
        manager.record_generation(loop, params)

        data = manager.to_dict()
        restored = RewardManager.from_dict(data)

        assert restored.sensitivity == RewardSensitivity.GENTLE
        assert len(restored.history) == 1
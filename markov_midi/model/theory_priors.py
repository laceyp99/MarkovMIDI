"""
Music theory-based prior probabilities for MarkovMIDI.

Provides initial transition probabilities based on common patterns in
Western tonal music. These priors allow the model to generate reasonable
music before any training or reward learning.

The priors are represented as relative weights (not normalized probabilities)
that can be loaded into MarkovChain instances.
"""

from typing import Final

from markov_midi.model.markov_chain import MarkovChain


# =============================================================================
# Chord Progression Priors
# =============================================================================

# Chord degrees as Roman numerals (using integers for simplicity)
# 1=I, 2=ii, 3=iii, 4=IV, 5=V, 6=vi, 7=vii°
CHORD_DEGREES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)

# First-order chord transition weights
# Based on common progressions in popular/classical music
# Higher weight = more likely transition
CHORD_TRANSITION_WEIGHTS: Final[dict[int, dict[int, float]]] = {
    # From I (tonic) - can go anywhere, but IV and V are most common
    1: {1: 1.0, 2: 2.0, 3: 1.5, 4: 4.0, 5: 4.0, 6: 3.0, 7: 0.5},
    # From ii - typically goes to V or vii
    2: {1: 1.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 5.0, 6: 1.0, 7: 2.0},
    # From iii - often goes to vi, IV, or ii
    3: {1: 1.0, 2: 2.0, 3: 0.5, 4: 3.0, 5: 1.5, 6: 4.0, 7: 0.5},
    # From IV - strong tendency to V or I
    4: {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 5.0, 6: 1.5, 7: 1.0},
    # From V (dominant) - very strong pull to I (resolution)
    5: {1: 6.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 0.5, 6: 3.0, 7: 0.5},
    # From vi - often goes to ii, IV, or V
    6: {1: 2.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 3.0, 6: 0.5, 7: 1.0},
    # From vii° - almost always resolves to I
    7: {1: 6.0, 2: 1.0, 3: 2.0, 4: 1.0, 5: 1.5, 6: 1.0, 7: 0.3},
}

# Common second-order progressions (gives extra weight to classic patterns)
# (prev2, prev1) -> {next: extra_weight}
CHORD_SECOND_ORDER_BOOSTS: Final[dict[tuple[int, int], dict[int, float]]] = {
    # ii-V-I is very common (jazz/pop standard)
    (2, 5): {1: 4.0},
    # IV-V-I cadence (authentic)
    (4, 5): {1: 3.0},
    # I-IV-V (common opening)
    (1, 4): {5: 2.5},
    # === Pop/Contemporary Progressions ===
    # I-V-vi-IV (Axis of Awesome - most common pop progression)
    (1, 5): {6: 4.0},
    (5, 6): {4: 4.0},
    (6, 4): {1: 3.5, 5: 2.5},
    # vi-IV-I-V (same chords, different start)
    (4, 1): {5: 3.0},
    # I-vi-IV-V (50s/doo-wop progression)
    (1, 6): {4: 3.5},
    (6, 4): {5: 3.0},
    # I-IV-vi-V (another common variant)
    (1, 4): {6: 2.5},
    (4, 6): {5: 3.0},
    # vi-ii-V-I (circle progression snippet)
    (6, 2): {5: 2.5},
    # IV-I-V-vi (Pachelbel-style)
    (4, 1): {5: 2.0},
    (1, 5): {6: 2.0},
}

# Starting chord weights (first chord of loop)
# Pop songs commonly start on I, vi, or IV
CHORD_START_WEIGHTS: Final[dict[int, float]] = {
    1: 6.0,  # I - most common start (tonic establishes key)
    6: 4.0,  # vi - pop progressions often start here
    4: 3.0,  # IV - also common
    5: 1.0,  # V - rare start but possible
    2: 1.5,  # ii - occasionally
    3: 0.5,  # iii - rare
    7: 0.2,  # vii° - almost never starts a progression
}

# Ending chord weights (last chord - should prep resolution back to top)
# Since loops repeat, ending on V or IV creates tension that resolves to I
CHORD_END_WEIGHTS: Final[dict[int, float]] = {
    5: 5.0,  # V - dominant, strongest pull to I
    4: 4.0,  # IV - plagal prep, common in pop
    2: 2.5,  # ii - pre-dominant function
    6: 2.0,  # vi - creates nice loop back to I
    1: 1.0,  # I - already resolved (less tension)
    7: 1.5,  # vii° - leading tone to I
    3: 0.5,  # iii - weak ending
}


# =============================================================================
# Chord Rhythm Priors
# =============================================================================

# Rhythm durations in 16th notes (1=16th, 2=8th, 4=quarter, 8=half, 16=whole)
CHORD_RHYTHM_DURATIONS: Final[tuple[int, ...]] = (2, 4, 8, 16)

# First-order rhythm transition weights
# Encourages variety but prefers common note values
CHORD_RHYTHM_WEIGHTS: Final[dict[int, dict[int, float]]] = {
    # After 8th note - likely another 8th or a quarter
    2: {2: 3.0, 4: 4.0, 8: 2.0, 16: 0.5},
    # After quarter note - most common, can go to anything
    4: {2: 2.0, 4: 4.0, 8: 3.0, 16: 1.0},
    # After half note - often followed by quarter or half
    8: {2: 1.5, 4: 3.0, 8: 3.0, 16: 2.0},
    # After whole note - typically quarter or half follows
    16: {2: 1.0, 4: 3.0, 8: 3.0, 16: 1.5},
}


# =============================================================================
# Melody Pitch Priors (Scale-Degree Intervals)
# =============================================================================

# Intervals in scale degrees (relative encoding)
# Range: -7 (octave down) to +7 (octave up), plus 0 (repeat)
# These are SCALE STEPS, not semitones - ensures all notes stay in key
MELODY_INTERVALS: Final[tuple[int, ...]] = tuple(range(-7, 8))


# Interval weights - stepwise motion is most common in melodies
# Based on analysis of folk and pop melodies
def _build_interval_weights() -> dict[int, dict[int, float]]:
    """Build interval transition weights favoring stepwise motion.

    Intervals are in scale degrees:
    - 0 = unison (repeated note)
    - ±1 = step (most common)
    - ±2 = third
    - ±3 = fourth
    - ±4 = fifth
    - ±5-7 = larger leaps (sixth, seventh, octave)
    """
    weights: dict[int, dict[int, float]] = {}

    for prev_interval in MELODY_INTERVALS:
        weights[prev_interval] = {}
        for next_interval in MELODY_INTERVALS:
            # Base weight based on interval size
            abs_interval = abs(next_interval)
            if abs_interval == 0:
                w = 2.5  # Repeated notes - fairly common
            elif abs_interval == 1:
                w = 5.0  # Steps - most common in melodies
            elif abs_interval == 2:
                w = 3.5  # Thirds - common melodic movement
            elif abs_interval == 3:
                w = 2.0  # Fourths - less common
            elif abs_interval == 4:
                w = 1.5  # Fifths - occasional
            elif abs_interval == 5:
                w = 1.0  # Sixths - rare but used
            elif abs_interval == 6:
                w = 0.6  # Sevenths - rare
            else:  # abs_interval == 7
                w = 0.8  # Octave - occasional dramatic leap

            # Penalize consecutive large leaps in same direction
            if prev_interval != 0 and next_interval != 0:
                same_direction = (prev_interval > 0) == (next_interval > 0)
                if same_direction and abs(prev_interval) >= 3 and abs_interval >= 3:
                    w *= 0.25  # Strongly discourage

            # After a leap (3+), favor stepwise motion in opposite direction
            if abs(prev_interval) >= 3 and abs_interval <= 1:
                opposite = (prev_interval > 0) != (next_interval > 0)
                if opposite or next_interval == 0:
                    w *= 1.8  # Encourage resolution

            weights[prev_interval][next_interval] = w

    return weights


MELODY_INTERVAL_WEIGHTS: Final[dict[int, dict[int, float]]] = _build_interval_weights()


# =============================================================================
# Melody Rhythm Priors
# =============================================================================

# Melody rhythm durations (in 16th notes)
MELODY_RHYTHM_DURATIONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8)

# Melody rhythm weights - more variety than chords
MELODY_RHYTHM_WEIGHTS: Final[dict[int, dict[int, float]]] = {
    # After 16th note
    1: {1: 3.0, 2: 4.0, 3: 1.0, 4: 2.0, 6: 0.5, 8: 0.5},
    # After 8th note
    2: {1: 2.0, 2: 3.0, 3: 1.5, 4: 4.0, 6: 1.0, 8: 1.0},
    # After dotted 8th
    3: {1: 2.0, 2: 2.5, 3: 1.0, 4: 3.0, 6: 1.5, 8: 1.0},
    # After quarter note
    4: {1: 1.5, 2: 3.0, 3: 1.5, 4: 3.0, 6: 2.0, 8: 2.0},
    # After dotted quarter
    6: {1: 1.0, 2: 2.5, 3: 1.5, 4: 3.0, 6: 1.5, 8: 2.0},
    # After half note
    8: {1: 1.0, 2: 2.0, 3: 1.0, 4: 3.0, 6: 2.0, 8: 2.0},
}


# =============================================================================
# Factory Functions
# =============================================================================


def create_chord_chain(smoothing: float = 0.5) -> MarkovChain[int]:
    """
    Create a Markov chain for chord progressions initialized with theory priors.

    Args:
        smoothing: Laplace smoothing factor

    Returns:
        MarkovChain initialized with chord progression priors
    """
    chain: MarkovChain[int] = MarkovChain(
        smoothing=smoothing,
        states=set(CHORD_DEGREES),
    )

    # Set first-order priors
    for prev, nexts in CHORD_TRANSITION_WEIGHTS.items():
        for next_state, weight in nexts.items():
            chain.set_transition((prev,), next_state, weight)

    # Add second-order boosts
    for context, nexts in CHORD_SECOND_ORDER_BOOSTS.items():
        for next_state, boost in nexts.items():
            # Get base weight from first-order
            base = CHORD_TRANSITION_WEIGHTS.get(context[1], {}).get(next_state, 1.0)
            chain.set_transition(context, next_state, base + boost)

    return chain


def create_chord_rhythm_chain(smoothing: float = 0.5) -> MarkovChain[int]:
    """
    Create a Markov chain for chord rhythms initialized with theory priors.

    Args:
        smoothing: Laplace smoothing factor

    Returns:
        MarkovChain initialized with chord rhythm priors
    """
    chain: MarkovChain[int] = MarkovChain(
        smoothing=smoothing,
        states=set(CHORD_RHYTHM_DURATIONS),
    )

    for prev, nexts in CHORD_RHYTHM_WEIGHTS.items():
        for next_state, weight in nexts.items():
            chain.set_transition((prev,), next_state, weight)

    return chain


def create_melody_pitch_chain(smoothing: float = 0.5) -> MarkovChain[int]:
    """
    Create a Markov chain for melody intervals initialized with theory priors.

    Args:
        smoothing: Laplace smoothing factor

    Returns:
        MarkovChain initialized with melodic interval priors
    """
    chain: MarkovChain[int] = MarkovChain(
        smoothing=smoothing,
        states=set(MELODY_INTERVALS),
    )

    for prev, nexts in MELODY_INTERVAL_WEIGHTS.items():
        for next_state, weight in nexts.items():
            chain.set_transition((prev,), next_state, weight)

    return chain


def create_melody_rhythm_chain(smoothing: float = 0.5) -> MarkovChain[int]:
    """
    Create a Markov chain for melody rhythms initialized with theory priors.

    Args:
        smoothing: Laplace smoothing factor

    Returns:
        MarkovChain initialized with melody rhythm priors
    """
    chain: MarkovChain[int] = MarkovChain(
        smoothing=smoothing,
        states=set(MELODY_RHYTHM_DURATIONS),
    )

    for prev, nexts in MELODY_RHYTHM_WEIGHTS.items():
        for next_state, weight in nexts.items():
            chain.set_transition((prev,), next_state, weight)

    return chain


def create_all_chains(smoothing: float = 0.5) -> dict[str, MarkovChain[int]]:
    """
    Create all four Markov chains with theory priors.

    Args:
        smoothing: Laplace smoothing factor for all chains

    Returns:
        Dictionary with keys: 'chord', 'chord_rhythm', 'melody_pitch', 'melody_rhythm'
    """
    return {
        "chord": create_chord_chain(smoothing),
        "chord_rhythm": create_chord_rhythm_chain(smoothing),
        "melody_pitch": create_melody_pitch_chain(smoothing),
        "melody_rhythm": create_melody_rhythm_chain(smoothing),
    }

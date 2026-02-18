"""
Tests for the UI module (app.py).

These tests focus on the logic functions rather than the Gradio UI itself,
since Gradio components are difficult to unit test.
"""

import tempfile
from pathlib import Path

import pytest

from markov_midi.generator.loop_generator import GenerationParams, GeneratedLoop
from markov_midi.model.chord_model import ChordModel
from markov_midi.model.melody_model import MelodyModel
from markov_midi.model.reward import RewardManager
from markov_midi.ui.app import (
    AppState,
    apply_rating,
    create_fresh_state,
    create_new_session,
    density_to_notes_per_bar,
    format_chord_progression,
    format_header,
    format_loop_info,
    format_stats,
    generate_loop,
    load_existing_session,
    save_current_session,
    stars_display,
    write_midi_to_temp,
)


# -----------------------------------------------------------------------------
# AppState Tests
# -----------------------------------------------------------------------------


class TestAppState:
    """Tests for AppState dataclass."""

    def test_default_state(self) -> None:
        """Test default AppState values."""
        state = AppState()
        assert isinstance(state.chord_model, ChordModel)
        assert isinstance(state.melody_model, MelodyModel)
        assert isinstance(state.reward_manager, RewardManager)
        assert state.session_name == "Untitled Session"
        assert state.current_loop is None
        assert state.current_generation_id is None
        assert state.has_unsaved_changes is False

    def test_serialization_roundtrip(self) -> None:
        """Test state serialization and deserialization."""
        state = AppState()
        state.session_name = "Test Session"
        state.session_created = "2026-02-17T10:00:00"
        state.has_unsaved_changes = True

        data = state.to_dict()
        restored = AppState.from_dict(data)

        assert restored.session_name == "Test Session"
        assert restored.session_created == "2026-02-17T10:00:00"
        assert restored.has_unsaved_changes is True


class TestCreateFreshState:
    """Tests for create_fresh_state function."""

    def test_creates_valid_state(self) -> None:
        """Test fresh state has proper defaults."""
        state = create_fresh_state()
        assert isinstance(state, AppState)
        assert state.session_created != ""
        assert state.session_name == "Untitled Session"


# -----------------------------------------------------------------------------
# Density Conversion Tests
# -----------------------------------------------------------------------------


class TestDensityToNotesPerBar:
    """Tests for density_to_notes_per_bar function."""

    def test_low_density(self) -> None:
        """Test low density range."""
        min_density, max_density = density_to_notes_per_bar("Low")
        assert min_density == pytest.approx(0.3, abs=0.01)
        assert max_density == pytest.approx(0.5, abs=0.01)

    def test_medium_density(self) -> None:
        """Test medium density range."""
        min_density, max_density = density_to_notes_per_bar("Medium")
        assert min_density == pytest.approx(0.5, abs=0.01)
        assert max_density == pytest.approx(0.75, abs=0.01)

    def test_high_density(self) -> None:
        """Test high density range."""
        min_density, max_density = density_to_notes_per_bar("High")
        assert min_density == pytest.approx(0.8, abs=0.01)
        assert max_density == pytest.approx(1.0, abs=0.01)

    def test_default_is_medium(self) -> None:
        """Test unknown density defaults to medium."""
        min_density, max_density = density_to_notes_per_bar("Unknown")
        assert min_density == pytest.approx(0.5, abs=0.01)
        assert max_density == pytest.approx(0.75, abs=0.01)


# -----------------------------------------------------------------------------
# Generation Tests
# -----------------------------------------------------------------------------


class TestGenerateLoop:
    """Tests for generate_loop function."""

    def test_generates_loop(self) -> None:
        """Test basic loop generation."""
        state = create_fresh_state()
        state, result = generate_loop(
            state,
            key="C",
            mode="Major",
            bars=4,
            tempo=120,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )

        assert result is not None
        assert isinstance(result, GeneratedLoop)
        assert state.current_loop is result
        assert state.current_generation_id is not None

    def test_generates_with_arpeggios(self) -> None:
        """Test loop generation with arpeggios."""
        state = create_fresh_state()
        state, result = generate_loop(
            state,
            key="A",
            mode="Minor",
            bars=8,
            tempo=90,
            voicing="Arpeggiated",
            chord_complexity="Include 7ths",
            density="High",
        )

        assert result is not None
        # Check params were applied
        from markov_midi.generator.voicing import VoicingStyle

        assert result.params.voicing_style == VoicingStyle.ARPEGGIO_UP
        assert result.params.use_seventh_chords is True


class TestWriteMidiToTemp:
    """Tests for write_midi_to_temp function."""

    def test_writes_midi_file(self) -> None:
        """Test MIDI file is written to temp directory."""
        state = create_fresh_state()
        state, result = generate_loop(
            state,
            key="C",
            mode="Major",
            bars=4,
            tempo=120,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )

        midi_path = write_midi_to_temp(state, result)
        assert midi_path.exists()
        assert midi_path.suffix == ".mid"


# -----------------------------------------------------------------------------
# Rating Tests
# -----------------------------------------------------------------------------


class TestApplyRating:
    """Tests for apply_rating function."""

    def test_applies_rating(self) -> None:
        """Test rating is applied to current generation."""
        state = create_fresh_state()
        state, _ = generate_loop(
            state,
            key="C",
            mode="Major",
            bars=4,
            tempo=120,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )

        # Use new 5-parameter signature
        state = apply_rating(
            state, overall=5, melodic=4, harmonic=4, rhythmic=3, cohesion=5
        )

        assert state.has_unsaved_changes is True
        assert len(state.reward_manager.history) == 1
        assert state.reward_manager.history[0].rating is not None
        assert state.reward_manager.history[0].rating.overall == 5

    def test_no_op_without_loop(self) -> None:
        """Test rating does nothing without a current loop."""
        state = create_fresh_state()
        original_history_len = len(state.reward_manager.history)

        # Use new 5-parameter signature
        state = apply_rating(
            state, overall=5, melodic=4, harmonic=4, rhythmic=3, cohesion=5
        )

        assert len(state.reward_manager.history) == original_history_len


# -----------------------------------------------------------------------------
# Session Tests
# -----------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for session save/load functions."""

    def test_save_and_load_session(self) -> None:
        """Test saving and loading a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)

            # Create and save
            state = create_fresh_state()
            state.session_name = "TestSession"
            state, _ = generate_loop(
                state,
                key="G",
                mode="Major",
                bars=4,
                tempo=100,
                voicing="Block",
                chord_complexity="Triads Only",
                density="Low",
            )
            state = apply_rating(
                state, overall=4, melodic=4, harmonic=4, rhythmic=4, cohesion=4
            )
            state = save_current_session(state, session_dir)

            assert state.has_unsaved_changes is False

            # Load into new state
            new_state = create_fresh_state()
            new_state = load_existing_session(new_state, "TestSession", session_dir)

            assert new_state.session_name == "TestSession"
            assert len(new_state.reward_manager.history) == 1

    def test_create_new_session(self) -> None:
        """Test creating a new session."""
        state = create_fresh_state()
        state.has_unsaved_changes = True
        state.session_name = "OldSession"

        state = create_new_session(state, "NewSession")

        assert state.session_name == "NewSession"
        assert state.has_unsaved_changes is False
        assert len(state.reward_manager.history) == 0

    def test_create_new_session_default_name(self) -> None:
        """Test creating a new session with empty name."""
        state = create_fresh_state()
        state = create_new_session(state, "")

        assert state.session_name == "Untitled Session"


# -----------------------------------------------------------------------------
# Formatting Tests
# -----------------------------------------------------------------------------


class TestFormatChordProgression:
    """Tests for format_chord_progression function."""

    def test_formats_progression(self) -> None:
        """Test chord progression formatting."""
        state = create_fresh_state()
        state, result = generate_loop(
            state,
            key="C",
            mode="Major",
            bars=4,
            tempo=120,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )

        text = format_chord_progression(result)
        assert "->" in text
        assert len(text) > 0

    def test_formats_none(self) -> None:
        """Test formatting None result."""
        text = format_chord_progression(None)
        assert text == "No loop generated yet"


class TestFormatLoopInfo:
    """Tests for format_loop_info function."""

    def test_formats_info(self) -> None:
        """Test loop info formatting."""
        state = create_fresh_state()
        state, result = generate_loop(
            state,
            key="D",
            mode="Minor",
            bars=8,
            tempo=140,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )

        text = format_loop_info(result)
        assert "D" in text
        assert "Minor" in text
        assert "8 bars" in text
        assert "140" in text

    def test_formats_none(self) -> None:
        """Test formatting None result."""
        text = format_loop_info(None)
        assert text == ""


class TestFormatStats:
    """Tests for format_stats function."""

    def test_formats_empty_stats(self) -> None:
        """Test stats with no generations."""
        state = create_fresh_state()
        text = format_stats(state)
        assert "0 loops" in text
        assert "N/A" in text

    def test_formats_with_generations(self) -> None:
        """Test stats with generations."""
        state = create_fresh_state()
        state, _ = generate_loop(
            state,
            key="C",
            mode="Major",
            bars=4,
            tempo=120,
            voicing="Block",
            chord_complexity="Triads Only",
            density="Medium",
        )
        state = apply_rating(
            state, overall=4, melodic=4, harmonic=4, rhythmic=4, cohesion=4
        )

        text = format_stats(state)
        assert "1 loops" in text
        # Rating is 4, so average should be 4.0
        assert "4.0" in text


class TestFormatHeader:
    """Tests for format_header function."""

    def test_formats_header(self) -> None:
        """Test header formatting."""
        state = create_fresh_state()
        state.session_name = "MySession"
        state.has_unsaved_changes = False

        text = format_header(state)
        assert "MySession" in text
        assert "*" not in text

    def test_formats_header_unsaved(self) -> None:
        """Test header with unsaved changes."""
        state = create_fresh_state()
        state.session_name = "MySession"
        state.has_unsaved_changes = True

        text = format_header(state)
        assert "MySession" in text
        assert "*" in text


class TestStarsDisplay:
    """Tests for stars_display function."""

    def test_five_stars(self) -> None:
        """Test 5 star display."""
        text = stars_display(5)
        assert text.count("★") == 5
        assert "☆" not in text

    def test_three_stars(self) -> None:
        """Test 3 star display."""
        text = stars_display(3)
        assert text.count("★") == 3
        assert text.count("☆") == 2

    def test_one_star(self) -> None:
        """Test 1 star display."""
        text = stars_display(1)
        assert text.count("★") == 1
        assert text.count("☆") == 4


# -----------------------------------------------------------------------------
# Visualizer Tests
# -----------------------------------------------------------------------------


class TestVisualizerHelpers:
    """Tests for visualizer helper functions."""

    def test_midi_to_display_name_c4(self) -> None:
        """Test MIDI 60 converts to C4."""
        from markov_midi.ui.visualizer import midi_to_display_name

        result = midi_to_display_name(60)
        assert result == "C4"

    def test_midi_to_display_name_a4(self) -> None:
        """Test MIDI 69 converts to A4."""
        from markov_midi.ui.visualizer import midi_to_display_name

        result = midi_to_display_name(69)
        assert result == "A4"

    def test_midi_to_display_name_with_sharp(self) -> None:
        """Test MIDI 61 converts to C#4."""
        from markov_midi.ui.visualizer import midi_to_display_name

        result = midi_to_display_name(61)
        assert result == "C#4"

    def test_sixteenth_to_beats_four_sixteenths(self) -> None:
        """Test 4 sixteenths equals 1 beat."""
        from markov_midi.ui.visualizer import sixteenth_to_beats

        result = sixteenth_to_beats(4)
        assert result == 1.0

    def test_sixteenth_to_beats_one_sixteenth(self) -> None:
        """Test 1 sixteenth equals 0.25 beats."""
        from markov_midi.ui.visualizer import sixteenth_to_beats

        result = sixteenth_to_beats(1)
        assert result == 0.25

    def test_sixteenth_to_beats_sixteen_sixteenths(self) -> None:
        """Test 16 sixteenths equals 4 beats."""
        from markov_midi.ui.visualizer import sixteenth_to_beats

        result = sixteenth_to_beats(16)
        assert result == 4.0

    def test_get_pitch_range_with_notes(self) -> None:
        """Test pitch range calculation with notes."""
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.ui.visualizer import get_pitch_range

        notes = [
            VoicedNote(midi=60, start_time=0, duration=4, velocity=100),
            VoicedNote(midi=72, start_time=4, duration=4, velocity=100),
            VoicedNote(midi=65, start_time=8, duration=4, velocity=100),
        ]
        min_pitch, max_pitch = get_pitch_range(notes)
        assert min_pitch == 60
        assert max_pitch == 72

    def test_get_pitch_range_empty_notes(self) -> None:
        """Test pitch range returns default for empty list."""
        from markov_midi.ui.visualizer import get_pitch_range

        min_pitch, max_pitch = get_pitch_range([])
        assert min_pitch == 60
        assert max_pitch == 72

    def test_create_pitch_labels(self) -> None:
        """Test pitch labels are generated correctly."""
        from markov_midi.ui.visualizer import create_pitch_labels

        tick_vals, tick_labels = create_pitch_labels(60, 72)
        assert len(tick_vals) == len(tick_labels)
        assert len(tick_vals) > 0
        # C notes should be labeled
        assert "C4" in tick_labels or "C5" in tick_labels


class TestCreateNoteShapes:
    """Tests for create_note_shapes function."""

    def test_creates_shapes_for_notes(self) -> None:
        """Test that shapes are created for each note."""
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.ui.visualizer import create_note_shapes

        notes = [
            VoicedNote(midi=60, start_time=0, duration=4, velocity=100),
            VoicedNote(midi=64, start_time=4, duration=4, velocity=80),
        ]
        shapes = create_note_shapes(notes, "#ff6b00", row=1)
        assert len(shapes) == 2

    def test_shape_properties(self) -> None:
        """Test shape has correct properties."""
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.ui.visualizer import create_note_shapes

        notes = [VoicedNote(midi=60, start_time=0, duration=4, velocity=127)]
        shapes = create_note_shapes(notes, "#ff6b00", row=1)

        shape = shapes[0]
        assert shape["type"] == "rect"
        assert shape["x0"] == 0.0  # start time in beats
        assert shape["x1"] == 1.0  # end time in beats (4 sixteenths = 1 beat)
        assert shape["fillcolor"] == "#ff6b00"

    def test_velocity_affects_opacity(self) -> None:
        """Test that velocity affects opacity."""
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.ui.visualizer import create_note_shapes

        # Low velocity
        low_vel_notes = [VoicedNote(midi=60, start_time=0, duration=4, velocity=1)]
        low_shapes = create_note_shapes(low_vel_notes, "#ff6b00", row=1)

        # High velocity
        high_vel_notes = [VoicedNote(midi=60, start_time=0, duration=4, velocity=127)]
        high_shapes = create_note_shapes(high_vel_notes, "#ff6b00", row=1)

        assert low_shapes[0]["opacity"] < high_shapes[0]["opacity"]


class TestCreatePianoRollFigure:
    """Tests for create_piano_roll_figure function."""

    def test_creates_figure_with_data(self) -> None:
        """Test that a figure is created with loop data."""
        import plotly.graph_objects as go

        from markov_midi.generator.loop_generator import GeneratedLoop, GenerationParams
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.model.chord_model import ChordSequence
        from markov_midi.model.melody_model import MelodySequence
        from markov_midi.ui.visualizer import create_piano_roll_figure

        params = GenerationParams(
            key="C",
            mode="major",
            num_bars=4,
            tempo_bpm=120,
        )
        loop = GeneratedLoop(
            params=params,
            chord_sequence=ChordSequence(),
            melody_sequence=MelodySequence(),
            chord_notes=[
                VoicedNote(midi=48, start_time=0, duration=16, velocity=80),
                VoicedNote(midi=52, start_time=0, duration=16, velocity=80),
            ],
            melody_notes=[
                VoicedNote(midi=72, start_time=0, duration=4, velocity=100),
                VoicedNote(midi=74, start_time=4, duration=4, velocity=90),
            ],
        )

        fig = create_piano_roll_figure(loop)

        assert isinstance(fig, go.Figure)
        # Should have shapes for notes
        assert len(fig.layout.shapes) > 0

    def test_figure_has_two_subplots(self) -> None:
        """Test figure has melody and harmony subplots."""
        import plotly.graph_objects as go

        from markov_midi.generator.loop_generator import GeneratedLoop, GenerationParams
        from markov_midi.generator.voicing import VoicedNote
        from markov_midi.model.chord_model import ChordSequence
        from markov_midi.model.melody_model import MelodySequence
        from markov_midi.ui.visualizer import create_piano_roll_figure

        params = GenerationParams(
            key="C",
            mode="major",
            num_bars=4,
            tempo_bpm=120,
        )
        loop = GeneratedLoop(
            params=params,
            chord_sequence=ChordSequence(),
            melody_sequence=MelodySequence(),
            chord_notes=[VoicedNote(midi=48, start_time=0, duration=16, velocity=80)],
            melody_notes=[VoicedNote(midi=72, start_time=0, duration=4, velocity=100)],
        )

        fig = create_piano_roll_figure(loop)

        # Check for subplot annotations (titles)
        annotations = [a.text for a in fig.layout.annotations]
        assert "Melody" in annotations
        assert "Harmony" in annotations


class TestCreateEmptyFigure:
    """Tests for create_empty_figure function."""

    def test_creates_empty_figure(self) -> None:
        """Test that an empty placeholder figure is created."""
        import plotly.graph_objects as go

        from markov_midi.ui.visualizer import create_empty_figure

        fig = create_empty_figure()
        assert isinstance(fig, go.Figure)

    def test_has_placeholder_text(self) -> None:
        """Test empty figure has placeholder annotation."""
        from markov_midi.ui.visualizer import create_empty_figure

        fig = create_empty_figure()
        annotations = fig.layout.annotations
        assert len(annotations) == 1
        assert "Generate" in annotations[0].text

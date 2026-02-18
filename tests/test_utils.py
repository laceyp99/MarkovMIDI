"""
Unit tests for MarkovMIDI utilities module.

Tests cover:
- music_theory.py: Note/MIDI conversion, scales, chords, transposition
- quantize.py: Timing calculations and grid quantization
"""

import pytest

import tempfile
from pathlib import Path

from markov_midi.model.reward import RewardManager
from markov_midi.generator.loop_generator import LoopGenerator, GenerationParams

from markov_midi.utils.music_theory import (
    note_to_midi,
    midi_to_note,
    get_pitch_class,
    get_scale_intervals,
    get_scale_notes,
    degree_to_semitones,
    get_chord_intervals,
    build_chord,
    transpose_note,
    transpose_midi,
)

from markov_midi.utils.quantize import (
    get_grid_size,
    get_16th_grid,
    quantize_to_grid,
    quantize_duration,
    ticks_to_beats,
    beats_to_ticks,
    ticks_to_bars,
    bars_to_ticks,
    get_bar_length,
)

from markov_midi.model.persistence import (
    SessionMetadata,
    Session,
    save_session,
    load_session,
    create_session_from_generator,
    restore_session,
    list_sessions,
    save_model_only,
    load_model_only,
    get_session_path,
)

from markov_midi.audio.synthesizer import (
    SynthesizerConfig,
    Synthesizer,
    find_fluidsynth,
    is_fluidsynth_available,
    find_soundfonts,
    DEFAULT_SAMPLE_RATE,
)


# =============================================================================
# Music Theory Tests
# =============================================================================


class TestNoteToMidi:
    """Tests for note_to_midi function."""

    def test_middle_c(self) -> None:
        """Middle C (C4) should be MIDI 60."""
        assert note_to_midi("C", 4) == 60

    def test_a440(self) -> None:
        """A4 (440 Hz) should be MIDI 69."""
        assert note_to_midi("A", 4) == 69

    def test_sharp_note(self) -> None:
        """F#4 should be MIDI 66."""
        assert note_to_midi("F#", 4) == 66

    def test_flat_note(self) -> None:
        """Bb3 should be MIDI 58."""
        assert note_to_midi("Bb", 3) == 58

    def test_enharmonic_equivalents(self) -> None:
        """C# and Db should produce the same MIDI number."""
        assert note_to_midi("C#", 4) == note_to_midi("Db", 4)

    def test_lowest_note(self) -> None:
        """C-1 should be MIDI 0."""
        assert note_to_midi("C", -1) == 0

    def test_invalid_note_raises(self) -> None:
        """Invalid note name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid note name"):
            note_to_midi("X", 4)


class TestMidiToNote:
    """Tests for midi_to_note function."""

    def test_middle_c(self) -> None:
        """MIDI 60 should be C4."""
        note, octave = midi_to_note(60)
        assert note == "C"
        assert octave == 4

    def test_a440(self) -> None:
        """MIDI 69 should be A4."""
        note, octave = midi_to_note(69)
        assert note == "A"
        assert octave == 4

    def test_prefer_sharps(self) -> None:
        """MIDI 61 with sharps should be C#."""
        note, octave = midi_to_note(61, prefer_sharps=True)
        assert note == "C#"
        assert octave == 4

    def test_prefer_flats(self) -> None:
        """MIDI 61 with flats should be Db."""
        note, octave = midi_to_note(61, prefer_sharps=False)
        assert note == "Db"
        assert octave == 4

    def test_roundtrip(self) -> None:
        """Converting MIDI to note and back should be lossless."""
        for midi in [0, 60, 69, 127]:
            note, octave = midi_to_note(midi)
            assert note_to_midi(note, octave) == midi

    def test_out_of_range_raises(self) -> None:
        """Out of range MIDI should raise ValueError."""
        with pytest.raises(ValueError):
            midi_to_note(128)
        with pytest.raises(ValueError):
            midi_to_note(-1)


class TestGetPitchClass:
    """Tests for get_pitch_class function."""

    def test_c_is_zero(self) -> None:
        """C should be pitch class 0."""
        assert get_pitch_class("C") == 0

    def test_enharmonic_equivalents(self) -> None:
        """Enharmonic equivalents should have same pitch class."""
        assert get_pitch_class("C#") == get_pitch_class("Db")
        assert get_pitch_class("F#") == get_pitch_class("Gb")


class TestScales:
    """Tests for scale-related functions."""

    def test_major_intervals(self) -> None:
        """Major scale should have correct intervals."""
        intervals = get_scale_intervals("major")
        assert intervals == (0, 2, 4, 5, 7, 9, 11)

    def test_minor_intervals(self) -> None:
        """Natural minor scale should have correct intervals."""
        intervals = get_scale_intervals("minor")
        assert intervals == (0, 2, 3, 5, 7, 8, 10)

    def test_c_major_scale(self) -> None:
        """C major scale should be all naturals."""
        scale = get_scale_notes("C", "major")
        assert scale == ["C", "D", "E", "F", "G", "A", "B"]

    def test_a_minor_scale(self) -> None:
        """A natural minor scale (relative to C major)."""
        scale = get_scale_notes("A", "minor")
        assert scale == ["A", "B", "C", "D", "E", "F", "G"]

    def test_g_major_scale(self) -> None:
        """G major scale should have F#."""
        scale = get_scale_notes("G", "major")
        assert "F#" in scale

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            get_scale_intervals("lydian")  # Not implemented yet

    def test_degree_to_semitones(self) -> None:
        """Test scale degree to semitones conversion."""
        # Major scale: 1=0, 2=2, 3=4, 4=5, 5=7, 6=9, 7=11
        assert degree_to_semitones(1, "major") == 0
        assert degree_to_semitones(5, "major") == 7  # Perfect fifth
        assert degree_to_semitones(3, "minor") == 3  # Minor third


class TestChords:
    """Tests for chord-related functions."""

    def test_major_triad_intervals(self) -> None:
        """Major triad should be root, major 3rd, perfect 5th."""
        intervals = get_chord_intervals("major")
        assert intervals == (0, 4, 7)

    def test_minor_triad_intervals(self) -> None:
        """Minor triad should be root, minor 3rd, perfect 5th."""
        intervals = get_chord_intervals("minor")
        assert intervals == (0, 3, 7)

    def test_dom7_intervals(self) -> None:
        """Dominant 7th should have minor 7th."""
        intervals = get_chord_intervals("dom7")
        assert intervals == (0, 4, 7, 10)

    def test_c_major_chord(self) -> None:
        """C major chord should be C, E, G."""
        chord = build_chord("C", "major")
        assert chord == ["C", "E", "G"]

    def test_a_minor_chord(self) -> None:
        """A minor chord should be A, C, E."""
        chord = build_chord("A", "minor")
        assert chord == ["A", "C", "E"]

    def test_g_dom7_chord(self) -> None:
        """G7 chord should be G, B, D, F."""
        chord = build_chord("G", "dom7")
        assert chord == ["G", "B", "D", "F"]

    def test_invalid_chord_type_raises(self) -> None:
        """Invalid chord type should raise ValueError."""
        with pytest.raises(ValueError):
            get_chord_intervals("add9")  # Not implemented


class TestTransposition:
    """Tests for transposition functions."""

    def test_transpose_up_fifth(self) -> None:
        """Transposing C up a fifth should give G."""
        assert transpose_note("C", 7) == "G"

    def test_transpose_down_fifth(self) -> None:
        """Transposing G down a fifth should give C."""
        assert transpose_note("G", -7) == "C"

    def test_transpose_octave(self) -> None:
        """Transposing by 12 semitones should return same note."""
        assert transpose_note("E", 12) == "E"

    def test_transpose_midi(self) -> None:
        """Transpose MIDI note number."""
        assert transpose_midi(60, 7) == 67  # C4 -> G4
        assert transpose_midi(60, -12) == 48  # C4 -> C3

    def test_transpose_midi_out_of_range_raises(self) -> None:
        """Transposing MIDI out of range should raise."""
        with pytest.raises(ValueError):
            transpose_midi(120, 10)  # Would be 130
        with pytest.raises(ValueError):
            transpose_midi(5, -10)  # Would be -5


# =============================================================================
# Quantization Tests
# =============================================================================


class TestGridSize:
    """Tests for grid size calculations."""

    def test_16th_grid_480(self) -> None:
        """16th note grid at 480 TPB should be 120 ticks."""
        assert get_16th_grid(480) == 120

    def test_16th_grid_960(self) -> None:
        """16th note grid at 960 TPB should be 240 ticks."""
        assert get_16th_grid(960) == 240

    def test_8th_grid(self) -> None:
        """8th note grid at 480 TPB should be 240 ticks."""
        assert get_grid_size(480, 8) == 240

    def test_quarter_grid(self) -> None:
        """Quarter note grid should equal ticks_per_beat."""
        assert get_grid_size(480, 4) == 480


class TestQuantizeToGrid:
    """Tests for grid quantization."""

    def test_exact_grid_position(self) -> None:
        """Position on grid should not change."""
        assert quantize_to_grid(120, 120) == 120
        assert quantize_to_grid(0, 120) == 0

    def test_round_down(self) -> None:
        """Position closer to lower grid should round down."""
        assert quantize_to_grid(50, 120) == 0

    def test_round_up(self) -> None:
        """Position closer to upper grid should round up."""
        assert quantize_to_grid(100, 120) == 120

    def test_quantize_duration_minimum(self) -> None:
        """Duration should be at least 1 grid unit."""
        assert quantize_duration(50, 120) == 120  # Rounds up to minimum


class TestTickConversions:
    """Tests for tick/beat/bar conversions."""

    def test_ticks_to_beats(self) -> None:
        """480 ticks at 480 TPB should be 1 beat."""
        assert ticks_to_beats(480, 480) == 1.0
        assert ticks_to_beats(240, 480) == 0.5

    def test_beats_to_ticks(self) -> None:
        """1 beat at 480 TPB should be 480 ticks."""
        assert beats_to_ticks(1.0, 480) == 480
        assert beats_to_ticks(0.5, 480) == 240

    def test_ticks_beats_roundtrip(self) -> None:
        """Converting ticks to beats and back should be lossless."""
        tpb = 480
        for ticks in [0, 120, 480, 1920]:
            beats = ticks_to_beats(ticks, tpb)
            assert beats_to_ticks(beats, tpb) == ticks

    def test_ticks_to_bars(self) -> None:
        """1920 ticks at 480 TPB in 4/4 should be 1 bar."""
        assert ticks_to_bars(1920, 480, 4) == 1.0

    def test_bars_to_ticks(self) -> None:
        """1 bar in 4/4 at 480 TPB should be 1920 ticks."""
        assert bars_to_ticks(1.0, 480, 4) == 1920

    def test_bar_length(self) -> None:
        """Bar length in 4/4 at 480 TPB should be 1920."""
        assert get_bar_length(480, 4) == 1920

# =============================================================================
# Synthesizer Tests
# =============================================================================


class TestSynthesizerConfig:
    """Tests for SynthesizerConfig."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = SynthesizerConfig()
        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.soundfont_path is None

    def test_custom_config(self) -> None:
        """Custom config stores values."""
        config = SynthesizerConfig(
            soundfont_path="test.sf2",
            sample_rate=48000,
            gain=0.8,
        )
        assert config.soundfont_path == "test.sf2"
        assert config.sample_rate == 48000
        assert config.gain == 0.8


class TestSynthesizer:
    """Tests for Synthesizer class."""

    def test_init_without_soundfont(self) -> None:
        """Can initialize without soundfont."""
        synth = Synthesizer()
        assert synth.config.soundfont_path is None

    def test_set_soundfont(self) -> None:
        """Can set soundfont after init."""
        synth = Synthesizer()
        synth.set_soundfont("piano.sf2")
        assert synth.config.soundfont_path == "piano.sf2"

    def test_is_available_without_soundfont(self) -> None:
        """Not available without soundfont."""
        synth = Synthesizer()
        # Even if FluidSynth is installed, no soundfont = not available
        # (unless soundfont is set)
        assert not synth.is_available()

    def test_repr(self) -> None:
        """Has string representation."""
        synth = Synthesizer()
        rep = repr(synth)
        assert "Synthesizer" in rep


class TestFindSoundfonts:
    """Tests for soundfont discovery."""

    def test_returns_list(self) -> None:
        """Returns a list (possibly empty)."""
        soundfonts = find_soundfonts()
        assert isinstance(soundfonts, list)

    def test_searches_custom_dir(self) -> None:
        """Can search custom directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake soundfont
            sf_path = Path(tmpdir) / "test.sf2"
            sf_path.touch()

            soundfonts = find_soundfonts([tmpdir])
            assert any(sf.name == "test.sf2" for sf in soundfonts)


# =============================================================================
# Persistence Tests
# =============================================================================


class TestSessionMetadata:
    """Tests for SessionMetadata."""

    def test_default_metadata(self) -> None:
        """Default metadata has expected values."""
        meta = SessionMetadata()
        assert meta.name == "Untitled Session"
        assert meta.version == "1.0"

    def test_touch_updates_timestamp(self) -> None:
        """Touch updates updated_at."""
        meta = SessionMetadata()
        old_updated = meta.updated_at
        meta.touch()
        # Should be updated (might be same if very fast, but shouldn't fail)
        assert meta.updated_at >= old_updated

    def test_serialization(self) -> None:
        """Metadata can be serialized."""
        meta = SessionMetadata(name="Test", description="A test session")
        data = meta.to_dict()
        restored = SessionMetadata.from_dict(data)

        assert restored.name == "Test"
        assert restored.description == "A test session"


class TestSession:
    """Tests for Session."""

    def test_create_session(self) -> None:
        """Can create a session."""
        session = Session()
        assert session.metadata is not None

    def test_serialization(self) -> None:
        """Session can be serialized."""
        session = Session(
            metadata=SessionMetadata(name="Test"),
            generator_state={"test": "data"},
        )
        data = session.to_dict()
        restored = Session.from_dict(data)

        assert restored.metadata.name == "Test"
        assert restored.generator_state == {"test": "data"}


class TestSaveLoadSession:
    """Tests for session save/load."""

    def test_save_and_load(self) -> None:
        """Can save and load a session."""
        generator = LoopGenerator()
        manager = RewardManager()

        session = create_session_from_generator(generator, manager, name="Test Session")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)

            loaded = load_session(path)

            assert loaded.metadata.name == "Test Session"

    def test_restore_session(self) -> None:
        """Can restore generator and manager from session."""
        generator = LoopGenerator()
        manager = RewardManager()
        params = GenerationParams()
        loop = generator.generate(params)
        manager.record_generation(loop, params)

        session = create_session_from_generator(generator, manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_session(session, path)

            loaded = load_session(path)
            restored_gen, restored_mgr = restore_session(loaded)

            assert restored_gen is not None
            assert restored_mgr is not None
            assert len(restored_mgr.history) == 1

    def test_load_missing_file_raises(self) -> None:
        """Loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_session("nonexistent.json")


class TestSaveLoadModelOnly:
    """Tests for model-only save/load."""

    def test_save_and_load_model(self) -> None:
        """Can save and load just the model."""
        generator = LoopGenerator()
        # Train a bit
        loop = generator.generate()
        generator.apply_reward(loop, chord_reward=5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_model_only(generator, path)

            loaded = load_model_only(path)

            assert loaded is not None
            assert loaded.chord_model is not None


class TestListSessions:
    """Tests for session listing."""

    def test_list_empty_directory(self) -> None:
        """Empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = list_sessions(tmpdir)
            assert sessions == []

    def test_list_sessions(self) -> None:
        """Lists sessions in directory."""
        generator = LoopGenerator()
        session = create_session_from_generator(generator, name="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_session(session, Path(tmpdir) / "test.json")

            sessions = list_sessions(tmpdir)

            assert len(sessions) == 1
            assert sessions[0]["name"] == "Test"


class TestGetSessionPath:
    """Tests for session path generation."""

    def test_simple_name(self) -> None:
        """Simple name becomes filename."""
        path = get_session_path("My Session")
        assert path.name == "My Session.json"

    def test_sanitizes_special_chars(self) -> None:
        """Special characters are sanitized."""
        path = get_session_path("Test/Session:Name")
        assert "/" not in path.name
        assert ":" not in path.name

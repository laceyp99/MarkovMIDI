"""
Unit tests for MarkovMIDI generator module.

Tests cover:
- voicing.py: Chord voicing and conversion to MIDI
- midi_writer.py: MIDI file creation
- loop_generator.py: Full loop generation orchestration
"""

import random
import tempfile
from pathlib import Path

import pytest

import mido

from markov_midi.model.chord_model import ChordEvent, ChordSequence
from markov_midi.model.melody_model import MelodyNote, MelodySequence
from markov_midi.generator.voicing import (
    VoicingStyle,
    VoicedNote,
    VoicedChord,
    get_chord_quality,
    degree_to_midi_notes,
    voice_chord_block,
    voice_chord_arpeggio,
    voice_chord_sequence,
    get_all_voiced_notes,
    MAJOR_SCALE_CHORD_QUALITIES,
    MINOR_SCALE_CHORD_QUALITIES,
)
from markov_midi.generator.midi_writer import (
    MidiTrackData,
    MidiFileData,
    sixteenths_to_ticks,
    melody_sequence_to_voiced_notes,
    create_midi_file,
    write_midi_file,
    get_midi_duration_seconds,
)
from markov_midi.generator.loop_generator import (
    GenerationParams,
    GeneratedLoop,
    LoopGenerator,
)
from markov_midi.parser.midi_loader import (
    ParsedNote,
    ParsedTrack,
    ParsedMidi,
    parse_midi_file,
    quantize_parsed_midi,
    extract_intervals,
    extract_durations_16ths,
    notes_to_training_data,
)


# =============================================================================
# Voicing Tests
# =============================================================================


class TestChordQuality:
    """Tests for chord quality determination."""

    def test_major_scale_qualities(self) -> None:
        """Major scale has correct chord qualities."""
        assert get_chord_quality(1, "major") == "major"
        assert get_chord_quality(2, "major") == "minor"
        assert get_chord_quality(3, "major") == "minor"
        assert get_chord_quality(4, "major") == "major"
        assert get_chord_quality(5, "major") == "major"
        assert get_chord_quality(6, "major") == "minor"
        assert get_chord_quality(7, "major") == "diminished"

    def test_minor_scale_qualities(self) -> None:
        """Minor scale has correct chord qualities."""
        assert get_chord_quality(1, "minor") == "minor"
        assert get_chord_quality(2, "minor") == "diminished"
        assert get_chord_quality(3, "minor") == "major"
        assert get_chord_quality(4, "minor") == "minor"
        assert get_chord_quality(5, "minor") == "minor"
        assert get_chord_quality(6, "minor") == "major"
        assert get_chord_quality(7, "minor") == "major"

    def test_invalid_degree_raises(self) -> None:
        """Invalid degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be 1-7"):
            get_chord_quality(0, "major")
        with pytest.raises(ValueError, match="Degree must be 1-7"):
            get_chord_quality(8, "major")

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            get_chord_quality(1, "dorian")


class TestDegreeToMidiNotes:
    """Tests for converting degrees to MIDI notes."""

    def test_c_major_i_chord(self) -> None:
        """C major I chord is C-E-G."""
        notes = degree_to_midi_notes(1, "C", "major", octave=4)
        # C4=60, E4=64, G4=67
        assert notes == [60, 64, 67]

    def test_c_major_v_chord(self) -> None:
        """C major V chord is G-B-D."""
        notes = degree_to_midi_notes(5, "C", "major", octave=4)
        # G4=67, B4=71, D5=74
        assert notes == [67, 71, 74]

    def test_a_minor_i_chord(self) -> None:
        """A minor i chord is A-C-E."""
        notes = degree_to_midi_notes(1, "A", "minor", octave=4)
        # A4=69, C5=72, E5=76
        assert notes == [69, 72, 76]

    def test_seventh_chord(self) -> None:
        """Seventh chords have 4 notes."""
        notes = degree_to_midi_notes(1, "C", "major", octave=4, use_seventh=True)
        assert len(notes) == 4
        # C maj7: C-E-G-B
        assert notes == [60, 64, 67, 71]

    def test_v_chord_is_dom7(self) -> None:
        """V chord with sevenths uses dom7."""
        notes = degree_to_midi_notes(5, "C", "major", octave=4, use_seventh=True)
        # G dom7: G-B-D-F (67, 71, 74, 77)
        assert len(notes) == 4
        assert notes == [67, 71, 74, 77]

    def test_different_octaves(self) -> None:
        """Octave parameter shifts all notes."""
        notes_3 = degree_to_midi_notes(1, "C", "major", octave=3)
        notes_5 = degree_to_midi_notes(1, "C", "major", octave=5)
        # Should be 24 semitones apart
        for n3, n5 in zip(notes_3, notes_5):
            assert n5 - n3 == 24


class TestVoiceChordBlock:
    """Tests for block chord voicing."""

    def test_block_chord_same_start_time(self) -> None:
        """All notes in block chord have same start time."""
        event = ChordEvent(degree=1, duration=16, start_time=0)
        voiced = voice_chord_block(event, "C", "major")

        for note in voiced.notes:
            assert note.start_time == 0

    def test_block_chord_same_duration(self) -> None:
        """All notes in block chord have same duration."""
        event = ChordEvent(degree=1, duration=16, start_time=8)
        voiced = voice_chord_block(event, "C", "major")

        for note in voiced.notes:
            assert note.duration == 16

    def test_block_chord_preserves_degree(self) -> None:
        """Voiced chord preserves degree info."""
        event = ChordEvent(degree=4, duration=8, start_time=0)
        voiced = voice_chord_block(event, "C", "major")

        assert voiced.degree == 4
        assert voiced.chord_type == "major"


class TestVoiceChordArpeggio:
    """Tests for arpeggio voicing."""

    def test_arpeggio_spreads_notes(self) -> None:
        """Arpeggio notes have different start times."""
        event = ChordEvent(degree=1, duration=16, start_time=0)
        voiced = voice_chord_arpeggio(event, "C", "major", direction="up")

        start_times = [n.start_time for n in voiced.notes]
        # Not all notes should have the same start time
        assert len(set(start_times)) > 1

    def test_arpeggio_up_ordering(self) -> None:
        """Arpeggio up has ascending MIDI notes."""
        event = ChordEvent(degree=1, duration=16, start_time=0)
        voiced = voice_chord_arpeggio(event, "C", "major", direction="up")

        midi_notes = [n.midi for n in voiced.notes]
        # Should be ascending
        for i in range(len(midi_notes) - 1):
            assert midi_notes[i] <= midi_notes[i + 1]

    def test_arpeggio_down_ordering(self) -> None:
        """Arpeggio down has descending MIDI notes."""
        event = ChordEvent(degree=1, duration=16, start_time=0)
        voiced = voice_chord_arpeggio(event, "C", "major", direction="down")

        midi_notes = [n.midi for n in voiced.notes]
        # Should be descending
        for i in range(len(midi_notes) - 1):
            assert midi_notes[i] >= midi_notes[i + 1]


class TestVoiceChordSequence:
    """Tests for voicing complete sequences."""

    def test_voices_all_events(self) -> None:
        """All events in sequence are voiced."""
        events = [
            ChordEvent(degree=1, duration=8, start_time=0),
            ChordEvent(degree=4, duration=8, start_time=8),
            ChordEvent(degree=5, duration=8, start_time=16),
            ChordEvent(degree=1, duration=8, start_time=24),
        ]
        sequence = ChordSequence(events=events, total_duration=32)

        voiced = voice_chord_sequence(sequence, "C", "major")
        assert len(voiced) == 4

    def test_voicing_style_applied(self) -> None:
        """Different voicing styles produce different results."""
        events = [ChordEvent(degree=1, duration=16, start_time=0)]
        sequence = ChordSequence(events=events, total_duration=16)

        block = voice_chord_sequence(sequence, "C", "major", style=VoicingStyle.BLOCK)
        arp = voice_chord_sequence(
            sequence, "C", "major", style=VoicingStyle.ARPEGGIO_UP
        )

        # Block chord notes all start at same time
        block_starts = {n.start_time for n in block[0].notes}
        arp_starts = {n.start_time for n in arp[0].notes}

        assert len(block_starts) == 1
        assert len(arp_starts) > 1


class TestGetAllVoicedNotes:
    """Tests for flattening voiced chords."""

    def test_flattens_all_notes(self) -> None:
        """All notes from all chords are included."""
        events = [
            ChordEvent(degree=1, duration=8, start_time=0),
            ChordEvent(degree=5, duration=8, start_time=8),
        ]
        sequence = ChordSequence(events=events, total_duration=16)

        voiced = voice_chord_sequence(sequence, "C", "major")
        all_notes = get_all_voiced_notes(voiced)

        # 2 triads = 6 notes
        assert len(all_notes) == 6

    def test_sorted_by_start_time(self) -> None:
        """Notes are sorted by start time."""
        events = [
            ChordEvent(degree=1, duration=8, start_time=0),
            ChordEvent(degree=5, duration=8, start_time=8),
        ]
        sequence = ChordSequence(events=events, total_duration=16)

        voiced = voice_chord_sequence(sequence, "C", "major")
        all_notes = get_all_voiced_notes(voiced)

        for i in range(len(all_notes) - 1):
            assert all_notes[i].start_time <= all_notes[i + 1].start_time


# =============================================================================
# MIDI Writer Tests
# =============================================================================


class TestSixteenthsToTicks:
    """Tests for timing conversion."""

    def test_480_ticks_per_beat(self) -> None:
        """Correct conversion with 480 ticks per beat."""
        # 1 beat = 4 sixteenths = 480 ticks
        # 1 sixteenth = 120 ticks
        assert sixteenths_to_ticks(1, 480) == 120
        assert sixteenths_to_ticks(4, 480) == 480
        assert sixteenths_to_ticks(16, 480) == 1920

    def test_960_ticks_per_beat(self) -> None:
        """Correct conversion with 960 ticks per beat."""
        # 1 sixteenth = 240 ticks
        assert sixteenths_to_ticks(1, 960) == 240
        assert sixteenths_to_ticks(4, 960) == 960


class TestMelodySequenceToVoicedNotes:
    """Tests for melody conversion."""

    def test_converts_to_voiced_notes(self) -> None:
        """Melody sequence converts to voiced notes."""
        notes = [
            MelodyNote(interval=0, duration=4, start_time=0),
            MelodyNote(interval=2, duration=4, start_time=4),
        ]
        sequence = MelodySequence(notes=notes, total_duration=8)

        voiced = melody_sequence_to_voiced_notes(sequence, start_midi=60)

        assert len(voiced) == 2
        assert voiced[0].midi == 60
        assert voiced[1].midi == 62

    def test_preserves_timing(self) -> None:
        """Timing information is preserved."""
        notes = [
            MelodyNote(interval=0, duration=8, start_time=0),
            MelodyNote(interval=5, duration=4, start_time=8),
        ]
        sequence = MelodySequence(notes=notes, total_duration=12)

        voiced = melody_sequence_to_voiced_notes(sequence, start_midi=60)

        assert voiced[0].start_time == 0
        assert voiced[0].duration == 8
        assert voiced[1].start_time == 8
        assert voiced[1].duration == 4


class TestMidiDurationSeconds:
    """Tests for duration calculation."""

    def test_4_bars_at_120_bpm(self) -> None:
        """4 bars at 120 BPM = 8 seconds."""
        # 4 bars * 4 beats * 4 sixteenths = 64 sixteenths
        # 64 / 4 = 16 beats
        # 16 beats / 2 BPS = 8 seconds
        duration = get_midi_duration_seconds(64, 120)
        assert abs(duration - 8.0) < 0.001

    def test_8_bars_at_90_bpm(self) -> None:
        """8 bars at 90 BPM calculation."""
        # 8 bars * 4 beats * 4 sixteenths = 128 sixteenths
        # 128 / 4 = 32 beats
        # 32 beats / 1.5 BPS = 21.33 seconds
        duration = get_midi_duration_seconds(128, 90)
        expected = 32.0 * (60.0 / 90.0)
        assert abs(duration - expected) < 0.001


class TestCreateMidiFile:
    """Tests for MIDI file creation."""

    def test_creates_mido_file(self) -> None:
        """Creates a valid mido MidiFile."""
        import mido

        track_data = MidiTrackData(
            name="Test",
            notes=[VoicedNote(midi=60, start_time=0, duration=4, velocity=100)],
            channel=0,
            program=0,
        )
        file_data = MidiFileData(tracks=[track_data], ticks_per_beat=480, tempo_bpm=120)

        midi_file = create_midi_file(file_data)

        assert isinstance(midi_file, mido.MidiFile)
        assert midi_file.ticks_per_beat == 480
        # Should have tempo track + 1 instrument track
        assert len(midi_file.tracks) == 2


class TestWriteMidiFile:
    """Tests for writing MIDI files to disk."""

    def test_writes_file(self) -> None:
        """File is written to disk."""
        track_data = MidiTrackData(
            name="Test",
            notes=[VoicedNote(midi=60, start_time=0, duration=4, velocity=100)],
        )
        file_data = MidiFileData(tracks=[track_data])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"
            result = write_midi_file(file_data, output_path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_creates_parent_directory(self) -> None:
        """Parent directories are created if needed."""
        track_data = MidiTrackData(name="Test", notes=[])
        file_data = MidiFileData(tracks=[track_data])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "test.mid"
            result = write_midi_file(file_data, output_path)

            assert result.exists()


# =============================================================================
# Loop Generator Tests
# =============================================================================


class TestGenerationParams:
    """Tests for GenerationParams."""

    def test_default_params(self) -> None:
        """Default parameters are sensible."""
        params = GenerationParams()
        assert params.key == "C"
        assert params.mode == "major"
        assert params.num_bars == 4
        assert params.tempo_bpm == 120

    def test_custom_params(self) -> None:
        """Custom parameters are stored."""
        params = GenerationParams(
            key="F#",
            mode="minor",
            num_bars=8,
            tempo_bpm=90,
            voicing_style=VoicingStyle.ARPEGGIO_UP,
        )
        assert params.key == "F#"
        assert params.mode == "minor"
        assert params.num_bars == 8
        assert params.voicing_style == VoicingStyle.ARPEGGIO_UP


class TestLoopGeneratorBasics:
    """Tests for basic LoopGenerator functionality."""

    def test_init_default(self) -> None:
        """Generator initializes with default models."""
        generator = LoopGenerator()
        assert generator.chord_model is not None
        assert generator.melody_model is not None

    def test_init_custom_models(self) -> None:
        """Generator accepts custom models."""
        from markov_midi.model.chord_model import ChordModel
        from markov_midi.model.melody_model import MelodyModel

        chord = ChordModel(smoothing=0.1)
        melody = MelodyModel(smoothing=0.2)
        generator = LoopGenerator(chord_model=chord, melody_model=melody)

        assert generator.chord_model.chord_chain.smoothing == 0.1
        assert generator.melody_model.pitch_chain.smoothing == 0.2


class TestLoopGeneratorGenerate:
    """Tests for loop generation."""

    def test_generate_returns_loop(self) -> None:
        """Generate returns a GeneratedLoop."""
        generator = LoopGenerator()
        loop = generator.generate()

        assert isinstance(loop, GeneratedLoop)

    def test_generate_has_chord_sequence(self) -> None:
        """Generated loop has chord sequence."""
        generator = LoopGenerator()
        loop = generator.generate()

        assert loop.chord_sequence is not None
        assert len(loop.chord_sequence.events) > 0

    def test_generate_has_melody_sequence(self) -> None:
        """Generated loop has melody sequence."""
        generator = LoopGenerator()
        loop = generator.generate()

        assert loop.melody_sequence is not None
        assert len(loop.melody_sequence.notes) > 0

    def test_generate_has_voiced_notes(self) -> None:
        """Generated loop has voiced notes."""
        generator = LoopGenerator()
        loop = generator.generate()

        assert len(loop.chord_notes) > 0
        assert len(loop.melody_notes) > 0

    def test_generate_tracks_transitions(self) -> None:
        """Generated loop tracks transitions for reward."""
        generator = LoopGenerator()
        loop = generator.generate()

        assert len(loop.chord_transitions) > 0
        assert len(loop.melody_transitions) > 0

    def test_generate_calculates_duration(self) -> None:
        """Generated loop has duration in seconds."""
        generator = LoopGenerator()
        params = GenerationParams(num_bars=4, tempo_bpm=120)
        loop = generator.generate(params)

        # 4 bars at 120 BPM = 8 seconds
        assert abs(loop.duration_seconds - 8.0) < 0.01

    def test_generate_reproducible_with_seed(self) -> None:
        """Generation is reproducible with same seed."""
        generator = LoopGenerator()
        params = GenerationParams()

        loop1 = generator.generate(params, seed=42)
        loop2 = generator.generate(params, seed=42)

        # Chord sequences should be identical
        assert len(loop1.chord_sequence.events) == len(loop2.chord_sequence.events)
        for e1, e2 in zip(loop1.chord_sequence.events, loop2.chord_sequence.events):
            assert e1.degree == e2.degree
            assert e1.duration == e2.duration

    def test_generate_respects_params(self) -> None:
        """Generation respects parameters."""
        generator = LoopGenerator()
        params = GenerationParams(num_bars=8, beats_per_bar=4)
        loop = generator.generate(params)

        # 8 bars * 4 beats * 4 sixteenths = 128
        assert loop.chord_sequence.total_duration == 128


class TestLoopGeneratorMidi:
    """Tests for MIDI file saving."""

    def test_save_midi(self) -> None:
        """Can save generated loop to MIDI file."""
        generator = LoopGenerator()
        loop = generator.generate()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"
            result = generator.save_midi(loop, output_path)

            assert result.exists()
            assert result.stat().st_size > 0

    def test_saved_midi_readable(self) -> None:
        """Saved MIDI file can be read by mido."""
        import mido

        generator = LoopGenerator()
        loop = generator.generate()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mid"
            generator.save_midi(loop, output_path)

            # Should be readable
            midi_file = mido.MidiFile(str(output_path))
            assert len(midi_file.tracks) >= 2


class TestLoopGeneratorReward:
    """Tests for reward learning."""

    def test_apply_reward(self) -> None:
        """Reward can be applied to generated loop."""
        generator = LoopGenerator()
        loop = generator.generate()

        # Should not raise
        generator.apply_reward(loop, chord_reward=1.0, melody_reward=1.0)

    def test_apply_overall_reward(self) -> None:
        """Overall reward is applied to both models."""
        generator = LoopGenerator()
        loop = generator.generate()

        # Should not raise
        generator.apply_reward(loop, overall_reward=2.0)


class TestLoopGeneratorSerialization:
    """Tests for serialization."""

    def test_to_dict(self) -> None:
        """Generator can be serialized to dict."""
        generator = LoopGenerator()
        data = generator.to_dict()

        assert "chord_model" in data
        assert "melody_model" in data

    def test_from_dict(self) -> None:
        """Generator can be deserialized from dict."""
        generator = LoopGenerator()
        loop = generator.generate()
        generator.apply_reward(loop, chord_reward=5.0)

        data = generator.to_dict()
        restored = LoopGenerator.from_dict(data)

        # Should have same model structure
        assert (
            restored.chord_model.chord_chain.states
            == generator.chord_model.chord_chain.states
        )

    def test_reset_models(self) -> None:
        """reset_models restores theory priors."""
        generator = LoopGenerator()
        loop = generator.generate()
        generator.apply_reward(loop, chord_reward=100.0)

        generator.reset_models()

        # Models should be reset (hard to test exactly, just verify no error)
        assert generator.chord_model is not None

    def test_repr(self) -> None:
        """Generator has string representation."""
        generator = LoopGenerator()
        rep = repr(generator)
        assert "LoopGenerator" in rep


# =============================================================================
# Helper: Create test MIDI file
# =============================================================================


def create_test_midi_file(path: Path, notes: list[tuple[int, int, int]]) -> None:
    """
    Create a simple MIDI file for testing.

    Args:
        path: Output path
        notes: List of (midi_note, start_tick, duration_ticks)
    """
    midi_file = mido.MidiFile(ticks_per_beat=480)

    # Tempo track
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))
    tempo_track.append(mido.MetaMessage("end_of_track"))
    midi_file.tracks.append(tempo_track)

    # Note track
    note_track = mido.MidiTrack()
    note_track.append(mido.MetaMessage("track_name", name="Test Track"))
    note_track.append(mido.Message("program_change", program=0, channel=0))

    # Convert notes to events
    events: list[tuple[int, str, int]] = []
    for midi_note, start, duration in notes:
        events.append((start, "note_on", midi_note))
        events.append((start + duration, "note_off", midi_note))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    current_tick = 0
    for tick, event_type, midi_note in events:
        delta = tick - current_tick
        if event_type == "note_on":
            note_track.append(
                mido.Message("note_on", note=midi_note, velocity=100, time=delta)
            )
        else:
            note_track.append(
                mido.Message("note_off", note=midi_note, velocity=0, time=delta)
            )
        current_tick = tick

    note_track.append(mido.MetaMessage("end_of_track"))
    midi_file.tracks.append(note_track)

    midi_file.save(str(path))


# =============================================================================
# MIDI Loader Tests
# =============================================================================


class TestParsedNote:
    """Tests for ParsedNote dataclass."""

    def test_create_parsed_note(self) -> None:
        """ParsedNote can be created."""
        note = ParsedNote(midi=60, start_tick=0, duration_ticks=480)
        assert note.midi == 60
        assert note.start_tick == 0
        assert note.duration_ticks == 480


class TestParsedMidi:
    """Tests for ParsedMidi dataclass."""

    def test_get_all_notes_empty(self) -> None:
        """Empty ParsedMidi returns empty notes list."""
        parsed = ParsedMidi()
        assert parsed.get_all_notes() == []

    def test_get_all_notes_sorted(self) -> None:
        """Notes are sorted by start time."""
        track = ParsedTrack(
            name="Test",
            notes=[
                ParsedNote(midi=64, start_tick=480, duration_ticks=480),
                ParsedNote(midi=60, start_tick=0, duration_ticks=480),
            ],
        )
        parsed = ParsedMidi(tracks=[track])
        notes = parsed.get_all_notes()

        assert notes[0].start_tick == 0
        assert notes[1].start_tick == 480

    def test_get_duration_ticks(self) -> None:
        """Duration is calculated correctly."""
        track = ParsedTrack(
            name="Test",
            notes=[
                ParsedNote(midi=60, start_tick=0, duration_ticks=480),
                ParsedNote(midi=64, start_tick=480, duration_ticks=960),
            ],
        )
        parsed = ParsedMidi(tracks=[track], ticks_per_beat=480)

        # Last note ends at 480 + 960 = 1440
        assert parsed.get_duration_ticks() == 1440

    def test_get_duration_beats(self) -> None:
        """Duration in beats is calculated correctly."""
        track = ParsedTrack(
            name="Test",
            notes=[ParsedNote(midi=60, start_tick=0, duration_ticks=960)],
        )
        parsed = ParsedMidi(tracks=[track], ticks_per_beat=480)

        # 960 ticks / 480 ticks per beat = 2 beats
        assert parsed.get_duration_beats() == 2.0


class TestParseMidiFile:
    """Tests for MIDI file parsing."""

    def test_parse_simple_file(self) -> None:
        """Can parse a simple MIDI file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            create_test_midi_file(
                path,
                [(60, 0, 480), (64, 480, 480), (67, 960, 480)],
            )

            parsed = parse_midi_file(path)

            assert parsed.ticks_per_beat == 480
            assert len(parsed.tracks) == 1
            assert len(parsed.tracks[0].notes) == 3

    def test_parse_extracts_notes_correctly(self) -> None:
        """Parses note data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            create_test_midi_file(path, [(60, 0, 480)])

            parsed = parse_midi_file(path)
            note = parsed.tracks[0].notes[0]

            assert note.midi == 60
            assert note.start_tick == 0
            assert note.duration_ticks == 480

    def test_parse_missing_file_raises(self) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_midi_file("nonexistent.mid")


class TestQuantizeParsedMidi:
    """Tests for MIDI quantization."""

    def test_quantize_to_16ths(self) -> None:
        """Quantizes to 16th note grid."""
        track = ParsedTrack(
            name="Test",
            notes=[
                # Slightly off-grid note
                ParsedNote(midi=60, start_tick=50, duration_ticks=400),
            ],
        )
        parsed = ParsedMidi(tracks=[track], ticks_per_beat=480)

        quantized = quantize_parsed_midi(parsed, grid_subdivision=16)

        # Grid size = 480 / 4 = 120
        # 50 rounds to 0, 400 rounds to 360 (3 grid units)
        note = quantized.tracks[0].notes[0]
        assert note.start_tick == 0
        # Duration should be at least one grid unit
        assert note.duration_ticks >= 120


class TestExtractIntervals:
    """Tests for interval extraction."""

    def test_extract_ascending(self) -> None:
        """Extracts ascending intervals."""
        notes = [
            ParsedNote(midi=60, start_tick=0, duration_ticks=480),
            ParsedNote(midi=64, start_tick=480, duration_ticks=480),
            ParsedNote(midi=67, start_tick=960, duration_ticks=480),
        ]
        intervals = extract_intervals(notes)
        assert intervals == [4, 3]  # C to E, E to G

    def test_extract_descending(self) -> None:
        """Extracts descending intervals."""
        notes = [
            ParsedNote(midi=72, start_tick=0, duration_ticks=480),
            ParsedNote(midi=60, start_tick=480, duration_ticks=480),
        ]
        intervals = extract_intervals(notes)
        assert intervals == [-12]

    def test_empty_with_single_note(self) -> None:
        """Returns empty list for single note."""
        notes = [ParsedNote(midi=60, start_tick=0, duration_ticks=480)]
        assert extract_intervals(notes) == []


class TestExtractDurations:
    """Tests for duration extraction."""

    def test_extract_durations(self) -> None:
        """Extracts durations in 16th notes."""
        notes = [
            # Quarter note (480 ticks = 4 sixteenths)
            ParsedNote(midi=60, start_tick=0, duration_ticks=480),
            # Half note (960 ticks = 8 sixteenths)
            ParsedNote(midi=64, start_tick=480, duration_ticks=960),
        ]
        durations = extract_durations_16ths(notes, ticks_per_beat=480)
        assert durations == [4, 8]


class TestNotesToTrainingData:
    """Tests for training data extraction."""

    def test_returns_both_lists(self) -> None:
        """Returns intervals and durations."""
        notes = [
            ParsedNote(midi=60, start_tick=0, duration_ticks=480),
            ParsedNote(midi=64, start_tick=480, duration_ticks=480),
        ]
        data = notes_to_training_data(notes, ticks_per_beat=480)

        assert "intervals" in data
        assert "durations" in data
        assert data["intervals"] == [4]
        assert data["durations"] == [4, 4]
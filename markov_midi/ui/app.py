"""
Gradio Web UI for MarkovMIDI.

Provides a web interface for generating MIDI loops, rating them
for reward-based learning, viewing history, and managing sessions.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import plotly.graph_objects as go

from markov_midi.audio.synthesizer import Synthesizer, find_soundfonts
from markov_midi.generator.loop_generator import (
    GeneratedLoop,
    GenerationParams,
    LoopGenerator,
)
from markov_midi.generator.voicing import VoicingStyle
from markov_midi.model.chord_model import ChordModel
from markov_midi.model.melody_model import MelodyModel
from markov_midi.model.persistence import (
    Session,
    SessionMetadata,
    list_sessions,
    load_session,
    save_session,
)
from markov_midi.model.reward import (
    Rating,
    RewardManager,
)
from markov_midi.ui.visualizer import (
    create_empty_figure,
    create_piano_roll_figure,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Key choices with enharmonic equivalents
KEY_CHOICES: list[str] = [
    "C",
    "C#/Db",
    "D",
    "D#/Eb",
    "E",
    "F",
    "F#/Gb",
    "G",
    "G#/Ab",
    "A",
    "A#/Bb",
    "B",
]

# Custom CSS for styling
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&display=swap');

/* Apply Orbitron font globally */
.gradio-container,
.gradio-container *,
.gr-button,
.gr-input,
.gr-dropdown,
.gr-radio,
.gr-slider,
label,
span,
p,
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
}

/* Orange theme colors */
:root {
    --color-accent: #ff6b00 !important;
    --color-accent-soft: #ff6b0033 !important;
}

/* Primary buttons - orange */
.gr-button.primary {
    background: linear-gradient(135deg, #ff6b00 0%, #ff8533 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.gr-button.primary:hover {
    background: linear-gradient(135deg, #ff8533 0%, #ffa366 100%) !important;
    transform: translateY(-1px);
}

/* Secondary buttons */
.gr-button.secondary {
    border: 2px solid #ff6b00 !important;
    color: #ff6b00 !important;
    background: transparent !important;
}

.gr-button.secondary:hover {
    background: #ff6b0022 !important;
}

/* Remove blue label backgrounds */
.gr-form,
.gr-box,
.gr-panel,
label.svelte-1gfkn6j,
.label-wrap,
.gr-input-label,
.gr-block.gr-box {
    background: transparent !important;
    border: none !important;
}

/* Radio and checkbox styling */
.gr-radio label,
.gr-checkbox label {
    background: transparent !important;
}

/* Center the title */
.title-centered {
    text-align: center !important;
    color: #ff6b00 !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 0 20px #ff6b0044;
}

/* Center the header info */
.header-info {
    text-align: center !important;
    color: #888 !important;
    font-size: 0.9rem !important;
    margin-bottom: 1rem !important;
}

/* Star rating styles */
.star-container {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    margin: 0.5rem 0 !important;
}

.star-label {
    min-width: 100px !important;
    color: #ccc !important;
    font-size: 0.9rem !important;
}

/* Rating row styling */
.rating-row {
    align-items: center !important;
    margin-bottom: 0.5rem !important;
}

.rating-label {
    min-width: 100px !important;
    font-weight: 600 !important;
}

.star-display {
    font-size: 1.2rem !important;
    color: #ff6b00 !important;
}

.star-btn {
    font-size: 1.5rem !important;
    padding: 0.1rem 0.3rem !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
    transition: transform 0.1s ease, color 0.1s ease !important;
    min-width: 2rem !important;
    line-height: 1 !important;
    color: #ff6b00 !important;
}

.star-btn:hover {
    transform: scale(1.2) !important;
}

.star-empty {
    color: #444 !important;
}

.star-filled {
    color: #ff6b00 !important;
    text-shadow: 0 0 10px #ff6b0066 !important;
}

/* Tab styling */
.tabs > .tab-nav > button {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 500 !important;
}

.tabs > .tab-nav > button.selected {
    color: #ff6b00 !important;
    border-bottom-color: #ff6b00 !important;
}

/* Slider accent */
input[type="range"]::-webkit-slider-thumb {
    background: #ff6b00 !important;
}

input[type="range"]::-moz-range-thumb {
    background: #ff6b00 !important;
}

/* Dropdown styling */
.gr-dropdown {
    border-color: #444 !important;
}

.gr-dropdown:focus {
    border-color: #ff6b00 !important;
}

/* Section dividers */
.section-divider {
    border-top: 1px solid #333 !important;
    margin: 1rem 0 !important;
}

/* Disabled button */
.gr-button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}
"""


# -----------------------------------------------------------------------------
# App State
# -----------------------------------------------------------------------------


@dataclass
class AppState:
    """Holds the current application state."""

    chord_model: ChordModel = field(default_factory=ChordModel)
    melody_model: MelodyModel = field(default_factory=MelodyModel)
    reward_manager: RewardManager = field(default_factory=RewardManager)
    session_name: str = "Untitled Session"
    session_created: str = ""
    current_loop: GeneratedLoop | None = None
    current_generation_id: str | None = None
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))
    has_unsaved_changes: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "chord_model": self.chord_model.to_dict(),
            "melody_model": self.melody_model.to_dict(),
            "reward_manager": self.reward_manager.to_dict(),
            "session_name": self.session_name,
            "session_created": self.session_created,
            "has_unsaved_changes": self.has_unsaved_changes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppState":
        """Deserialize state from dictionary."""
        state = cls()
        state.chord_model = ChordModel.from_dict(data["chord_model"])
        state.melody_model = MelodyModel.from_dict(data["melody_model"])
        state.reward_manager = RewardManager.from_dict(data["reward_manager"])
        state.session_name = data.get("session_name", "Untitled Session")
        state.session_created = data.get("session_created", "")
        state.has_unsaved_changes = data.get("has_unsaved_changes", False)
        return state


def create_fresh_state() -> AppState:
    """Create a fresh application state."""
    state = AppState()
    state.session_created = datetime.now().isoformat()
    return state


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def parse_key(key_choice: str) -> str:
    """Extract the first key from an enharmonic pair (e.g., 'C#/Db' -> 'C#')."""
    return key_choice.split("/")[0]


def density_to_note_density(density: str) -> tuple[float, float]:
    """Convert density preset to note_density range (0.0-1.0)."""
    if density == "Low":
        return (0.3, 0.5)
    elif density == "High":
        return (0.8, 1.0)
    else:  # Medium (default)
        return (0.5, 0.75)


# Alias for backwards compatibility
density_to_notes_per_bar = density_to_note_density


def generate_loop(
    state: AppState,
    key: str,
    mode: str,
    bars: int,
    tempo: int,
    voicing: str,
    chord_complexity: str,
    density: str,
) -> tuple[AppState, GeneratedLoop]:
    """Generate a new loop with the given parameters."""
    import random

    # Parse key from enharmonic choice
    parsed_key = parse_key(key)

    # Map UI values to config
    voicing_style = (
        VoicingStyle.ARPEGGIO_UP if voicing == "Arpeggiated" else VoicingStyle.BLOCK
    )
    use_sevenths = chord_complexity == "Include 7ths"
    min_density, max_density = density_to_note_density(density)
    note_density = random.uniform(min_density, max_density)

    params = GenerationParams(
        key=parsed_key,
        mode=mode.lower(),
        num_bars=bars,
        tempo_bpm=tempo,
        voicing_style=voicing_style,
        use_seventh_chords=use_sevenths,
        note_density=note_density,
    )

    generator = LoopGenerator(
        chord_model=state.chord_model,
        melody_model=state.melody_model,
    )

    result = generator.generate(params)

    # Update state
    state.current_loop = result
    state.current_generation_id = str(uuid.uuid4())

    return state, result


def write_midi_to_temp(state: AppState, result: GeneratedLoop) -> Path:
    """Write the MIDI result to a temporary file."""
    filename = f"markov_midi_{state.current_generation_id}.mid"
    filepath = state.temp_dir / filename

    generator = LoopGenerator(
        chord_model=state.chord_model,
        melody_model=state.melody_model,
    )
    generator.save_midi(result, filepath)

    return filepath


def render_audio(synthesizer: Synthesizer, midi_path: Path) -> Path | None:
    """Render MIDI to audio if synthesizer is available."""
    if not synthesizer.is_available():
        return None

    wav_path = midi_path.with_suffix(".wav")
    success = synthesizer.render(midi_path, wav_path)
    return wav_path if success else None


# -----------------------------------------------------------------------------
# Rating Logic
# -----------------------------------------------------------------------------


def apply_rating(
    state: AppState,
    melodic: int,
    melodic_rhythm: int,
    harmonic: int,
    harmonic_rhythm: int,
    cohesion: int,
    overall: int,
) -> AppState:
    """Apply a rating to the current generation."""
    if state.current_loop is None or state.current_generation_id is None:
        return state

    # Create rating with all 6 categories
    # Convert 0 (unrated) to 3 (neutral) for the backend
    rating = Rating(
        melodic=melodic if melodic > 0 else 3,
        melodic_rhythm=melodic_rhythm if melodic_rhythm > 0 else 3,
        harmonic=harmonic if harmonic > 0 else 3,
        harmonic_rhythm=harmonic_rhythm if harmonic_rhythm > 0 else 3,
        cohesion=cohesion if cohesion > 0 else 3,
        overall=overall if overall > 0 else 3,
    )

    result = state.current_loop

    # Record generation using the proper API
    generator = LoopGenerator(
        chord_model=state.chord_model,
        melody_model=state.melody_model,
    )

    record = state.reward_manager.record_generation(
        loop=result,
        params=result.params,
    )

    # Apply rating using the recorded generation ID
    state.reward_manager.apply_rating(
        generation_id=record.generation_id,
        rating=rating,
        generator=generator,
    )

    state.has_unsaved_changes = True
    return state


# -----------------------------------------------------------------------------
# Session Logic
# -----------------------------------------------------------------------------


def save_current_session(state: AppState, session_dir: Path | None = None) -> AppState:
    """Save the current session to disk."""
    metadata = SessionMetadata(name=state.session_name)
    if state.session_created:
        metadata.created_at = state.session_created

    # Create a generator to serialize its state
    generator = LoopGenerator(
        chord_model=state.chord_model,
        melody_model=state.melody_model,
    )

    session = Session(
        metadata=metadata,
        generator_state=generator.to_dict(),
        reward_manager_state=state.reward_manager.to_dict(),
    )

    # Build file path
    if session_dir is None:
        session_dir = Path.home() / ".markov_midi" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize session name for filename
    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_" for c in state.session_name
    )
    file_path = session_dir / f"{safe_name}.json"

    save_session(session, file_path)
    state.has_unsaved_changes = False
    return state


def load_existing_session(
    state: AppState,
    session_name: str,
    session_dir: Path | None = None,
) -> AppState:
    """Load an existing session from disk."""
    # Build file path
    if session_dir is None:
        session_dir = Path.home() / ".markov_midi" / "sessions"

    # Sanitize session name for filename
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in session_name)
    file_path = session_dir / f"{safe_name}.json"

    session = load_session(file_path)

    # Restore generator state
    generator = LoopGenerator.from_dict(session.generator_state)
    state.chord_model = generator.chord_model
    state.melody_model = generator.melody_model

    # Restore reward manager
    state.reward_manager = RewardManager.from_dict(session.reward_manager_state)

    state.session_name = session.metadata.name
    state.session_created = session.metadata.created_at
    state.has_unsaved_changes = False
    state.current_loop = None
    state.current_generation_id = None

    return state


def create_new_session(state: AppState, name: str) -> AppState:
    """Create a new fresh session."""
    state = create_fresh_state()
    state.session_name = name if name else "Untitled Session"
    state.session_created = datetime.now().isoformat()
    return state


# -----------------------------------------------------------------------------
# UI Formatting Helpers
# -----------------------------------------------------------------------------


def format_stats(state: AppState) -> str:
    """Format training statistics."""
    stats = state.reward_manager.get_statistics()
    total = stats["total_generations"]
    avg = stats["average_overall"]
    avg_str = f"{avg:.1f}" if avg is not None else "N/A"
    return f"Trained: {total} loops | Avg Rating: {avg_str}"


def format_header(state: AppState) -> str:
    """Format the header text."""
    unsaved = " *" if state.has_unsaved_changes else ""
    stats = format_stats(state)
    return f"Session: {state.session_name}{unsaved} | {stats}"


def format_chord_progression(result: GeneratedLoop | None) -> str:
    """Format chord progression display."""
    if result is None:
        return "No loop generated yet"
    # Display chord degrees with Roman numerals
    degree_names = ["I", "ii", "iii", "IV", "V", "vi", "vii"]
    chord_names = []
    for event in result.chord_sequence.events:
        if 1 <= event.degree <= 7:
            chord_names.append(degree_names[event.degree - 1])
        else:
            chord_names.append(str(event.degree))
    return " -> ".join(chord_names)


def format_loop_info(result: GeneratedLoop | None) -> str:
    """Format loop info display."""
    if result is None:
        return ""
    mode_display = result.params.mode.capitalize()
    return (
        f"Key: {result.params.key} {mode_display} | "
        f"{result.params.num_bars} bars @ {result.params.tempo_bpm} BPM"
    )


def stars_display(rating: int) -> str:
    """Convert rating to star display."""
    if rating == 0:
        return "☆☆☆☆☆"
    filled = rating
    empty = 5 - rating
    return "★" * filled + "☆" * empty


def render_stars(value: int) -> str:
    """Render star buttons HTML for a given value."""
    stars = []
    for i in range(1, 6):
        if i <= value:
            stars.append("★")
        else:
            stars.append("☆")
    return " ".join(stars)


# -----------------------------------------------------------------------------
# Gradio UI Builder
# -----------------------------------------------------------------------------


def create_ui(
    session_dir: Path | None = None,
    share: bool = False,
) -> gr.Blocks:
    """Create and return the Gradio UI."""
    # Check for soundfonts
    soundfonts = find_soundfonts()
    synthesizer = Synthesizer(soundfonts[0]) if soundfonts else Synthesizer()

    # Default session directory
    if session_dir is None:
        session_dir = Path.home() / ".markov_midi" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

    # Store session_dir for use in closures
    _session_dir = session_dir

    def get_session_choices() -> list[str]:
        """Get list of available sessions."""
        try:
            sessions = list_sessions(_session_dir)
            return [s["name"] for s in sessions]
        except Exception:
            return []

    # Build UI
    with gr.Blocks(
        title="MarkovMIDI",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=CUSTOM_CSS,
    ) as app:
        # State
        state = gr.State(create_fresh_state())

        # Rating states (0 = unrated)
        rating_melodic = gr.State(0)
        rating_melodic_rhythm = gr.State(0)
        rating_harmonic = gr.State(0)
        rating_harmonic_rhythm = gr.State(0)
        rating_cohesion = gr.State(0)
        rating_overall = gr.State(0)

        # Header - Centered title
        gr.HTML("<h1 class='title-centered'>MarkovMIDI</h1>")
        header_text = gr.Markdown(
            "Session: Untitled Session | Trained: 0 loops | Avg Rating: N/A",
            elem_classes=["header-info"],
        )

        # Tabs
        with gr.Tabs():
            # -----------------------------------------------------------------
            # Tab 1: Generate
            # -----------------------------------------------------------------
            with gr.TabItem("Generate"):
                # Row 1: Key, Mode, Loop Length
                with gr.Row():
                    key_dropdown = gr.Dropdown(
                        choices=KEY_CHOICES,
                        value="C",
                        label="Key",
                        scale=1,
                    )
                    mode_radio = gr.Radio(
                        choices=["Major", "Minor"],
                        value="Major",
                        label="Mode",
                        scale=1,
                    )
                    length_radio = gr.Radio(
                        choices=["4 bars", "8 bars"],
                        value="4 bars",
                        label="Loop Length",
                        scale=1,
                    )

                # Row 2: Tempo, Melody Density
                with gr.Row():
                    tempo_slider = gr.Slider(
                        minimum=60,
                        maximum=180,
                        value=120,
                        step=1,
                        label="Tempo (BPM)",
                        scale=2,
                    )
                    density_dropdown = gr.Dropdown(
                        choices=["Low", "Medium", "High"],
                        value="Medium",
                        label="Melody Density",
                        scale=1,
                    )

                # Row 3: Voicing Style, Chord Complexity
                with gr.Row():
                    voicing_radio = gr.Radio(
                        choices=["Block", "Arpeggiated"],
                        value="Block",
                        label="Voicing Style",
                        scale=1,
                    )
                    chord_complexity_radio = gr.Radio(
                        choices=["Triads Only", "Include 7ths"],
                        value="Triads Only",
                        label="Chord Complexity",
                        scale=1,
                    )

                # Generate button
                generate_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg",
                )

                # Output section
                gr.HTML("<hr class='section-divider'>")
                gr.Markdown("## Loop")

                # Audio/MIDI output
                audio_output = gr.Audio(
                    label="Audio Preview",
                    visible=synthesizer.is_available(),
                    show_label=False,
                )
                if not synthesizer.is_available():
                    gr.Markdown(
                        "*Audio preview unavailable - no soundfont configured.*",
                    )
                midi_download = gr.File(
                    label="Download MIDI",
                    visible=False,
                )

                # Piano roll visualization
                piano_roll_plot = gr.Plot(
                    value=create_empty_figure(),
                    label="Piano Roll",
                    show_label=False,
                )

                # Rating section
                gr.HTML("<hr class='section-divider'>")
                gr.Markdown("## Rate This Loop")

                # Star rating rows - label + 5 star buttons in single row
                # Each button shows ☆ (empty) or ★ (filled) based on rating
                # Order: Melodic, Melodic Rhythm, Harmonic, Harmonic Rhythm, Cohesion, Overall

                # Melodic
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Melodic**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            melodic_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Melodic Rhythm
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Melodic Rhythm**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            melodic_rhythm_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_rhythm_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_rhythm_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_rhythm_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            melodic_rhythm_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Harmonic
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Harmonic**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            harmonic_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Harmonic Rhythm
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Harmonic Rhythm**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            harmonic_rhythm_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_rhythm_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_rhythm_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_rhythm_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            harmonic_rhythm_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Cohesion
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Cohesion**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            cohesion_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            cohesion_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            cohesion_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            cohesion_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            cohesion_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Overall
                with gr.Row(elem_classes=["rating-row"]):
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown("**Overall**")
                    with gr.Column(scale=3):
                        with gr.Row():
                            overall_btn_1 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            overall_btn_2 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            overall_btn_3 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            overall_btn_4 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )
                            overall_btn_5 = gr.Button(
                                "☆", size="sm", elem_classes=["star-btn"]
                            )

                # Submit button
                submit_btn = gr.Button(
                    "Submit",
                    variant="primary",
                    size="lg",
                    interactive=False,  # Disabled until rating given
                )

                # ----- Star click handlers -----
                # Each handler returns: (new_rating, btn1, btn2, btn3, btn4, btn5, submit_update)
                # Where btn1-5 are the new button labels and submit_update enables the button

                def make_star_click_handler(
                    rating_num: int,
                ) -> Callable[
                    [int, int, int, int, int, int, int],
                    tuple[int, str, str, str, str, str, dict[str, Any]],
                ]:
                    """Create a click handler for star button N."""

                    def handler(
                        current: int,
                        m: int,
                        mr: int,
                        h: int,
                        hr: int,
                        c: int,
                        o: int,
                    ) -> tuple[int, str, str, str, str, str, dict[str, Any]]:
                        # Generate button labels for this rating
                        labels = ["★" if i < rating_num else "☆" for i in range(5)]
                        # Check if submit should be enabled (at least one rating > 0)
                        # Note: 'current' is the old value, rating_num is the new value
                        # We need to check if ANY of the 6 ratings will be > 0 after this click
                        # Since we're setting one of them to rating_num, submit should be enabled
                        return (
                            rating_num,
                            labels[0],
                            labels[1],
                            labels[2],
                            labels[3],
                            labels[4],
                            gr.update(interactive=True),
                        )

                    return handler

                # Wire up Melodic star buttons
                melodic_btns = [
                    melodic_btn_1,
                    melodic_btn_2,
                    melodic_btn_3,
                    melodic_btn_4,
                    melodic_btn_5,
                ]
                for i, btn in enumerate(melodic_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_melodic,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_melodic,
                            melodic_btn_1,
                            melodic_btn_2,
                            melodic_btn_3,
                            melodic_btn_4,
                            melodic_btn_5,
                            submit_btn,
                        ],
                    )

                # Wire up Melodic Rhythm star buttons
                melodic_rhythm_btns = [
                    melodic_rhythm_btn_1,
                    melodic_rhythm_btn_2,
                    melodic_rhythm_btn_3,
                    melodic_rhythm_btn_4,
                    melodic_rhythm_btn_5,
                ]
                for i, btn in enumerate(melodic_rhythm_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_melodic_rhythm,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_melodic_rhythm,
                            melodic_rhythm_btn_1,
                            melodic_rhythm_btn_2,
                            melodic_rhythm_btn_3,
                            melodic_rhythm_btn_4,
                            melodic_rhythm_btn_5,
                            submit_btn,
                        ],
                    )

                # Wire up Harmonic star buttons
                harmonic_btns = [
                    harmonic_btn_1,
                    harmonic_btn_2,
                    harmonic_btn_3,
                    harmonic_btn_4,
                    harmonic_btn_5,
                ]
                for i, btn in enumerate(harmonic_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_harmonic,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_harmonic,
                            harmonic_btn_1,
                            harmonic_btn_2,
                            harmonic_btn_3,
                            harmonic_btn_4,
                            harmonic_btn_5,
                            submit_btn,
                        ],
                    )

                # Wire up Harmonic Rhythm star buttons
                harmonic_rhythm_btns = [
                    harmonic_rhythm_btn_1,
                    harmonic_rhythm_btn_2,
                    harmonic_rhythm_btn_3,
                    harmonic_rhythm_btn_4,
                    harmonic_rhythm_btn_5,
                ]
                for i, btn in enumerate(harmonic_rhythm_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_harmonic_rhythm,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_harmonic_rhythm,
                            harmonic_rhythm_btn_1,
                            harmonic_rhythm_btn_2,
                            harmonic_rhythm_btn_3,
                            harmonic_rhythm_btn_4,
                            harmonic_rhythm_btn_5,
                            submit_btn,
                        ],
                    )

                # Wire up Cohesion star buttons
                cohesion_btns = [
                    cohesion_btn_1,
                    cohesion_btn_2,
                    cohesion_btn_3,
                    cohesion_btn_4,
                    cohesion_btn_5,
                ]
                for i, btn in enumerate(cohesion_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_cohesion,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_cohesion,
                            cohesion_btn_1,
                            cohesion_btn_2,
                            cohesion_btn_3,
                            cohesion_btn_4,
                            cohesion_btn_5,
                            submit_btn,
                        ],
                    )

                # Wire up Overall star buttons
                overall_btns = [
                    overall_btn_1,
                    overall_btn_2,
                    overall_btn_3,
                    overall_btn_4,
                    overall_btn_5,
                ]
                for i, btn in enumerate(overall_btns, 1):
                    btn.click(
                        make_star_click_handler(i),
                        inputs=[
                            rating_overall,
                            rating_melodic,
                            rating_melodic_rhythm,
                            rating_harmonic,
                            rating_harmonic_rhythm,
                            rating_cohesion,
                            rating_overall,
                        ],
                        outputs=[
                            rating_overall,
                            overall_btn_1,
                            overall_btn_2,
                            overall_btn_3,
                            overall_btn_4,
                            overall_btn_5,
                            submit_btn,
                        ],
                    )

                # ----- Generate handler -----
                def on_generate(
                    state_val: AppState,
                    key: str,
                    mode: str,
                    length: str,
                    tempo: int,
                    voicing: str,
                    complexity: str,
                    density: str,
                ) -> tuple[AppState, str, str | None, dict[str, Any], go.Figure]:
                    """Handle generate button click."""
                    # Parse bars from length string
                    bars = 4 if length == "4 bars" else 8

                    # Generate
                    state_val, result = generate_loop(
                        state_val,
                        key=key,
                        mode=mode,
                        bars=bars,
                        tempo=tempo,
                        voicing=voicing,
                        chord_complexity=complexity,
                        density=density,
                    )

                    # Write MIDI
                    midi_path = write_midi_to_temp(state_val, result)

                    # Try to render audio
                    audio_path: str | None = None
                    if synthesizer.is_available():
                        wav_path = render_audio(synthesizer, midi_path)
                        if wav_path and wav_path.exists():
                            audio_path = str(wav_path)

                    # Format header
                    header = format_header(state_val)

                    # Create piano roll figure
                    figure = create_piano_roll_figure(result)

                    return (
                        state_val,
                        header,
                        audio_path,
                        gr.update(value=str(midi_path), visible=True),
                        figure,
                    )

                generate_btn.click(
                    fn=on_generate,
                    inputs=[
                        state,
                        key_dropdown,
                        mode_radio,
                        length_radio,
                        tempo_slider,
                        voicing_radio,
                        chord_complexity_radio,
                        density_dropdown,
                    ],
                    outputs=[
                        state,
                        header_text,
                        audio_output,
                        midi_download,
                        piano_roll_plot,
                    ],
                )

                # ----- Submit handler -----
                def on_submit(
                    state_val: AppState,
                    r_melodic: int,
                    r_melodic_rhythm: int,
                    r_harmonic: int,
                    r_harmonic_rhythm: int,
                    r_cohesion: int,
                    r_overall: int,
                    key: str,
                    mode: str,
                    length: str,
                    tempo: int,
                    voicing: str,
                    complexity: str,
                    density: str,
                ) -> tuple[
                    AppState,
                    str,
                    str | None,
                    dict[str, Any],
                    go.Figure,  # piano roll
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,  # 6 rating states
                    str,
                    str,
                    str,
                    str,
                    str,  # melodic_btn_1-5
                    str,
                    str,
                    str,
                    str,
                    str,  # melodic_rhythm_btn_1-5
                    str,
                    str,
                    str,
                    str,
                    str,  # harmonic_btn_1-5
                    str,
                    str,
                    str,
                    str,
                    str,  # harmonic_rhythm_btn_1-5
                    str,
                    str,
                    str,
                    str,
                    str,  # cohesion_btn_1-5
                    str,
                    str,
                    str,
                    str,
                    str,  # overall_btn_1-5
                    dict[str, Any],  # submit_btn
                ]:
                    """Handle submit rating and auto-generate next."""
                    # Apply rating
                    state_val = apply_rating(
                        state_val,
                        r_melodic,
                        r_melodic_rhythm,
                        r_harmonic,
                        r_harmonic_rhythm,
                        r_cohesion,
                        r_overall,
                    )

                    # Auto-generate next
                    bars = 4 if length == "4 bars" else 8
                    state_val, result = generate_loop(
                        state_val,
                        key=key,
                        mode=mode,
                        bars=bars,
                        tempo=tempo,
                        voicing=voicing,
                        chord_complexity=complexity,
                        density=density,
                    )

                    # Write MIDI
                    midi_path = write_midi_to_temp(state_val, result)

                    # Try to render audio
                    audio_path: str | None = None
                    if synthesizer.is_available():
                        wav_path = render_audio(synthesizer, midi_path)
                        if wav_path and wav_path.exists():
                            audio_path = str(wav_path)

                    # Format header
                    header = format_header(state_val)

                    # Create piano roll figure
                    figure = create_piano_roll_figure(result)

                    # Reset all ratings to 0 and reset star buttons to empty
                    empty = "☆"

                    return (
                        state_val,
                        header,
                        audio_path,
                        gr.update(value=str(midi_path), visible=True),
                        figure,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,  # 6 rating states reset
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # melodic btns
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # melodic_rhythm btns
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # harmonic btns
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # harmonic_rhythm btns
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # cohesion btns
                        empty,
                        empty,
                        empty,
                        empty,
                        empty,  # overall btns
                        gr.update(interactive=False),  # submit_btn
                    )

                submit_btn.click(
                    fn=on_submit,
                    inputs=[
                        state,
                        rating_melodic,
                        rating_melodic_rhythm,
                        rating_harmonic,
                        rating_harmonic_rhythm,
                        rating_cohesion,
                        rating_overall,
                        key_dropdown,
                        mode_radio,
                        length_radio,
                        tempo_slider,
                        voicing_radio,
                        chord_complexity_radio,
                        density_dropdown,
                    ],
                    outputs=[
                        state,
                        header_text,
                        audio_output,
                        midi_download,
                        piano_roll_plot,
                        rating_melodic,
                        rating_melodic_rhythm,
                        rating_harmonic,
                        rating_harmonic_rhythm,
                        rating_cohesion,
                        rating_overall,
                        melodic_btn_1,
                        melodic_btn_2,
                        melodic_btn_3,
                        melodic_btn_4,
                        melodic_btn_5,
                        melodic_rhythm_btn_1,
                        melodic_rhythm_btn_2,
                        melodic_rhythm_btn_3,
                        melodic_rhythm_btn_4,
                        melodic_rhythm_btn_5,
                        harmonic_btn_1,
                        harmonic_btn_2,
                        harmonic_btn_3,
                        harmonic_btn_4,
                        harmonic_btn_5,
                        harmonic_rhythm_btn_1,
                        harmonic_rhythm_btn_2,
                        harmonic_rhythm_btn_3,
                        harmonic_rhythm_btn_4,
                        harmonic_rhythm_btn_5,
                        cohesion_btn_1,
                        cohesion_btn_2,
                        cohesion_btn_3,
                        cohesion_btn_4,
                        cohesion_btn_5,
                        overall_btn_1,
                        overall_btn_2,
                        overall_btn_3,
                        overall_btn_4,
                        overall_btn_5,
                        submit_btn,
                    ],
                )

            # -----------------------------------------------------------------
            # Tab 2: History
            # -----------------------------------------------------------------
            with gr.TabItem("History"):
                gr.Markdown("## Generation History")

                history_filter = gr.Dropdown(
                    choices=["All Ratings", "5 Stars", "4+ Stars", "3+ Stars"],
                    value="All Ratings",
                    label="Filter",
                )

                history_display = gr.Dataframe(
                    headers=["#", "Key", "Mode", "Chords", "Rating"],
                    datatype=["number", "str", "str", "str", "str"],
                    label="History",
                    interactive=False,
                )

                refresh_history_btn = gr.Button("Refresh History")

                def get_history_data(
                    state_val: AppState,
                    filter_val: str,
                ) -> list[list[str | int]]:
                    """Get history data for display."""
                    records = state_val.reward_manager.history
                    rows: list[list[str | int]] = []

                    for i, record in enumerate(reversed(records), 1):
                        # Check filter
                        if record.rating is not None:
                            rating_val = record.rating.overall
                            if filter_val == "5 Stars" and rating_val != 5:
                                continue
                            if filter_val == "4+ Stars" and rating_val < 4:
                                continue
                            if filter_val == "3+ Stars" and rating_val < 3:
                                continue
                            rating_str = stars_display(rating_val)
                        else:
                            rating_str = "Not rated"

                        # Get key and mode from params dict
                        key = record.params.get("key", "?")
                        mode = record.params.get("mode", "?")

                        # Format chord transitions (show degrees)
                        chord_degrees = [
                            str(t[1]) for t in record.chord_transitions[:4]
                        ]
                        chords_str = " → ".join(chord_degrees) if chord_degrees else "-"
                        if len(record.chord_transitions) > 4:
                            chords_str += " ..."

                        rows.append(
                            [
                                i,
                                str(key),
                                str(mode).title(),
                                chords_str,
                                rating_str,
                            ]
                        )

                    return rows

                refresh_history_btn.click(
                    fn=get_history_data,
                    inputs=[state, history_filter],
                    outputs=[history_display],
                )

                history_filter.change(
                    fn=get_history_data,
                    inputs=[state, history_filter],
                    outputs=[history_display],
                )

            # -----------------------------------------------------------------
            # Tab 3: Session Management
            # -----------------------------------------------------------------
            with gr.TabItem("Session"):
                gr.Markdown("## Current Session")

                session_info_text = gr.Markdown(
                    "**Name:** Untitled Session\n\n**Created:** -\n\n**Generations:** 0"
                )

                with gr.Row():
                    save_session_btn = gr.Button(
                        "Save Session",
                        variant="primary",
                    )
                    export_model_btn = gr.Button("Export Model Only")

                save_status_text = gr.Markdown("")

                gr.HTML("<hr class='section-divider'>")
                gr.Markdown("## Load Existing Session")

                session_dropdown = gr.Dropdown(
                    choices=get_session_choices(),
                    label="Select Session",
                )
                refresh_sessions_btn = gr.Button("Refresh List")
                load_session_btn = gr.Button("Load Session")
                load_status_text = gr.Markdown("")

                gr.HTML("<hr class='section-divider'>")
                gr.Markdown("## Start Fresh")

                new_session_name = gr.Textbox(
                    label="New Session Name",
                    placeholder="Enter session name...",
                )
                create_session_btn = gr.Button("Create New Session")

                # ----- Event Handlers for Session -----

                def on_save_session(
                    state_val: AppState,
                ) -> tuple[AppState, str, str]:
                    """Save current session."""
                    try:
                        state_val = save_current_session(state_val, _session_dir)
                        status = f"Session '{state_val.session_name}' saved!"
                        header = format_header(state_val)
                        return state_val, status, header
                    except Exception as e:
                        return state_val, f"Error saving session: {e}", ""

                save_session_btn.click(
                    fn=on_save_session,
                    inputs=[state],
                    outputs=[state, save_status_text, header_text],
                )

                def on_refresh_sessions() -> dict[str, Any]:
                    """Refresh session list."""
                    result: dict[str, Any] = gr.update(choices=get_session_choices())
                    return result

                refresh_sessions_btn.click(
                    fn=on_refresh_sessions,
                    inputs=[],
                    outputs=[session_dropdown],
                )

                def on_load_session(
                    state_val: AppState,
                    selected_session: str | None,
                ) -> tuple[AppState, str, str, str]:
                    """Load selected session."""
                    if not selected_session:
                        return (
                            state_val,
                            "Please select a session to load.",
                            "",
                            "",
                        )

                    try:
                        state_val = load_existing_session(
                            state_val,
                            selected_session,
                            _session_dir,
                        )
                        status = f"Session '{selected_session}' loaded!"
                        header = format_header(state_val)

                        # Session info
                        total_gens = state_val.reward_manager.get_statistics()
                        info = (
                            f"**Name:** {state_val.session_name}\n\n"
                            f"**Created:** {state_val.session_created[:19]}\n\n"
                            f"**Generations:** {total_gens['total_generations']}"
                        )

                        return state_val, status, header, info
                    except Exception as e:
                        return (
                            state_val,
                            f"Error loading session: {e}",
                            "",
                            "",
                        )

                load_session_btn.click(
                    fn=on_load_session,
                    inputs=[state, session_dropdown],
                    outputs=[
                        state,
                        load_status_text,
                        header_text,
                        session_info_text,
                    ],
                )

                def on_create_session(
                    state_val: AppState,
                    name: str,
                ) -> tuple[AppState, str, str]:
                    """Create new session."""
                    state_val = create_new_session(state_val, name)
                    header = format_header(state_val)

                    # Session info
                    info = (
                        f"**Name:** {state_val.session_name}\n\n"
                        f"**Created:** {state_val.session_created[:19]}\n\n"
                        f"**Generations:** 0"
                    )

                    return state_val, header, info

                create_session_btn.click(
                    fn=on_create_session,
                    inputs=[state, new_session_name],
                    outputs=[state, header_text, session_info_text],
                )

    return app  # type: ignore[no-any-return]


def launch(
    session_dir: Path | None = None,
    share: bool = False,
    server_port: int = 7860,
) -> None:
    """Launch the Gradio UI."""
    app = create_ui(session_dir=session_dir, share=share)
    app.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()

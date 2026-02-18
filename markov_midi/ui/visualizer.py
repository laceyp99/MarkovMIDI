"""
Piano roll visualization for MarkovMIDI.

Provides interactive Plotly-based visualization of generated MIDI loops,
with separate panels for melody and harmony to facilitate independent rating.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from markov_midi.generator.loop_generator import GeneratedLoop
from markov_midi.generator.voicing import VoicedNote
from markov_midi.utils.music_theory import midi_to_note


# Color scheme (matching the orange theme)
MELODY_COLOR = "#ff6b00"  # Orange for melody
HARMONY_COLOR = "#6b5bff"  # Purple/blue for harmony
BACKGROUND_COLOR = "#1a1a1a"  # Dark background
GRID_COLOR = "#333333"  # Subtle grid
TEXT_COLOR = "#cccccc"  # Light text


def midi_to_display_name(midi_num: int) -> str:
    """
    Convert MIDI number to display name (e.g., 60 -> 'C4').

    Args:
        midi_num: MIDI note number (0-127)

    Returns:
        Note name with octave (e.g., 'C4', 'F#5')
    """
    note_name, octave = midi_to_note(midi_num)
    return f"{note_name}{octave}"


def sixteenth_to_beats(sixteenths: int | float) -> float:
    """
    Convert 16th notes to beats (quarter notes).

    Args:
        sixteenths: Duration or position in 16th notes

    Returns:
        Duration or position in beats
    """
    return sixteenths / 4.0


def create_note_shapes(
    notes: list[VoicedNote],
    color: str,
    row: int,
) -> list[dict[str, Any]]:
    """
    Create Plotly shape definitions for a list of notes.

    Each note becomes a rounded rectangle positioned by start_time (x)
    and MIDI pitch (y).

    Args:
        notes: List of VoicedNote objects
        color: Fill color for the notes
        row: Subplot row (1 or 2)

    Returns:
        List of Plotly shape dictionaries
    """
    shapes: list[dict[str, Any]] = []

    for note in notes:
        x0 = sixteenth_to_beats(note.start_time)
        x1 = sixteenth_to_beats(note.start_time + note.duration)
        y0 = note.midi - 0.4  # Slight padding
        y1 = note.midi + 0.4

        # Calculate opacity based on velocity (0-127 -> 0.4-1.0)
        opacity = 0.4 + (note.velocity / 127) * 0.6

        shapes.append(
            {
                "type": "rect",
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "fillcolor": color,
                "opacity": opacity,
                "line": {"width": 1, "color": color},
                "xref": f"x{row}" if row > 1 else "x",
                "yref": f"y{row}" if row > 1 else "y",
            }
        )

    return shapes


def get_pitch_range(notes: list[VoicedNote]) -> tuple[int, int]:
    """
    Get the min and max MIDI pitch from a list of notes.

    Args:
        notes: List of VoicedNote objects

    Returns:
        Tuple of (min_pitch, max_pitch)
    """
    if not notes:
        return (60, 72)  # Default C4-C5 range

    pitches = [n.midi for n in notes]
    return (min(pitches), max(pitches))


def create_pitch_labels(min_pitch: int, max_pitch: int) -> tuple[list[int], list[str]]:
    """
    Create pitch tick values and labels for Y-axis.

    Shows note names for C notes and important scale degrees.

    Args:
        min_pitch: Minimum MIDI pitch
        max_pitch: Maximum MIDI pitch

    Returns:
        Tuple of (tick_values, tick_labels)
    """
    tick_vals: list[int] = []
    tick_labels: list[str] = []

    # Add padding
    min_pitch = max(0, min_pitch - 2)
    max_pitch = min(127, max_pitch + 2)

    for midi in range(min_pitch, max_pitch + 1):
        note_name, octave = midi_to_note(midi)
        # Show labels for C notes and every 4th semitone
        if note_name == "C" or midi % 4 == 0:
            tick_vals.append(midi)
            tick_labels.append(f"{note_name}{octave}")

    return tick_vals, tick_labels


def create_piano_roll_figure(
    loop: GeneratedLoop,
) -> go.Figure:
    """
    Create a dual piano roll visualization of a generated loop.

    The figure contains two subplots:
    - Top: Melody notes (orange)
    - Bottom: Harmony/chord notes (purple/blue)

    Args:
        loop: Generated loop containing melody and chord notes

    Returns:
        Plotly Figure object ready for display
    """
    # Create subplots with shared X-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Melody", "Harmony"),
        row_heights=[0.5, 0.5],
    )

    # Get pitch ranges for each track
    melody_min, melody_max = get_pitch_range(loop.melody_notes)
    chord_min, chord_max = get_pitch_range(loop.chord_notes)

    # Calculate total duration in beats
    total_sixteenths = loop.params.num_bars * loop.params.beats_per_bar * 4
    total_beats = sixteenth_to_beats(total_sixteenths)

    # Create shapes for notes
    melody_shapes = create_note_shapes(loop.melody_notes, MELODY_COLOR, row=1)
    chord_shapes = create_note_shapes(loop.chord_notes, HARMONY_COLOR, row=2)

    # Add invisible scatter traces to create hover info for melody
    if loop.melody_notes:
        fig.add_trace(
            go.Scatter(
                x=[
                    sixteenth_to_beats(n.start_time + n.duration / 2)
                    for n in loop.melody_notes
                ],
                y=[n.midi for n in loop.melody_notes],
                mode="markers",
                marker={"size": 1, "opacity": 0},
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Beat: %{x:.2f}<br>"
                    "Duration: %{customdata[1]:.2f} beats<br>"
                    "Velocity: %{customdata[2]}"
                    "<extra></extra>"
                ),
                customdata=[
                    [
                        midi_to_display_name(n.midi),
                        sixteenth_to_beats(n.duration),
                        n.velocity,
                    ]
                    for n in loop.melody_notes
                ],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Add invisible scatter traces for chord hover info
    if loop.chord_notes:
        fig.add_trace(
            go.Scatter(
                x=[
                    sixteenth_to_beats(n.start_time + n.duration / 2)
                    for n in loop.chord_notes
                ],
                y=[n.midi for n in loop.chord_notes],
                mode="markers",
                marker={"size": 1, "opacity": 0},
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Beat: %{x:.2f}<br>"
                    "Duration: %{customdata[1]:.2f} beats<br>"
                    "Velocity: %{customdata[2]}"
                    "<extra></extra>"
                ),
                customdata=[
                    [
                        midi_to_display_name(n.midi),
                        sixteenth_to_beats(n.duration),
                        n.velocity,
                    ]
                    for n in loop.chord_notes
                ],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Add all shapes to the figure
    fig.update_layout(shapes=melody_shapes + chord_shapes)

    # Get pitch labels
    melody_ticks, melody_labels = create_pitch_labels(melody_min, melody_max)
    chord_ticks, chord_labels = create_pitch_labels(chord_min, chord_max)

    # Create bar markers for X-axis
    bar_ticks = list(range(0, int(total_beats) + 1, loop.params.beats_per_bar))
    bar_labels = [f"Bar {i + 1}" if i > 0 else "Bar 1" for i in range(len(bar_ticks))]

    # Update layout
    fig.update_layout(
        title={
            "text": f"{loop.params.key} {loop.params.mode.capitalize()} | "
            f"{loop.params.num_bars} bars @ {loop.params.tempo_bpm} BPM",
            "font": {"color": TEXT_COLOR, "size": 14},
            "x": 0.5,
            "xanchor": "center",
        },
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font={"color": TEXT_COLOR},
        height=450,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        showlegend=False,
    )

    # Update X-axis (shared, only visible on bottom)
    fig.update_xaxes(
        range=[0, total_beats],
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickvals=bar_ticks,
        ticktext=bar_labels,
        tickfont={"size": 10},
        row=2,
        col=1,
    )
    fig.update_xaxes(
        range=[0, total_beats],
        gridcolor=GRID_COLOR,
        gridwidth=1,
        showticklabels=False,
        row=1,
        col=1,
    )

    # Update Y-axes
    fig.update_yaxes(
        range=[melody_min - 2, melody_max + 2],
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickvals=melody_ticks,
        ticktext=melody_labels,
        tickfont={"size": 9},
        title_text="",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[chord_min - 2, chord_max + 2],
        gridcolor=GRID_COLOR,
        gridwidth=1,
        tickvals=chord_ticks,
        ticktext=chord_labels,
        tickfont={"size": 9},
        title_text="",
        row=2,
        col=1,
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = {"color": TEXT_COLOR, "size": 12}

    return fig


def create_empty_figure() -> go.Figure:
    """
    Create an empty placeholder figure before any loop is generated.

    Returns:
        Empty Plotly Figure with "Generate a loop" message
    """
    fig = go.Figure()

    fig.add_annotation(
        text="Generate a loop to see visualization",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": TEXT_COLOR},
    )

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=450,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )

    return fig

"""
FluidSynth CLI wrapper for audio preview in MarkovMIDI.

Provides audio synthesis capabilities for previewing generated MIDI
loops using the FluidSynth command-line tool and SoundFonts.
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# Default FluidSynth settings
DEFAULT_SAMPLE_RATE: Final[int] = 44100
DEFAULT_GAIN: Final[float] = 0.5


@dataclass
class SynthesizerConfig:
    """
    Configuration for the FluidSynth synthesizer.

    Attributes:
        soundfont_path: Path to the .sf2 SoundFont file
        sample_rate: Audio sample rate in Hz
        gain: Output gain (0.0 to 1.0)
        fluidsynth_path: Path to FluidSynth executable (auto-detected if None)
    """

    soundfont_path: str | Path | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    gain: float = DEFAULT_GAIN
    fluidsynth_path: str | None = None


def find_fluidsynth() -> str | None:
    """
    Find the FluidSynth executable in the system PATH.

    Returns:
        Path to FluidSynth executable, or None if not found
    """
    # Try common executable names
    for name in ["fluidsynth", "fluidsynth.exe"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def is_fluidsynth_available(config: SynthesizerConfig | None = None) -> bool:
    """
    Check if FluidSynth is available on the system.

    Args:
        config: Optional config with custom FluidSynth path

    Returns:
        True if FluidSynth is available
    """
    if config and config.fluidsynth_path:
        return Path(config.fluidsynth_path).exists()
    return find_fluidsynth() is not None


def find_soundfonts(search_dirs: list[str | Path] | None = None) -> list[Path]:
    """
    Find SoundFont files in common locations.

    Args:
        search_dirs: Additional directories to search

    Returns:
        List of paths to .sf2 files
    """
    soundfonts: list[Path] = []

    # Default search directories
    dirs_to_search: list[Path] = [
        Path("soundfonts"),  # Project directory
        Path.home() / "soundfonts",
        Path("/usr/share/sounds/sf2"),  # Linux
        Path("/usr/share/soundfonts"),  # Linux
    ]

    # Add custom search dirs
    if search_dirs:
        dirs_to_search.extend(Path(d) for d in search_dirs)

    for directory in dirs_to_search:
        if directory.exists() and directory.is_dir():
            soundfonts.extend(directory.glob("*.sf2"))
            soundfonts.extend(directory.glob("*.SF2"))

    # Remove duplicates and sort
    unique_soundfonts = list(set(soundfonts))
    unique_soundfonts.sort(key=lambda p: p.name.lower())

    return unique_soundfonts


def render_midi_to_wav(
    midi_path: str | Path,
    output_path: str | Path,
    config: SynthesizerConfig,
) -> Path:
    """
    Render a MIDI file to WAV using FluidSynth.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path for output WAV file
        config: Synthesizer configuration

    Returns:
        Path to the rendered WAV file

    Raises:
        FileNotFoundError: If MIDI file or SoundFont not found
        RuntimeError: If FluidSynth is not available or rendering fails
    """
    midi_path = Path(midi_path)
    output_path = Path(output_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if not config.soundfont_path:
        raise FileNotFoundError("No SoundFont specified")

    soundfont_path = Path(config.soundfont_path)
    if not soundfont_path.exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

    # Find FluidSynth
    fluidsynth = config.fluidsynth_path or find_fluidsynth()
    if not fluidsynth:
        raise RuntimeError(
            "FluidSynth not found. Install FluidSynth and ensure it's in PATH."
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FluidSynth command
    cmd = [
        fluidsynth,
        "-ni",  # No interactive mode
        "-F",
        str(output_path),  # Output file
        "-r",
        str(config.sample_rate),  # Sample rate
        "-g",
        str(config.gain),  # Gain
        str(soundfont_path),  # SoundFont
        str(midi_path),  # MIDI file
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"FluidSynth failed: {result.stderr or result.stdout}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("FluidSynth timed out")
    except FileNotFoundError:
        raise RuntimeError(f"FluidSynth executable not found: {fluidsynth}")

    if not output_path.exists():
        raise RuntimeError("FluidSynth did not create output file")

    return output_path


def render_midi_to_temp_wav(
    midi_path: str | Path,
    config: SynthesizerConfig,
) -> Path:
    """
    Render a MIDI file to a temporary WAV file.

    The caller is responsible for cleaning up the temporary file.

    Args:
        midi_path: Path to input MIDI file
        config: Synthesizer configuration

    Returns:
        Path to the temporary WAV file
    """
    # Create temp file with .wav extension
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    # Close the file descriptor (we just need the path)
    import os

    os.close(temp_fd)

    return render_midi_to_wav(midi_path, temp_path, config)


class Synthesizer:
    """
    High-level synthesizer interface for audio preview.

    Example:
        >>> synth = Synthesizer(soundfont_path="soundfonts/piano.sf2")
        >>> if synth.is_available():
        ...     wav_path = synth.render("output/loop.mid", "output/loop.wav")
    """

    def __init__(
        self,
        soundfont_path: str | Path | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        gain: float = DEFAULT_GAIN,
        fluidsynth_path: str | None = None,
    ) -> None:
        """
        Initialize the synthesizer.

        Args:
            soundfont_path: Path to SoundFont file
            sample_rate: Audio sample rate
            gain: Output gain
            fluidsynth_path: Custom FluidSynth path
        """
        self.config = SynthesizerConfig(
            soundfont_path=soundfont_path,
            sample_rate=sample_rate,
            gain=gain,
            fluidsynth_path=fluidsynth_path,
        )

    def is_available(self) -> bool:
        """Check if synthesis is available (FluidSynth + SoundFont)."""
        if not is_fluidsynth_available(self.config):
            return False
        if not self.config.soundfont_path:
            return False
        return Path(self.config.soundfont_path).exists()

    def set_soundfont(self, soundfont_path: str | Path) -> None:
        """Set the SoundFont to use."""
        self.config.soundfont_path = soundfont_path

    def render(
        self,
        midi_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Render a MIDI file to WAV.

        Args:
            midi_path: Input MIDI file
            output_path: Output WAV file

        Returns:
            Path to rendered WAV file
        """
        return render_midi_to_wav(midi_path, output_path, self.config)

    def render_to_temp(self, midi_path: str | Path) -> Path:
        """
        Render MIDI to a temporary WAV file.

        Args:
            midi_path: Input MIDI file

        Returns:
            Path to temporary WAV file
        """
        return render_midi_to_temp_wav(midi_path, self.config)

    def get_available_soundfonts(
        self,
        search_dirs: list[str | Path] | None = None,
    ) -> list[Path]:
        """
        Find available SoundFont files.

        Args:
            search_dirs: Additional directories to search

        Returns:
            List of SoundFont file paths
        """
        return find_soundfonts(search_dirs)

    def __repr__(self) -> str:
        sf = self.config.soundfont_path or "None"
        return f"Synthesizer(soundfont={sf}, available={self.is_available()})"

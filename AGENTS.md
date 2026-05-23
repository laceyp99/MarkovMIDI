# AGENTS.md - Guidelines for AI Coding Agents

This document provides conventions and commands for AI agents working on MarkovMIDI.

## Build, Lint, and Test Commands

### Package Installation
```bash
pip install -e .                    # Editable install
pip install -r requirements.txt     # Production dependencies
pip install -r requirements-dev.txt # Development dependencies
```

### Running Tests
```bash
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest --tb=short                   # Short tracebacks (default in pyproject.toml)
pytest --cov                        # With coverage

# Run single test file
pytest tests/test_model.py

# Run single test class
pytest tests/test_model.py::TestMarkovChainBasics

# Run single test method
pytest tests/test_model.py::TestMarkovChainBasics::test_init_empty

# Run tests matching pattern
pytest -k "test_chord"
```

### Type Checking
```bash
mypy markov_midi                    # Check entire package
mypy markov_midi/model/reward.py    # Check single file
```

### Running the Application
```bash
python -m markov_midi               # Launch Gradio UI
```

## Code Style Guidelines

### Import Order and Format

Always use this exact ordering with blank lines between sections:

```python
"""Module docstring."""

from __future__ import annotations  # ALWAYS first

# Standard library
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TYPE_CHECKING

# Third-party
import numpy as np
import mido

# Local imports
from markov_midi.model.markov_chain import MarkovChain
from markov_midi.model.theory_priors import (
    CHORD_DEGREES,
    create_chord_chain,
)

if TYPE_CHECKING:
    from markov_midi.model.chord_model import ChordModel
```

### Type Annotations

- **All functions must have complete type annotations** including return types
- Use `-> None` for functions that don't return a value
- Use `|` syntax for unions: `str | None`, not `Optional[str]`
- Use lowercase generics: `list[str]`, `dict[str, Any]`, `tuple[int, ...]`
- Use `Final` for constants: `NAME: Final[str] = "value"`

```python
def process_data(
    items: list[str],
    config: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[int], bool]:
    """Process items and return results."""
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Files | lowercase_underscore | `markov_chain.py` |
| Classes | PascalCase | `ChordModel`, `VoicedNote` |
| Functions/Methods | snake_case | `apply_reward`, `to_dict` |
| Variables | snake_case | `chord_chain`, `total_duration` |
| Constants | SCREAMING_SNAKE | `CHORD_DEGREES`, `DEFAULT_TEMPO` |
| Private | leading underscore | `_compute_probs`, `_cache` |
| Enums | PascalCase class, SCREAMING values | `VoicingStyle.BLOCK` |

### Docstrings (Google Style)

```python
def calculate_reward(
    rating: int,
    weight: float = 1.0,
) -> float:
    """
    Calculate weighted reward from a rating value.

    Converts a 1-5 rating to a -1 to +1 reward scale,
    then applies the weight multiplier.

    Args:
        rating: User rating from 1 (bad) to 5 (good)
        weight: Multiplier for the final reward

    Returns:
        Weighted reward value in range [-weight, +weight]

    Raises:
        ValueError: If rating is not in range 1-5
    """
```

### Error Handling

Use `ValueError` for invalid inputs with descriptive messages:

```python
if not 1 <= value <= 5:
    raise ValueError(f"Rating must be 1-5, got {value}")

if key not in VALID_KEYS:
    raise ValueError(f"Invalid key: {key}")
```

Use dataclass `__post_init__` for validation:

```python
@dataclass
class Rating:
    melodic: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.melodic <= 5:
            raise ValueError(f"melodic must be 1-5, got {self.melodic}")
```

Return `None` or `False` for "not found" cases instead of raising:

```python
def get_record(self, id: str) -> Record | None:
    return self._records.get(id)

def apply_rating(self, id: str, rating: Rating) -> bool:
    record = self.get_record(id)
    if not record:
        return False
    # ... apply rating
    return True
```

### Serialization Pattern

All persistent classes implement `to_dict` and `from_dict`:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "field": self.field,
        "nested": self.nested.to_dict() if self.nested else None,
        "enum": self.enum.value,
    }

@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ClassName":
    return cls(
        field=data.get("field", default),
        nested=Nested.from_dict(data["nested"]) if data.get("nested") else None,
        enum=EnumClass(data.get("enum", "default")),
    )
```

### Test Conventions

```python
class TestClassName:
    """Tests for ClassName."""

    def test_specific_behavior(self) -> None:
        """Description of what is being tested."""
        # Arrange
        obj = ClassName()

        # Act
        result = obj.method()

        # Assert
        assert result == expected

    def test_raises_on_invalid(self) -> None:
        """Invalid input raises ValueError."""
        with pytest.raises(ValueError):
            ClassName(invalid=-1)
```

- All test methods return `-> None`
- Use `tempfile.TemporaryDirectory()` for file tests
- Use `random.Random(seed)` for reproducibility tests
- Test serialization round-trips: `obj == Class.from_dict(obj.to_dict())`

## Package Structure

```
markov_midi/
    __init__.py          # Package exports
    __main__.py          # CLI entry point
    model/               # Core models (MarkovChain, ChordModel, MelodyModel)
    generator/           # Loop generation, MIDI writing, voicing
    parser/              # MIDI file parsing
    utils/               # Music theory, quantization
    audio/               # Synthesizer for audio playback
    ui/                  # Gradio web interface
tests/
    test_model.py        # Tests for model/
    test_generator.py    # Tests for generator/
    test_utils.py        # Tests for utils/ and audio/
    test_ui.py           # Tests for ui/
```

## Key Dependencies

- **mido**: MIDI file I/O
- **gradio**: Web UI framework
- **numpy**: Probability calculations
- **plotly**: Piano roll visualization

## Common Patterns

### Avoid mutable default arguments
```python
# Correct
def __init__(self, items: list[str] | None = None) -> None:
    self.items = items.copy() if items else []
```

### Use `from __future__ import annotations`
This enables forward references and modern type syntax. Always include it.

### Factory functions for complex initialization
```python
def create_chord_chain(smoothing: float = 1.0) -> MarkovChain[int]:
    """Create pre-initialized chord chain with theory priors."""
    chain: MarkovChain[int] = MarkovChain(smoothing=smoothing)
    # ... setup
    return chain
```

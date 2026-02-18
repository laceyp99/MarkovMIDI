"""
Persistence utilities for MarkovMIDI.

Handles saving and loading of:
- Model states (Markov chain transition counts)
- Training sessions (models + reward history)
- JSON serialization/deserialization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from markov_midi.model.reward import RewardManager


@dataclass
class SessionMetadata:
    """
    Metadata for a saved session.

    Attributes:
        name: Session name
        created_at: When the session was created
        updated_at: When the session was last updated
        description: Optional description
        version: Schema version for compatibility
    """

    name: str = "Untitled Session"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    version: str = "1.0"

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "SessionMetadata":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Untitled Session"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )


@dataclass
class Session:
    """
    A complete training session with models and history.

    Attributes:
        metadata: Session metadata
        generator_state: Serialized LoopGenerator state
        reward_manager_state: Serialized RewardManager state
    """

    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    generator_state: dict[str, Any] = field(default_factory=dict)
    reward_manager_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "generator": self.generator_state,
            "reward_manager": self.reward_manager_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            metadata=SessionMetadata.from_dict(data.get("metadata", {})),
            generator_state=data.get("generator", {}),
            reward_manager_state=data.get("reward_manager", {}),
        )


def save_session(
    session: Session,
    file_path: str | Path,
    pretty: bool = True,
) -> Path:
    """
    Save a session to a JSON file.

    Args:
        session: Session to save
        file_path: Output file path
        pretty: If True, format JSON with indentation

    Returns:
        Path to saved file
    """
    path = Path(file_path)

    # Ensure .json extension
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    session.metadata.touch()

    # Serialize
    data = session.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

    return path


def load_session(file_path: str | Path) -> Session:
    """
    Load a session from a JSON file.

    Args:
        file_path: Path to session file

    Returns:
        Loaded Session

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSON or session format
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in session file: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Session file must contain a JSON object")

    return Session.from_dict(data)


def create_session_from_generator(
    generator: Any,  # LoopGenerator
    reward_manager: RewardManager | None = None,
    name: str = "Untitled Session",
    description: str = "",
) -> Session:
    """
    Create a session from a generator and optional reward manager.

    Args:
        generator: LoopGenerator to save
        reward_manager: Optional RewardManager to save
        name: Session name
        description: Session description

    Returns:
        Session ready for saving
    """
    session = Session(
        metadata=SessionMetadata(
            name=name,
            description=description,
        ),
        generator_state=generator.to_dict(),
        reward_manager_state=reward_manager.to_dict() if reward_manager else {},
    )

    return session


def restore_session(
    session: Session,
) -> tuple[Any, RewardManager | None]:
    """
    Restore a generator and reward manager from a session.

    Args:
        session: Session to restore from

    Returns:
        Tuple of (LoopGenerator, RewardManager or None)
    """
    # Import here to avoid circular imports
    from markov_midi.generator.loop_generator import LoopGenerator

    generator = LoopGenerator.from_dict(session.generator_state)

    reward_manager = None
    if session.reward_manager_state:
        reward_manager = RewardManager.from_dict(session.reward_manager_state)

    return generator, reward_manager


def list_sessions(directory: str | Path = "sessions") -> list[dict[str, Any]]:
    """
    List all sessions in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of session info dicts with name, path, and metadata
    """
    dir_path = Path(directory)
    sessions: list[dict[str, Any]] = []

    if not dir_path.exists():
        return sessions

    for file_path in dir_path.glob("*.json"):
        try:
            session = load_session(file_path)
            sessions.append(
                {
                    "path": str(file_path),
                    "name": session.metadata.name,
                    "created_at": session.metadata.created_at,
                    "updated_at": session.metadata.updated_at,
                    "description": session.metadata.description,
                }
            )
        except (ValueError, KeyError):
            # Skip invalid session files
            continue

    # Sort by updated_at, most recent first
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)

    return sessions


def save_model_only(
    generator: Any,  # LoopGenerator
    file_path: str | Path,
) -> Path:
    """
    Save just the model state (without reward history).

    Useful for sharing trained models without session history.

    Args:
        generator: LoopGenerator to save
        file_path: Output file path

    Returns:
        Path to saved file
    """
    path = Path(file_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "type": "model",
        "saved_at": datetime.now().isoformat(),
        "model": generator.to_dict(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def load_model_only(file_path: str | Path) -> Any:
    """
    Load just a model state file.

    Args:
        file_path: Path to model file

    Returns:
        LoopGenerator with loaded state

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid
    """
    from markov_midi.generator.loop_generator import LoopGenerator

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "model":
        raise ValueError("File is not a model file")

    return LoopGenerator.from_dict(data["model"])


def get_session_path(name: str, directory: str | Path = "sessions") -> Path:
    """
    Get the file path for a session by name.

    Args:
        name: Session name
        directory: Sessions directory

    Returns:
        Path for the session file
    """
    # Sanitize name for filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()

    if not safe_name:
        safe_name = "session"

    return Path(directory) / f"{safe_name}.json"

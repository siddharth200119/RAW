"""LocalStorage — filesystem-backed storage backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from RAW.metrics import Metrics
from RAW.models.file import File
from .base import Storage


class LocalStorage(Storage):
    """Stores files on the local filesystem under a configurable root directory.

    Args:
        root:    Base directory for all managed files.  Defaults to the current
                 working directory.  Sub-directories are created automatically.
        metrics: Optional :class:`~RAW.metrics.Metrics` implementation.
                 Defaults to :class:`~RAW.metrics.NullMetrics` (no-op).

    Example::

        storage = LocalStorage(root="/var/app/uploads")
        with storage.open(file) as stream:
            contents = stream.read()
    """

    def __init__(
        self,
        root: str | Path = ".",
        *,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(metrics=metrics)
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, file: File) -> Path:
        """Return the absolute :class:`~pathlib.Path` for *file*."""
        return (self._root / file.location).resolve()

    # ------------------------------------------------------------------
    # Storage interface
    # ------------------------------------------------------------------

    def open(self, file: File) -> BinaryIO:
        """Open *file* for reading.

        Returns:
            A binary file object.  The caller must close it (context manager
            recommended).

        Raises:
            FileNotFoundError: If the path does not exist on disk.
        """
        with self._timed_op("open", direction="download") as ctx:
            path = self._resolve(file)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            ctx.bytes_transferred = path.stat().st_size
            return path.open("rb")

    def save(self, file: File, data: BinaryIO) -> None:
        """Write *data* to the local filesystem.

        Parent directories are created automatically if they do not already
        exist, so callers need not pre-create any directory structure.

        Args:
            file: Metadata describing the target location.
            data: Readable binary stream.  Position is assumed to be at 0.
        """
        with self._timed_op("save", direction="upload") as ctx:
            path = self._resolve(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as dest:
                shutil.copyfileobj(data, dest)
            ctx.bytes_transferred = path.stat().st_size

    def delete(self, file: File) -> None:
        """Remove *file* from the filesystem.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        with self._timed_op("delete"):
            path = self._resolve(file)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            path.unlink()

    def exists(self, file: File) -> bool:
        """Return ``True`` if the file exists on disk."""
        with self._timed_op("exists"):
            return self._resolve(file).exists()

    def generate_presigned_url(
        self,
        file: File,
        *,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> str | None:
        """Not supported for local storage. Always returns ``None``."""
        return None

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The resolved root directory managed by this instance."""
        return self._root

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LocalStorage root={self._root!r}>"

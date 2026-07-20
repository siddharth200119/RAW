"""Abstract base class that all storage backends must implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import BinaryIO, Generator

from RAW.metrics import Metrics, NullMetrics
from RAW.models.file import File


# ---------------------------------------------------------------------------
# Internal operation context
# ---------------------------------------------------------------------------

@dataclass
class _OpContext:
    """Mutable context yielded by :meth:`Storage._timed_op`.

    Backends can set ``bytes_transferred`` before the ``with`` block exits
    and the base class will include it in the emitted metrics automatically.
    """
    bytes_transferred: int = field(default=0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Storage(ABC):
    """Protocol-like ABC for file storage backends.

    Every public operation is expressed in terms of a :class:`~RAW.models.file.File`
    instance so callers never touch the filesystem or any SDK directly.

    Implementing a new backend (S3, Azure Blob, GCS, …) requires only:

    1. Subclass :class:`Storage`.
    2. Implement the four abstract methods below.
    3. Optionally pass a :class:`~RAW.metrics.Metrics` instance to ``__init__``.

    Args:
        metrics: A :class:`~RAW.metrics.Metrics` implementation for emitting
                 operation counters, durations, and byte throughput.
                 Defaults to :class:`~RAW.metrics.NullMetrics` (no-op).

    Metric names emitted:

    +-----------------------------------------------+------+-------------------------------------+
    | Name                                          | Kind | Tags                                |
    +===============================================+======+=====================================+
    | ``raw.storage.operations``                    | ctr  | ``operation``, ``status``           |
    +-----------------------------------------------+------+-------------------------------------+
    | ``raw.storage.operation_duration_seconds``    | hist | ``operation``                       |
    +-----------------------------------------------+------+-------------------------------------+
    | ``raw.storage.bytes_transferred``             | ctr  | ``direction`` (upload / download)   |
    +-----------------------------------------------+------+-------------------------------------+
    """

    def __init__(self, metrics: Metrics | None = None) -> None:
        self._metrics: Metrics = metrics or NullMetrics()

    # ------------------------------------------------------------------
    # Internal timing helper
    # ------------------------------------------------------------------

    @contextmanager
    def _timed_op(
        self,
        operation: str,
        *,
        direction: str | None = None,
    ) -> Generator[_OpContext, None, None]:
        """Context manager that times a storage operation and emits metrics.

        Usage inside a backend::

            def save(self, file: File, data: BinaryIO) -> None:
                with self._timed_op("save", direction="upload") as ctx:
                    # ... perform the write ...
                    ctx.bytes_transferred = bytes_written

        Args:
            operation: Short name for the operation (``"open"``, ``"save"``,
                       ``"delete"``, ``"exists"``, etc.).
            direction: Optional transfer direction for byte tracking —
                       ``"upload"`` or ``"download"``.  Only emits a byte
                       counter when set *and* ``ctx.bytes_transferred > 0``.
        """
        start = time.perf_counter()
        status = "success"
        ctx = _OpContext()
        try:
            yield ctx
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            self._metrics.increment(
                "raw.storage.operations",
                tags={"operation": operation, "status": status},
            )
            self._metrics.histogram(
                "raw.storage.operation_duration_seconds",
                duration,
                tags={"operation": operation},
            )
            if direction and ctx.bytes_transferred > 0:
                self._metrics.increment(
                    "raw.storage.bytes_transferred",
                    ctx.bytes_transferred,
                    tags={"direction": direction},
                )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self, file: File) -> BinaryIO:
        """Return a **readable** binary stream for *file*.

        The caller is responsible for closing the returned stream (using it as
        a context manager is strongly recommended)::

            with storage.open(file) as stream:
                data = stream.read()

        Raises:
            FileNotFoundError: If the file does not exist in the backend.
        """
        ...

    @abstractmethod
    def save(self, file: File, data: BinaryIO) -> None:
        """Persist *data* to the backend at the location described by *file*.

        Args:
            file: Metadata describing where to store the file.
            data: A **readable** binary stream whose contents will be written.
                  The stream position is assumed to be at the beginning; if not,
                  seek to 0 before calling this method.
        """
        ...

    @abstractmethod
    def delete(self, file: File) -> None:
        """Remove *file* from the backend.

        Raises:
            FileNotFoundError: If the file does not exist in the backend.
        """
        ...

    @abstractmethod
    def exists(self, file: File) -> bool:
        """Return ``True`` if *file* exists in the backend, ``False`` otherwise."""
        ...

    @abstractmethod
    def generate_presigned_url(
        self,
        file: File,
        *,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> str | None:
        """Generate a presigned URL for *file*.

        Returns:
            A presigned URL string, or ``None`` if generation failed or is
            unsupported by the underlying backend (e.g., local storage).
        """
        ...

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__}>"

"""File model and related enumerations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from RAW.storage.base import Storage


class FileType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    EXCEL = "excel"
    PDF = "pdf"
    WORD = "word"
    ZIP = "zip"
    UNKNOWN = "unknown"


SUPPORTED_EXTENSIONS: dict[FileType, set[str]] = {
    FileType.IMAGE: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"},
    FileType.TEXT: {".txt", ".md", ".csv"},
    FileType.EXCEL: {".xls", ".xlsx", ".xlsm", ".ods"},
    FileType.PDF: {".pdf"},
    FileType.ZIP: {".zip"},
    FileType.WORD: {".doc", ".docx", ".odt"},
}


@dataclass(slots=True)
class File:
    """Represents a file reference managed by a :class:`~RAW.storage.Storage` backend.

    Attributes:
        name:      Human-readable filename (e.g. ``"report.xlsx"``).
        file_type: Semantic type of the file content.
        storage:   The :class:`~RAW.storage.Storage` instance that owns this
                   file.  All I/O must go through this object.
        location:  Absolute local path **or** object-storage key — interpreted
                   by the attached ``storage`` backend.
        bucket:    Bucket / container name required by object-storage backends.
                   ``None`` for local storage.
    """

    name: str
    file_type: FileType
    storage: Storage
    location: str
    bucket: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience I/O methods
    # ------------------------------------------------------------------

    def open(self) -> BinaryIO:
        """Open this file for reading via its storage backend.

        Delegates to :meth:`~RAW.storage.Storage.open` so the caller never
        needs to reference ``file.storage`` directly.  The returned stream
        must be closed by the caller — a context manager is recommended::

            with my_file.open() as stream:
                data = stream.read()

        Returns:
            A readable binary stream (:class:`~typing.BinaryIO`).

        Raises:
            FileNotFoundError: If the file does not exist in the backend.
        """
        return self.storage.open(self)

    def download(self, dest: str | Path | None = None) -> Path:
        """Download this file to the local filesystem.

        Streams the file content without loading it fully into memory.

        Args:
            dest: Local path to write the file to.  If ``None`` (default),
                  a temporary file is created automatically.  The temp file
                  is **not** deleted automatically — the caller is responsible
                  for cleanup when it's no longer needed.

        Returns:
            :class:`~pathlib.Path` pointing to the downloaded file on disk.

        Raises:
            FileNotFoundError: If the file does not exist in the backend.

        Example::

            # Download to a temp file
            path = my_file.download()
            process(path)
            path.unlink()   # clean up when done

            # Download to a specific location
            path = my_file.download("/tmp/report.pdf")
        """
        import shutil
        import tempfile
        from pathlib import Path

        suffix = Path(self.name).suffix

        if dest is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            dest_path = Path(tmp.name)
            tmp.close()
        else:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        with self.storage.open(self) as stream:
            with dest_path.open("wb") as out:
                shutil.copyfileobj(stream, out)

        return dest_path
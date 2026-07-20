"""File model and related enumerations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

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
"""Storage abstraction layer for the RAW package.

Public surface::

    from RAW.storage import Storage, LocalStorage, S3Storage

The concrete backends implement the :class:`Storage` ABC, so callers only
ever depend on the abstract interface – swapping backends requires no changes
to the rest of the application.
"""

from .base import Storage
from .local import LocalStorage
from .s3 import S3Storage

__all__ = ["Storage", "LocalStorage", "S3Storage"]

"""S3Storage — AWS S3 / S3-compatible object-storage backend.

Uses ``boto3`` and works with any S3-compatible endpoint (MinIO, Tigris,
Cloudflare R2, Backblaze B2, …).  Pass a custom ``endpoint_url`` when
constructing the boto3 client to point at a non-AWS provider::

    import boto3
    from RAW.storage import S3Storage

    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",   # MinIO / any S3-compatible
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    )
    storage = S3Storage(client=client)

Files are **never fully loaded into memory** — uploads use ``upload_fileobj``
and downloads wrap the boto3 ``StreamingBody`` in a standard
:class:`io.RawIOBase` so callers always receive a uniform :class:`~typing.BinaryIO`.
"""

from __future__ import annotations

import io
from typing import BinaryIO, Optional

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from RAW.metrics import Metrics
from RAW.models.file import File
from .base import Storage


class S3Storage(Storage):
    """Object-storage backend backed by an S3-compatible service.

    Args:
        client:         An already-configured ``boto3`` S3 client
                        (``boto3.client("s3", ...)``).  Injecting the client
                        keeps credentials, endpoint, and retry config fully
                        under the caller's control and makes unit testing
                        trivial.
        default_bucket: Optional bucket to fall back to when ``file.bucket``
                        is ``None``.  If neither is set a :class:`ValueError`
                        is raised at call time.
        metrics:        Optional :class:`~RAW.metrics.Metrics` implementation.
                        Defaults to :class:`~RAW.metrics.NullMetrics` (no-op).

    Note:
        Bucket auto-creation is intentionally **not** performed.  Creating
        buckets is an administrative action; callers should provision buckets
        outside application code.
    """

    def __init__(
        self,
        client: BaseClient,
        *,
        default_bucket: str | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(metrics=metrics)
        self._client = client
        self._default_bucket = default_bucket

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_bucket(self, file: File) -> str:
        bucket = file.bucket or self._default_bucket
        if not bucket:
            raise ValueError(
                f"File '{file.name}' has no bucket set and no default_bucket "
                "was configured on S3Storage."
            )
        return bucket

    def _is_not_found(self, exc: ClientError) -> bool:
        code = exc.response["Error"]["Code"]
        return code in ("NoSuchKey", "NoSuchBucket", "404")

    # ------------------------------------------------------------------
    # Storage interface
    # ------------------------------------------------------------------

    def open(self, file: File) -> BinaryIO:
        """Stream *file* from S3.

        Returns a :class:`io.RawIOBase`-compatible object backed by the boto3
        ``StreamingBody``.  The caller **must** close the stream (context
        manager recommended) to release the underlying HTTP connection::

            with file.storage.open(file) as stream:
                chunk = stream.read(8192)

        Raises:
            FileNotFoundError: If the object key does not exist in the bucket.
            ValueError:        If no bucket can be resolved.
        """
        with self._timed_op("open", direction="download") as ctx:
            bucket = self._resolve_bucket(file)
            try:
                response = self._client.get_object(Bucket=bucket, Key=file.location)
            except ClientError as exc:
                if self._is_not_found(exc):
                    raise FileNotFoundError(
                        f"Object '{file.location}' not found in bucket '{bucket}'."
                    ) from exc
                raise
            ctx.bytes_transferred = response.get("ContentLength", 0)
            return _StreamingBodyWrapper(response["Body"])

    def save(self, file: File, data: BinaryIO) -> None:
        """Upload *data* to S3 using a streaming multipart upload.

        ``upload_fileobj`` automatically uses multipart upload for large
        objects and streams data without loading it fully into memory.

        Args:
            file: Metadata describing the target key and bucket.
            data: Readable binary stream positioned at the beginning.

        Raises:
            ValueError: If no bucket can be resolved.
        """
        with self._timed_op("save", direction="upload") as ctx:
            bucket = self._resolve_bucket(file)
            # Wrap in a counting stream so we know how many bytes were sent
            counted = _CountingStream(data)
            self._client.upload_fileobj(counted, bucket, file.location)
            ctx.bytes_transferred = counted.bytes_read

    def delete(self, file: File) -> None:
        """Remove an object from S3.

        Raises:
            FileNotFoundError: If the object does not exist.
            ValueError:        If no bucket can be resolved.
        """
        with self._timed_op("delete"):
            bucket = self._resolve_bucket(file)
            if not self.exists(file):
                raise FileNotFoundError(
                    f"Object '{file.location}' not found in bucket '{bucket}'."
                )
            self._client.delete_object(Bucket=bucket, Key=file.location)

    def exists(self, file: File) -> bool:
        """Return ``True`` if the object exists.

        Uses ``head_object`` — only metadata is fetched, no object body is
        transferred.

        Raises:
            ValueError: If no bucket can be resolved.
        """
        with self._timed_op("exists"):
            bucket = self._resolve_bucket(file)
            try:
                self._client.head_object(Bucket=bucket, Key=file.location)
                return True
            except ClientError as exc:
                if self._is_not_found(exc):
                    return False
                raise

    # ------------------------------------------------------------------
    # S3-specific extras (not on the Storage ABC)
    # ------------------------------------------------------------------

    def generate_presigned_url(
        self,
        file: File,
        *,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> Optional[str]:
        """Generate a presigned URL for *file*.

        Args:
            file:       The file whose key/bucket will be encoded in the URL.
            expiration: Lifetime of the URL in seconds (default: 1 hour).
            method:     boto3 client method to sign — ``"get_object"``
                        (download) or ``"put_object"`` (upload).

        Returns:
            A presigned URL string, or ``None`` if generation failed.

        Raises:
            ValueError: If no bucket can be resolved.
        """
        with self._timed_op("presign"):
            bucket = self._resolve_bucket(file)
            try:
                return self._client.generate_presigned_url(
                    ClientMethod=method,
                    Params={"Bucket": bucket, "Key": file.location},
                    ExpiresIn=expiration,
                )
            except ClientError:
                return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<S3Storage default_bucket={self._default_bucket!r}>"


# ---------------------------------------------------------------------------
# Internal stream helpers
# ---------------------------------------------------------------------------

class _StreamingBodyWrapper(io.RawIOBase):
    """Wraps a boto3 ``StreamingBody`` to expose the standard
    :class:`io.RawIOBase` interface so all storage backends return a
    uniform :class:`~typing.BinaryIO` to callers.

    Closing the wrapper also closes the underlying HTTP response.
    """

    def __init__(self, body) -> None:  # body: botocore.response.StreamingBody
        self._body = body

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        chunk = self._body.read(len(b))
        n = len(chunk)
        b[:n] = chunk
        return n

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        return self._body.read() if size == -1 else self._body.read(size)

    def close(self) -> None:
        self._body.close()
        super().close()


class _CountingStream(io.RawIOBase):
    """Wraps a readable :class:`~typing.BinaryIO` and tracks how many bytes
    have been read from it.  Used by :meth:`S3Storage.save` to report upload
    byte counts to the metrics layer without buffering the entire stream.
    """

    def __init__(self, inner: BinaryIO) -> None:
        self._inner = inner
        self.bytes_read: int = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        chunk = self._inner.read() if size == -1 else self._inner.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def readinto(self, b: bytearray) -> int:
        chunk = self._inner.read(len(b))
        n = len(chunk)
        b[:n] = chunk
        self.bytes_read += n
        return n

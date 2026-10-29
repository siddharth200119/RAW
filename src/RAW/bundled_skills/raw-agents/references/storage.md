# Files and storage

`File` is a reference to bytes held by a storage backend; the same code works on local disk and S3.

```python
from RAW.utils.storage import LocalStorage, S3Storage
from RAW.models import File, FileType

storage = LocalStorage(root="/tmp/uploads")
# S3 / MinIO / R2 (needs raw-agents[s3]):
# import boto3
# storage = S3Storage(client=boto3.client("s3", endpoint_url="http://localhost:9000"), default_bucket="my-bucket")

f = File(name="doc.pdf", file_type=FileType.PDF, storage=storage, location="user_uploads/doc.pdf")

with open("local.pdf", "rb") as src:
    f.storage.save(f, src)              # streams
with f.storage.open(f) as stream:       # streams
    data = stream.read()
f.storage.exists(f)
f.storage.delete(f)
path = f.download()                     # copy to a local temp path (or f.download(dest))
```

- `FileType`: `IMAGE, TEXT, EXCEL, PDF, WORD, ZIP, UNKNOWN`.
- Storage calls are synchronous.
- Pass `metrics=` to a storage backend to get operation counts, durations and byte counters (`Metrics` protocol, see `observability.md`).
- Implement your own backend by subclassing `Storage` (`RAW.utils.storage`): provide `open`, `save`, `delete`, `exists`.
- Images given to an `Agent` via `user_files` are read with `Image.from_file(file.download())`; `Image` needs the `[agentic]` extra.

import io
import pytest
from RAW.models.file import File, FileType
from RAW.utils.storage import LocalStorage


@pytest.fixture
def storage(tmp_path, metrics):
    return LocalStorage(root=tmp_path / "root", metrics=metrics)


def make_file(storage, location="docs/note.txt"):
    return File(name="note.txt", file_type=FileType.TEXT, storage=storage, location=location)


def test_root_is_created_and_resolved(tmp_path):
    storage = LocalStorage(root=tmp_path / "a" / "b")
    assert storage.root == (tmp_path / "a" / "b").resolve()
    assert storage.root.is_dir()


def test_save_open_roundtrip_creates_parent_dirs(storage):
    f = make_file(storage)
    storage.save(f, io.BytesIO(b"hello"))

    assert (storage.root / "docs" / "note.txt").read_bytes() == b"hello"
    with storage.open(f) as stream:
        assert stream.read() == b"hello"


def test_exists(storage):
    f = make_file(storage)
    assert storage.exists(f) is False
    storage.save(f, io.BytesIO(b"x"))
    assert storage.exists(f) is True


def test_delete(storage):
    f = make_file(storage)
    storage.save(f, io.BytesIO(b"x"))
    storage.delete(f)
    assert not storage.exists(f)


def test_open_missing_raises(storage):
    with pytest.raises(FileNotFoundError):
        storage.open(make_file(storage))


def test_delete_missing_raises(storage):
    with pytest.raises(FileNotFoundError):
        storage.delete(make_file(storage))


def test_presigned_url_is_unsupported(storage):
    assert storage.generate_presigned_url(make_file(storage)) is None


def test_byte_metrics(storage, metrics):
    f = make_file(storage)
    storage.save(f, io.BytesIO(b"hello"))
    with storage.open(f):
        pass

    assert metrics.named("raw.storage.bytes_transferred") == [
        ("increment", "raw.storage.bytes_transferred", 5, {"direction": "upload"}),
        ("increment", "raw.storage.bytes_transferred", 5, {"direction": "download"}),
    ]


def test_failed_open_records_error(storage, metrics):
    with pytest.raises(FileNotFoundError):
        storage.open(make_file(storage))
    assert metrics.named("raw.storage.operations")[-1][3] == {"operation": "open", "status": "error"}


def test_file_convenience_methods(storage, tmp_path):
    f = make_file(storage)
    storage.save(f, io.BytesIO(b"data"))

    with f.open() as stream:
        assert stream.read() == b"data"

    dest = f.download(tmp_path / "out" / "copy.txt")
    assert dest.read_bytes() == b"data"

    tmp = f.download()
    try:
        assert tmp.suffix == ".txt"
        assert tmp.read_bytes() == b"data"
    finally:
        tmp.unlink()

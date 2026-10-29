import io
import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from RAW.models.file import File, FileType
from RAW.utils.storage import S3Storage


def client_error(code, op="HeadObject"):
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


@pytest.fixture
def client():
    return MagicMock()


@pytest.fixture
def storage(client, metrics):
    return S3Storage(client=client, default_bucket="default-bucket", metrics=metrics)


def make_file(storage, bucket=None):
    return File(name="a.pdf", file_type=FileType.PDF, storage=storage, location="k/a.pdf", bucket=bucket)


def test_file_bucket_overrides_default(storage, client):
    storage.exists(make_file(storage, bucket="own-bucket"))
    client.head_object.assert_called_once_with(Bucket="own-bucket", Key="k/a.pdf")


def test_missing_bucket_raises(client):
    storage = S3Storage(client=client)
    with pytest.raises(ValueError):
        storage.exists(make_file(storage))


def test_open_wraps_streaming_body(storage, client, metrics):
    body = MagicMock()
    body.read.side_effect = lambda size=None: b"pdf-bytes"
    client.get_object.return_value = {"Body": body, "ContentLength": 9}

    with storage.open(make_file(storage)) as stream:
        assert stream.readable()
        assert stream.read() == b"pdf-bytes"

    client.get_object.assert_called_once_with(Bucket="default-bucket", Key="k/a.pdf")
    body.close.assert_called_once()
    assert metrics.named("raw.storage.bytes_transferred") == [
        ("increment", "raw.storage.bytes_transferred", 9, {"direction": "download"})
    ]


def test_open_streaming_readinto(storage, client):
    body = MagicMock()
    body.read.return_value = b"abc"
    client.get_object.return_value = {"Body": body, "ContentLength": 3}

    stream = storage.open(make_file(storage))
    buf = bytearray(8)
    assert stream.readinto(buf) == 3
    assert bytes(buf[:3]) == b"abc"
    body.read.assert_called_with(8)


@pytest.mark.parametrize("code", ["NoSuchKey", "NoSuchBucket", "404"])
def test_open_not_found(storage, client, code):
    client.get_object.side_effect = client_error(code, "GetObject")
    with pytest.raises(FileNotFoundError):
        storage.open(make_file(storage))


def test_open_other_client_error_propagates(storage, client):
    client.get_object.side_effect = client_error("AccessDenied", "GetObject")
    with pytest.raises(ClientError):
        storage.open(make_file(storage))


def test_save_uploads_and_counts_bytes(storage, client, metrics):
    def fake_upload(fileobj, bucket, key):
        while fileobj.read(2):
            pass

    client.upload_fileobj.side_effect = fake_upload
    storage.save(make_file(storage), io.BytesIO(b"12345"))

    _, bucket, key = client.upload_fileobj.call_args[0]
    assert (bucket, key) == ("default-bucket", "k/a.pdf")
    assert metrics.named("raw.storage.bytes_transferred") == [
        ("increment", "raw.storage.bytes_transferred", 5, {"direction": "upload"})
    ]


def test_exists(storage, client):
    assert storage.exists(make_file(storage)) is True
    client.head_object.side_effect = client_error("404")
    assert storage.exists(make_file(storage)) is False


def test_exists_other_error_propagates(storage, client):
    client.head_object.side_effect = client_error("403")
    with pytest.raises(ClientError):
        storage.exists(make_file(storage))


def test_delete(storage, client):
    storage.delete(make_file(storage))
    client.delete_object.assert_called_once_with(Bucket="default-bucket", Key="k/a.pdf")


def test_delete_missing_raises(storage, client):
    client.head_object.side_effect = client_error("NoSuchKey")
    with pytest.raises(FileNotFoundError):
        storage.delete(make_file(storage))
    client.delete_object.assert_not_called()


def test_presigned_url(storage, client):
    client.generate_presigned_url.return_value = "https://signed"
    url = storage.generate_presigned_url(make_file(storage), expiration=60, method="put_object")

    assert url == "https://signed"
    client.generate_presigned_url.assert_called_once_with(
        ClientMethod="put_object",
        Params={"Bucket": "default-bucket", "Key": "k/a.pdf"},
        ExpiresIn=60,
    )


def test_presigned_url_failure_returns_none(storage, client):
    client.generate_presigned_url.side_effect = client_error("AccessDenied", "GeneratePresignedUrl")
    assert storage.generate_presigned_url(make_file(storage)) is None

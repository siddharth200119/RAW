import pytest
from RAW.utils.metrics import NullMetrics
from RAW.utils.storage import Storage


class DummyStorage(Storage):
    def open(self, file):
        raise NotImplementedError

    def save(self, file, data):
        raise NotImplementedError

    def delete(self, file):
        raise NotImplementedError

    def exists(self, file):
        return False

    def generate_presigned_url(self, file, *, expiration=3600, method="get_object"):
        return None


def test_storage_is_abstract():
    with pytest.raises(TypeError):
        Storage()


def test_defaults_to_null_metrics():
    assert isinstance(DummyStorage()._metrics, NullMetrics)


def test_timed_op_success_emits_counter_and_histogram(metrics):
    storage = DummyStorage(metrics=metrics)
    with storage._timed_op("save", direction="upload") as ctx:
        ctx.bytes_transferred = 42

    ops = metrics.named("raw.storage.operations")
    assert ops == [("increment", "raw.storage.operations", 1.0, {"operation": "save", "status": "success"})]

    durations = metrics.named("raw.storage.operation_duration_seconds")
    assert len(durations) == 1
    assert durations[0][2] >= 0
    assert durations[0][3] == {"operation": "save"}

    assert metrics.named("raw.storage.bytes_transferred") == [
        ("increment", "raw.storage.bytes_transferred", 42, {"direction": "upload"})
    ]


def test_timed_op_error_marks_status_and_reraises(metrics):
    storage = DummyStorage(metrics=metrics)
    with pytest.raises(RuntimeError):
        with storage._timed_op("open"):
            raise RuntimeError("boom")

    assert metrics.named("raw.storage.operations")[0][3] == {"operation": "open", "status": "error"}
    assert len(metrics.named("raw.storage.operation_duration_seconds")) == 1


def test_timed_op_skips_bytes_without_direction_or_bytes(metrics):
    storage = DummyStorage(metrics=metrics)
    with storage._timed_op("delete") as ctx:
        ctx.bytes_transferred = 10
    with storage._timed_op("open", direction="download"):
        pass

    assert metrics.named("raw.storage.bytes_transferred") == []

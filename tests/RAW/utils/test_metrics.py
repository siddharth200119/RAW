from RAW.utils.metrics import Metrics, NullMetrics


class RecordingMetrics:
    """Structurally satisfies Metrics without inheriting from it."""

    def increment(self, name, value=1.0, *, tags=None):
        pass

    def histogram(self, name, value, *, tags=None):
        pass

    def gauge(self, name, value, *, tags=None):
        pass


class Incomplete:
    def increment(self, name, value=1.0, *, tags=None):
        pass


def test_null_metrics_satisfies_protocol():
    assert isinstance(NullMetrics(), Metrics)


def test_null_metrics_methods_are_noops():
    m = NullMetrics()
    assert m.increment("x") is None
    assert m.increment("x", 5, tags={"a": "b"}) is None
    assert m.histogram("x", 0.5, tags={"a": "b"}) is None
    assert m.gauge("x", 3) is None


def test_null_metrics_has_no_instance_dict():
    assert not hasattr(NullMetrics(), "__dict__")


def test_duck_typed_class_satisfies_protocol():
    assert isinstance(RecordingMetrics(), Metrics)


def test_incomplete_class_does_not_satisfy_protocol():
    assert not isinstance(Incomplete(), Metrics)

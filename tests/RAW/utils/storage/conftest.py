import pytest


class RecordingMetrics:
    def __init__(self):
        self.calls = []

    def increment(self, name, value=1.0, *, tags=None):
        self.calls.append(("increment", name, value, tags))

    def histogram(self, name, value, *, tags=None):
        self.calls.append(("histogram", name, value, tags))

    def gauge(self, name, value, *, tags=None):
        self.calls.append(("gauge", name, value, tags))

    def named(self, name):
        return [c for c in self.calls if c[1] == name]


@pytest.fixture
def metrics():
    return RecordingMetrics()

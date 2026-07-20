"""Generic metrics abstraction for the RAW library.

Defines a lightweight :class:`Metrics` protocol that any subsystem (storage,
LLMs, agents, …) can use to emit counters, histograms, and gauges — without
coupling the library to a specific metrics backend.

**Usage (library internals):**

.. code-block:: python

    from RAW.metrics import Metrics, NullMetrics

    class MyComponent:
        def __init__(self, metrics: Metrics = NullMetrics()) -> None:
            self._m = metrics

        def do_work(self) -> None:
            self._m.increment("my_component.work_done", tags={"type": "heavy"})

**Usage (application side — Prometheus example):**

.. code-block:: python

    from prometheus_client import Counter, Histogram
    from RAW.metrics import Metrics

    class PrometheusMetrics:
        _ops     = Counter("raw_operations_total", "...", ["name", "status"])
        _dur     = Histogram("raw_operation_duration_seconds", "...", ["name"])
        _counter = Counter("raw_counter_total", "...", ["name"])

        def increment(self, name, value=1.0, *, tags=None):
            label = (tags or {}).get("status", "")
            self._counter.labels(name=name, status=label).inc(value)

        def histogram(self, name, value, *, tags=None):
            self._dur.labels(name=name).observe(value)

        def gauge(self, name, value, *, tags=None):
            pass  # implement with prometheus_client.Gauge if needed

    # Wire in:
    storage = S3Storage(client=client, metrics=PrometheusMetrics())
    llm     = GeminiLLM(..., metrics=PrometheusMetrics())
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Metrics(Protocol):
    """Structural protocol for emitting metrics from any RAW component.

    Implementors need only satisfy the three method signatures — no base class
    or registration is required.  The :class:`NullMetrics` no-op is the
    default so components never need to guard against ``None``.

    **Tag conventions** (recommended, not enforced):

    +-------------------+-------------------------------+
    | Tag key           | Typical values                |
    +===================+===============================+
    | ``operation``     | ``"save"``, ``"open"``, …     |
    +-------------------+-------------------------------+
    | ``status``        | ``"success"``, ``"error"``    |
    +-------------------+-------------------------------+
    | ``component``     | ``"storage"``, ``"llm"``, …   |
    +-------------------+-------------------------------+
    | ``model``         | ``"gemini-2.0-flash"``, …     |
    +-------------------+-------------------------------+
    | ``direction``     | ``"upload"``, ``"download"``  |
    +-------------------+-------------------------------+
    """

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric by *value*.

        Args:
            name:  Dot-separated metric name, e.g. ``"raw.storage.operations"``.
            value: Amount to add.  Defaults to ``1.0``.
            tags:  Arbitrary key/value labels for the metric.
        """
        ...

    def histogram(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram / distribution metric.

        Args:
            name:  Metric name, e.g. ``"raw.storage.operation_duration_seconds"``.
            value: Observed value (e.g. elapsed seconds, byte count).
            tags:  Arbitrary key/value labels.
        """
        ...

    def gauge(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric to an absolute *value*.

        Args:
            name:  Metric name, e.g. ``"raw.agent.queue_depth"``.
            value: Current absolute value.
            tags:  Arbitrary key/value labels.
        """
        ...


class NullMetrics:
    """No-op :class:`Metrics` implementation — the default for all components.

    Satisfies the :class:`Metrics` protocol without emitting anything.
    Use this when no metrics backend is configured, or in unit tests where
    metrics output is irrelevant.

    Example::

        assert isinstance(NullMetrics(), Metrics)  # True — satisfies protocol
    """

    __slots__ = ()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def histogram(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def gauge(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass
